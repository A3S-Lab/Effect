use a3s_effect::{
    answer_fact, coding_actor, component, confirm_fact, cut_log, ingest, message_fact,
    parse_fact_json, Actor, ActorError, CodingPhase, CodingServices, Compactor, Completion,
    CompletionRequest, Effect, Exit, Fact, FileLog, HarnessConfig, LogStore, MemoryLog,
    ModelDecision, NewFact, ToolCall, ToolRunner, ToolSpec,
};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

fn text(text: &str) -> ModelDecision {
    ModelDecision::Text {
        text: text.to_string(),
    }
}

fn tool(id: &str, name: &str, confirm: bool) -> ModelDecision {
    ModelDecision::Tool {
        call: ToolCall {
            id: id.to_string(),
            name: name.to_string(),
            args: serde_json::json!({ "path": "src/lib.rs" }),
            needs_confirmation: confirm,
            text: None,
            reasoning: None,
        },
    }
}

struct ScriptModel {
    decisions: Mutex<Vec<Result<ModelDecision, ActorError>>>,
    calls: Arc<AtomicUsize>,
}

impl Completion for ScriptModel {
    fn complete(
        &self,
        _request: CompletionRequest,
    ) -> a3s_effect::coding::BoxFuture<Result<ModelDecision, ActorError>> {
        let decision = self
            .decisions
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .pop()
            .unwrap_or(Ok(text("empty")));
        self.calls.fetch_add(1, Ordering::SeqCst);
        Box::pin(async move { decision })
    }
}

struct CountingTools {
    calls: Arc<AtomicUsize>,
    fail_first: AtomicUsize,
}

impl ToolRunner for CountingTools {
    fn run(
        &self,
        call: ToolCall,
    ) -> a3s_effect::coding::BoxFuture<Result<serde_json::Value, ActorError>> {
        let n = self.calls.fetch_add(1, Ordering::SeqCst);
        let name = call.name;
        let fail = self.fail_first.load(Ordering::SeqCst) > 0 && n == 0;
        Box::pin(async move {
            if fail {
                Err(ActorError::Handler {
                    key: name,
                    message: "down".into(),
                })
            } else {
                Ok(serde_json::json!(format!("ran {name}")))
            }
        })
    }
}

struct CountingCompactor {
    calls: Arc<AtomicUsize>,
}

impl Compactor for CountingCompactor {
    fn compact(
        &self,
        messages: &[String],
    ) -> a3s_effect::coding::BoxFuture<Result<String, ActorError>> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        let summary = messages.join(" ");
        Box::pin(async move { Ok(format!("summary:{summary}")) })
    }
}

struct Harness {
    services: Arc<CodingServices>,
    model_calls: Arc<AtomicUsize>,
    tool_calls: Arc<AtomicUsize>,
    compact_calls: Arc<AtomicUsize>,
    log: MemoryLog,
    actor: a3s_effect::Actor<CodingServices, a3s_effect::CodingView>,
    limit: u32,
}

fn harness(
    decisions: Vec<Result<ModelDecision, ActorError>>,
    budget: u32,
    compact_after: usize,
    attempts: u32,
    fail_first_tool: bool,
) -> Harness {
    let model_calls = Arc::new(AtomicUsize::new(0));
    let tool_calls = Arc::new(AtomicUsize::new(0));
    let compact_calls = Arc::new(AtomicUsize::new(0));
    let model = ScriptModel {
        decisions: Mutex::new(decisions.into_iter().rev().collect()),
        calls: Arc::clone(&model_calls),
    };
    let tools = CountingTools {
        calls: Arc::clone(&tool_calls),
        fail_first: AtomicUsize::new(u8::from(fail_first_tool) as usize),
    };
    let compactor = CountingCompactor {
        calls: Arc::clone(&compact_calls),
    };
    let config = HarnessConfig::new(
        budget,
        compact_after,
        16,
        attempts,
        vec!["You are a coding harness.".into()],
        vec![ToolSpec {
            name: "read".into(),
            description: "Read a file".into(),
        }],
    )
    .expect("config");
    let limit = config.step_limit();
    Harness {
        services: Arc::new(CodingServices {
            completion: Arc::new(model),
            tools: Arc::new(tools),
            compactor: Arc::new(compactor),
        }),
        model_calls,
        tool_calls,
        compact_calls,
        log: MemoryLog::new(),
        actor: coding_actor(config),
        limit,
    }
}

async fn send(
    harness: &Harness,
    key: &str,
    text: &str,
) -> a3s_effect::Settlement<a3s_effect::CodingView> {
    a3s_effect::ingest_coding(
        &harness.actor,
        &harness.log,
        Arc::clone(&harness.services),
        "thread-1",
        message_fact(key, text),
        harness.limit,
    )
    .await
    .expect("ingest")
}

#[tokio::test]
async fn a_text_turn_runs_once_and_resume_does_not_call_the_model_again() {
    let harness = harness(vec![Ok(text("hello"))], 2, 10_000, 1, false);
    let settled = send(&harness, "m1", "hi").await;
    assert_eq!(settled.view.assistant.as_deref(), Some("hello"));
    assert_eq!(settled.view.phase, CodingPhase::Done);
    assert_eq!(harness.model_calls.load(Ordering::SeqCst), 1);

    let again = a3s_effect::resume_coding(
        &harness.actor,
        &harness.log,
        Arc::clone(&harness.services),
        "thread-1",
        harness.limit,
    )
    .await
    .expect("resume");
    assert_eq!(again.steps, 0);
    assert_eq!(harness.model_calls.load(Ordering::SeqCst), 1);
    assert_eq!(cut_log(&again.log).seq, settled.cut.seq);
}

#[tokio::test]
async fn confirmation_parks_until_a_fact_and_denial_does_not_run_the_tool() {
    let harness = harness(vec![Ok(tool("t1", "read", true))], 2, 10_000, 1, false);
    let parked = send(&harness, "m1", "read it").await;
    assert_eq!(parked.view.phase, CodingPhase::Confirm);
    assert_eq!(
        parked
            .view
            .pending_confirmation
            .as_ref()
            .map(|item| item.tool_call_id.as_str()),
        Some("t1")
    );
    assert_eq!(harness.tool_calls.load(Ordering::SeqCst), 0);

    let denied = a3s_effect::ingest_coding(
        &harness.actor,
        &harness.log,
        Arc::clone(&harness.services),
        "thread-1",
        confirm_fact("c1", "t1", false),
        harness.limit,
    )
    .await
    .expect("deny");
    assert_eq!(denied.view.assistant.as_deref(), Some("denied"));
    assert_eq!(harness.tool_calls.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn an_approved_tool_runs_once_across_resume() {
    let harness = harness(
        vec![Ok(tool("t1", "read", true)), Ok(text("done"))],
        2,
        10_000,
        1,
        false,
    );
    send(&harness, "m1", "read it").await;
    let settled = a3s_effect::ingest_coding(
        &harness.actor,
        &harness.log,
        Arc::clone(&harness.services),
        "thread-1",
        confirm_fact("c1", "t1", true),
        harness.limit,
    )
    .await
    .expect("approve");
    assert_eq!(settled.view.assistant.as_deref(), Some("done"));
    assert_eq!(harness.tool_calls.load(Ordering::SeqCst), 1);
    a3s_effect::resume_coding(
        &harness.actor,
        &harness.log,
        Arc::clone(&harness.services),
        "thread-1",
        harness.limit,
    )
    .await
    .expect("resume");
    assert_eq!(harness.tool_calls.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn a_failed_tool_is_not_recorded_and_resume_runs_it_once() {
    let harness = harness(
        vec![Ok(tool("t1", "read", false)), Ok(text("after"))],
        2,
        10_000,
        1,
        true,
    );
    let failed = a3s_effect::ingest_coding(
        &harness.actor,
        &harness.log,
        Arc::clone(&harness.services),
        "thread-1",
        message_fact("m1", "go"),
        harness.limit,
    )
    .await;
    assert!(matches!(
        failed,
        Err(Exit::Fail(ActorError::Handler { .. }))
    ));
    assert_eq!(harness.tool_calls.load(Ordering::SeqCst), 1);
    assert!(harness
        .log
        .read("thread-1")
        .unwrap()
        .iter()
        .all(|fact| fact.kind != "tool.result"));

    let settled = a3s_effect::resume_coding(
        &harness.actor,
        &harness.log,
        Arc::clone(&harness.services),
        "thread-1",
        harness.limit,
    )
    .await
    .expect("resume");
    assert_eq!(settled.view.assistant.as_deref(), Some("after"));
    assert_eq!(harness.tool_calls.load(Ordering::SeqCst), 2);
    assert_eq!(
        settled
            .log
            .iter()
            .filter(|fact| fact.kind == "tool.result")
            .count(),
        1
    );

    a3s_effect::resume_coding(
        &harness.actor,
        &harness.log,
        Arc::clone(&harness.services),
        "thread-1",
        harness.limit,
    )
    .await
    .expect("second resume");
    assert_eq!(harness.tool_calls.load(Ordering::SeqCst), 2);
}

#[tokio::test]
async fn budget_stops_a_second_tool_without_running_it() {
    let harness = harness(
        vec![Ok(tool("t1", "read", false)), Ok(tool("t2", "read", false))],
        1,
        10_000,
        1,
        false,
    );
    let settled = send(&harness, "m1", "edit").await;
    assert_eq!(settled.view.assistant.as_deref(), Some("budget"));
    assert_eq!(harness.tool_calls.load(Ordering::SeqCst), 1);
    assert!(settled.log.iter().any(|fact| fact.kind == "budget.denied"));
}

#[tokio::test]
async fn compaction_runs_once_before_inference() {
    let harness = harness(vec![Ok(text("short"))], 2, 5, 1, false);
    let settled = send(&harness, "m1", "hello").await;
    assert_eq!(settled.view.assistant.as_deref(), Some("short"));
    assert_eq!(harness.compact_calls.load(Ordering::SeqCst), 1);
    assert!(settled
        .log
        .iter()
        .any(|fact| fact.kind == "compaction.done"));
    a3s_effect::resume_coding(
        &harness.actor,
        &harness.log,
        Arc::clone(&harness.services),
        "thread-1",
        harness.limit,
    )
    .await
    .expect("resume");
    assert_eq!(harness.compact_calls.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn a_question_stays_parked_with_allow_free_text_until_an_answer_fact() {
    let harness = harness(
        vec![
            Ok(ModelDecision::Question {
                question_id: "q1".into(),
                question: "Which module?".into(),
                allow_free_text: true,
                options: vec!["scheduler".into()],
            }),
            Ok(text("scheduler")),
        ],
        2,
        10_000,
        1,
        false,
    );
    let parked = send(&harness, "m1", "look").await;
    assert_eq!(parked.view.phase, CodingPhase::Question);
    let question = parked.view.pending_question.expect("question");
    assert!(question.allow_free_text);
    assert_eq!(question.question_id, "q1");
    assert_eq!(harness.model_calls.load(Ordering::SeqCst), 1);

    let still = a3s_effect::resume_coding(
        &harness.actor,
        &harness.log,
        Arc::clone(&harness.services),
        "thread-1",
        harness.limit,
    )
    .await
    .expect("parked resume");
    assert_eq!(still.steps, 0);
    assert_eq!(harness.model_calls.load(Ordering::SeqCst), 1);

    let settled = a3s_effect::ingest_coding(
        &harness.actor,
        &harness.log,
        Arc::clone(&harness.services),
        "thread-1",
        answer_fact("a1", "the scheduler"),
        harness.limit,
    )
    .await
    .expect("answer");
    assert_eq!(settled.view.assistant.as_deref(), Some("scheduler"));
    assert_eq!(harness.model_calls.load(Ordering::SeqCst), 2);
}

#[tokio::test]
async fn replaying_an_ingress_key_does_not_run_the_model_twice() {
    let harness = harness(vec![Ok(text("once"))], 2, 10_000, 1, false);
    send(&harness, "m1", "hi").await;
    send(&harness, "m1", "hi").await;
    assert_eq!(harness.model_calls.load(Ordering::SeqCst), 1);
    assert_eq!(
        harness
            .log
            .read("thread-1")
            .unwrap()
            .iter()
            .filter(|fact| fact.kind == "user.message")
            .count(),
        1
    );
}

#[tokio::test]
async fn duplicate_transition_keys_fail_before_any_effect_runs() {
    let calls = Arc::new(AtomicUsize::new(0));
    let left_calls = Arc::clone(&calls);
    let right_calls = Arc::clone(&calls);
    let left = component(
        || (),
        |state, _fact: &Fact| state,
        move |_state| {
            let left_calls = Arc::clone(&left_calls);
            (
                (),
                vec![a3s_effect::Transition {
                    key: "same".into(),
                    run: Effect::from_async(move |_services, _cancel| {
                        let left_calls = Arc::clone(&left_calls);
                        async move {
                            left_calls.fetch_add(1, Ordering::SeqCst);
                            Ok(vec![NewFact {
                                kind: "tick".into(),
                                key: "tick-left".into(),
                                payload: serde_json::json!({}),
                            }])
                        }
                    }),
                }],
            )
        },
    );
    let right = component(
        || (),
        |state, _fact: &Fact| state,
        move |_state| {
            let right_calls = Arc::clone(&right_calls);
            (
                (),
                vec![a3s_effect::Transition {
                    key: "same".into(),
                    run: Effect::from_async(move |_services, _cancel| {
                        let right_calls = Arc::clone(&right_calls);
                        async move {
                            right_calls.fetch_add(1, Ordering::SeqCst);
                            Ok(vec![NewFact {
                                kind: "tick".into(),
                                key: "tick-right".into(),
                                payload: serde_json::json!({}),
                            }])
                        }
                    }),
                }],
            )
        },
    );
    let actor = Actor::new("dup", vec![left, right], |_views| ());
    let log = MemoryLog::new();
    let error = ingest(
        &actor,
        &log,
        Arc::new(()),
        "thread-1",
        message_fact("m1", "go"),
        4,
    )
    .await;
    assert!(matches!(
        error,
        Err(Exit::Fail(ActorError::DuplicateTransition { .. }))
    ));
    assert_eq!(calls.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn file_log_recovery_runs_a_recorded_tool_request_once() {
    let dir = std::env::temp_dir().join(format!("effect-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let log = FileLog::open(&dir).expect("open");
    let tools = Arc::new(CountingTools {
        calls: Arc::new(AtomicUsize::new(0)),
        fail_first: AtomicUsize::new(0),
    });
    let services = Arc::new(CodingServices {
        completion: Arc::new(ScriptModel {
            decisions: Mutex::new(vec![Ok(text("after"))]),
            calls: Arc::new(AtomicUsize::new(0)),
        }),
        tools: Arc::clone(&tools) as Arc<dyn ToolRunner>,
        compactor: Arc::new(CountingCompactor {
            calls: Arc::new(AtomicUsize::new(0)),
        }),
    });
    let config = HarnessConfig::new(2, 10_000, 8, 1, Vec::new(), Vec::new()).unwrap();
    let actor = coding_actor(config);
    log.append(
        "recover",
        &[
            message_fact("m1", "go"),
            NewFact {
                kind: "model.turn".into(),
                key: "model:1:0".into(),
                payload: serde_json::to_value(tool("t1", "read", false)).unwrap(),
            },
        ],
        None,
    )
    .unwrap();

    let settled = a3s_effect::resume_coding(&actor, &log, Arc::clone(&services), "recover", 8)
        .await
        .expect("resume");
    assert_eq!(tools.calls.load(Ordering::SeqCst), 1);
    assert_eq!(settled.view.assistant.as_deref(), Some("after"));

    let reopened = FileLog::open(&dir).expect("reopen");
    a3s_effect::resume_coding(&actor, &reopened, Arc::clone(&services), "recover", 8)
        .await
        .expect("second");
    assert_eq!(tools.calls.load(Ordering::SeqCst), 1);
    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test]
async fn a_corrupt_log_line_is_a_schema_error() {
    let error = parse_fact_json("{\"seq\":1,\"type\":\"user.message\"}").unwrap_err();
    assert!(matches!(error, ActorError::Schema(_)));
    let fact = parse_fact_json(
        r#"{"seq":1,"type":"user.message","key":"m1","cause":null,"payload":{"text":"hi"}}"#,
    )
    .unwrap();
    assert_eq!(fact.kind, "user.message");
    assert_eq!(fact.key, "m1");
}

#[tokio::test]
async fn infer_fail_invokes_completion_model_attempts_times() {
    for attempts in [2_u32, 1] {
        let harness = harness(
            vec![Err(ActorError::Config("m".into())); attempts as usize],
            2,
            10_000,
            attempts,
            false,
        );
        let error = a3s_effect::ingest_coding(
            &harness.actor,
            &harness.log,
            Arc::clone(&harness.services),
            "thread-1",
            message_fact("m1", "hi"),
            harness.limit,
        )
        .await;
        assert!(matches!(error, Err(Exit::Fail(ActorError::Config(message))) if message == "m"));
        assert_eq!(
            harness.model_calls.load(Ordering::SeqCst),
            attempts as usize
        );
        assert_eq!(harness.tool_calls.load(Ordering::SeqCst), 0);
        let stored = harness.log.read("thread-1").unwrap();
        assert!(stored.iter().any(|fact| fact.kind == "user.message"));
        assert!(stored.iter().all(|fact| fact.kind != "model.turn"));
    }
}

#[tokio::test]
async fn a_model_turn_that_is_not_a_decision_settles_done() {
    let harness = harness(vec![], 2, 10_000, 1, false);
    harness
        .log
        .append("thread-1", &[message_fact("m1", "hi")], None)
        .unwrap();
    harness
        .log
        .append(
            "thread-1",
            &[NewFact {
                kind: "model.turn".into(),
                key: "model:1:0".into(),
                payload: serde_json::json!({ "kind": "nope" }),
            }],
            None,
        )
        .unwrap();
    let before = harness.log.read("thread-1").unwrap();
    let line = serde_json::to_string(before.last().unwrap()).unwrap();
    parse_fact_json(&line).expect("model.turn is a valid fact line");
    let settled = a3s_effect::resume_coding(
        &harness.actor,
        &harness.log,
        Arc::clone(&harness.services),
        "thread-1",
        harness.limit,
    )
    .await
    .expect("resume");
    assert_eq!(settled.view.phase, CodingPhase::Done);
    assert!(settled.view.schema_error.is_some());
    assert_eq!(harness.model_calls.load(Ordering::SeqCst), 0);
    assert_eq!(harness.tool_calls.load(Ordering::SeqCst), 0);
    assert_eq!(settled.log.len(), before.len());
}
