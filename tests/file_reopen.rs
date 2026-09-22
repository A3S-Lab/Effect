use std::io::Write;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use a3s_effect::{
    answer_fact, coding_actor, confirm_fact, ingest_coding, message_fact, resume_coding,
    ActorError, CodingPhase, CodingServices, Compactor, Completion, CompletionRequest, Exit,
    FileLog, HarnessConfig, LogStore, ModelDecision, ToolCall, ToolRunner,
};

const LIMIT: u32 = 8;

struct RemovedDir(PathBuf);

impl RemovedDir {
    fn new(label: &str) -> Self {
        static N: AtomicUsize = AtomicUsize::new(0);
        let dir = std::env::temp_dir().join(format!(
            "a3s-effect-{label}-{}-{}",
            std::process::id(),
            N.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = std::fs::remove_dir_all(&dir);
        Self(dir)
    }
}

impl Drop for RemovedDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn text(text: &str) -> ModelDecision {
    ModelDecision::Text {
        text: text.to_string(),
    }
}

fn tool(id: &str, confirm: bool) -> ModelDecision {
    ModelDecision::Tool {
        call: ToolCall {
            id: id.to_string(),
            name: "read".into(),
            args: serde_json::json!({ "path": "src/lib.rs" }),
            needs_confirmation: confirm,
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

struct Session {
    dir: RemovedDir,
    services: Arc<CodingServices>,
    model_calls: Arc<AtomicUsize>,
    tool_calls: Arc<AtomicUsize>,
    compact_calls: Arc<AtomicUsize>,
    actor: a3s_effect::Actor<CodingServices, a3s_effect::CodingView>,
}

impl Session {
    fn open(
        label: &str,
        decisions: Vec<Result<ModelDecision, ActorError>>,
        fail_first: bool,
    ) -> Self {
        let model_calls = Arc::new(AtomicUsize::new(0));
        let tool_calls = Arc::new(AtomicUsize::new(0));
        let compact_calls = Arc::new(AtomicUsize::new(0));
        let config = HarnessConfig::new(2, 10_000, LIMIT, 1, Vec::new(), Vec::new()).unwrap();
        Self {
            dir: RemovedDir::new(label),
            services: Arc::new(CodingServices {
                completion: Arc::new(ScriptModel {
                    decisions: Mutex::new(decisions.into_iter().rev().collect()),
                    calls: Arc::clone(&model_calls),
                }),
                tools: Arc::new(CountingTools {
                    calls: Arc::clone(&tool_calls),
                    fail_first: AtomicUsize::new(usize::from(fail_first)),
                }),
                compactor: Arc::new(CountingCompactor {
                    calls: Arc::clone(&compact_calls),
                }),
            }),
            model_calls,
            tool_calls,
            compact_calls,
            actor: coding_actor(config),
        }
    }

    fn log(&self) -> FileLog {
        FileLog::open(&self.dir.0).expect("open")
    }

    fn jsonl(&self) -> PathBuf {
        self.dir.0.join("thread-1.jsonl")
    }
}

#[tokio::test]
async fn finished_text_turn_does_not_call_the_model_after_reopen() {
    let session = Session::open("text", vec![Ok(text("hello"))], false);
    let assistant = {
        let log = session.log();
        let settled = ingest_coding(
            &session.actor,
            &log,
            Arc::clone(&session.services),
            "thread-1",
            message_fact("m1", "hi"),
            LIMIT,
        )
        .await
        .expect("ingest");
        assert_eq!(settled.view.phase, CodingPhase::Done);
        assert_eq!(session.model_calls.load(Ordering::SeqCst), 1);
        settled.view.assistant.clone()
    };
    let log = session.log();
    let again = resume_coding(
        &session.actor,
        &log,
        Arc::clone(&session.services),
        "thread-1",
        LIMIT,
    )
    .await
    .expect("resume");
    assert_eq!(again.steps, 0);
    assert_eq!(again.view.phase, CodingPhase::Done);
    assert_eq!(again.view.assistant, assistant);
    assert_eq!(session.model_calls.load(Ordering::SeqCst), 1);
    assert_eq!(session.tool_calls.load(Ordering::SeqCst), 0);
    assert_eq!(session.compact_calls.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn parked_confirmation_survives_reopen_and_then_runs_once() {
    let session = Session::open(
        "confirm",
        vec![Ok(tool("t1", true)), Ok(text("done"))],
        false,
    );
    let id = {
        let log = session.log();
        let parked = ingest_coding(
            &session.actor,
            &log,
            Arc::clone(&session.services),
            "thread-1",
            message_fact("m1", "read it"),
            LIMIT,
        )
        .await
        .expect("park");
        assert_eq!(parked.view.phase, CodingPhase::Confirm);
        parked
            .view
            .pending_confirmation
            .expect("confirmation")
            .tool_call_id
    };
    let log = session.log();
    let still = resume_coding(
        &session.actor,
        &log,
        Arc::clone(&session.services),
        "thread-1",
        LIMIT,
    )
    .await
    .expect("parked resume");
    assert_eq!(still.steps, 0);
    assert_eq!(still.view.phase, CodingPhase::Confirm);
    assert_eq!(
        still
            .view
            .pending_confirmation
            .expect("confirmation")
            .tool_call_id,
        id
    );
    assert_eq!(session.model_calls.load(Ordering::SeqCst), 1);
    assert_eq!(session.tool_calls.load(Ordering::SeqCst), 0);
    assert_eq!(session.compact_calls.load(Ordering::SeqCst), 0);

    let settled = ingest_coding(
        &session.actor,
        &log,
        Arc::clone(&session.services),
        "thread-1",
        confirm_fact("c1", id, true),
        LIMIT,
    )
    .await
    .expect("confirm");
    assert_eq!(session.tool_calls.load(Ordering::SeqCst), 1);
    assert_eq!(session.model_calls.load(Ordering::SeqCst), 2);
    assert_eq!(settled.view.assistant.as_deref(), Some("done"));
}

#[tokio::test]
async fn parked_question_keeps_allow_free_text_across_reopen() {
    let session = Session::open(
        "question",
        vec![
            Ok(ModelDecision::Question {
                question_id: "q1".into(),
                question: "Which module?".into(),
                allow_free_text: true,
            }),
            Ok(text("scheduler")),
        ],
        false,
    );
    {
        let log = session.log();
        let parked = ingest_coding(
            &session.actor,
            &log,
            Arc::clone(&session.services),
            "thread-1",
            message_fact("m1", "look"),
            LIMIT,
        )
        .await
        .expect("park");
        assert_eq!(parked.view.phase, CodingPhase::Question);
        assert!(
            parked
                .view
                .pending_question
                .expect("question")
                .allow_free_text
        );
    }
    let log = session.log();
    let still = resume_coding(
        &session.actor,
        &log,
        Arc::clone(&session.services),
        "thread-1",
        LIMIT,
    )
    .await
    .expect("parked resume");
    assert_eq!(still.steps, 0);
    assert_eq!(still.view.phase, CodingPhase::Question);
    assert!(
        still
            .view
            .pending_question
            .expect("question")
            .allow_free_text
    );
    assert_eq!(session.model_calls.load(Ordering::SeqCst), 1);
    assert_eq!(session.tool_calls.load(Ordering::SeqCst), 0);
    assert_eq!(session.compact_calls.load(Ordering::SeqCst), 0);

    let settled = ingest_coding(
        &session.actor,
        &log,
        Arc::clone(&session.services),
        "thread-1",
        answer_fact("a1", "the scheduler"),
        LIMIT,
    )
    .await
    .expect("answer");
    assert_eq!(session.model_calls.load(Ordering::SeqCst), 2);
    assert_eq!(session.tool_calls.load(Ordering::SeqCst), 0);
    assert_eq!(settled.view.assistant.as_deref(), Some("scheduler"));
}

#[tokio::test]
async fn failed_tool_without_a_result_runs_once_after_reopen() {
    let session = Session::open(
        "tool-fail",
        vec![Ok(tool("t1", false)), Ok(text("after"))],
        true,
    );
    {
        let log = session.log();
        let failed = ingest_coding(
            &session.actor,
            &log,
            Arc::clone(&session.services),
            "thread-1",
            message_fact("m1", "go"),
            LIMIT,
        )
        .await;
        assert!(matches!(
            failed,
            Err(Exit::Fail(ActorError::Handler { .. }))
        ));
        let stored = log.read("thread-1").unwrap();
        assert!(stored.iter().any(|fact| fact.kind == "model.turn"));
        assert!(stored.iter().all(|fact| fact.kind != "tool.result"));
        assert_eq!(session.tool_calls.load(Ordering::SeqCst), 1);
    }
    {
        let log = session.log();
        let settled = resume_coding(
            &session.actor,
            &log,
            Arc::clone(&session.services),
            "thread-1",
            LIMIT,
        )
        .await
        .expect("retry tool");
        assert_eq!(session.tool_calls.load(Ordering::SeqCst), 2);
        assert_eq!(session.model_calls.load(Ordering::SeqCst), 2);
        assert_eq!(
            settled
                .log
                .iter()
                .filter(|fact| fact.kind == "tool.result")
                .count(),
            1
        );
    }
    let log = session.log();
    let quiet = resume_coding(
        &session.actor,
        &log,
        Arc::clone(&session.services),
        "thread-1",
        LIMIT,
    )
    .await
    .expect("third");
    assert_eq!(quiet.steps, 0);
    assert_eq!(session.tool_calls.load(Ordering::SeqCst), 2);
    assert_eq!(session.model_calls.load(Ordering::SeqCst), 2);
}

#[tokio::test]
async fn duplicate_fact_after_a_finished_turn_does_not_call_the_model() {
    let session = Session::open("duplicate", vec![Ok(text("hello"))], false);
    let bytes = {
        let log = session.log();
        let settled = ingest_coding(
            &session.actor,
            &log,
            Arc::clone(&session.services),
            "thread-1",
            message_fact("m1", "hi"),
            LIMIT,
        )
        .await
        .expect("turn");
        assert_eq!(settled.view.phase, CodingPhase::Done);
        assert_eq!(session.model_calls.load(Ordering::SeqCst), 1);
        let bytes = std::fs::read(session.jsonl()).unwrap();
        let error = ingest_coding(
            &session.actor,
            &log,
            Arc::clone(&session.services),
            "thread-1",
            message_fact("m1", "other"),
            LIMIT,
        )
        .await;
        assert!(matches!(
            error,
            Err(Exit::Fail(ActorError::DuplicateFact { key })) if key == "m1"
        ));
        assert_eq!(session.model_calls.load(Ordering::SeqCst), 1);
        assert_eq!(std::fs::read(session.jsonl()).unwrap(), bytes);
        bytes
    };
    let log = session.log();
    let stored = log.read("thread-1").unwrap();
    let message = stored.iter().find(|fact| fact.key == "m1").unwrap();
    assert_eq!(message.payload["text"], "hi");
    let again = resume_coding(
        &session.actor,
        &log,
        Arc::clone(&session.services),
        "thread-1",
        LIMIT,
    )
    .await
    .expect("resume");
    assert_eq!(again.steps, 0);
    assert_eq!(again.view.phase, CodingPhase::Done);
    assert_eq!(session.model_calls.load(Ordering::SeqCst), 1);
    assert_eq!(std::fs::read(session.jsonl()).unwrap(), bytes);
}

#[tokio::test]
async fn a_corrupt_jsonl_line_makes_resume_return_schema_without_appending() {
    let session = Session::open("corrupt", vec![], false);
    let bytes = {
        let log = session.log();
        log.append("thread-1", &[message_fact("m1", "hi")], None)
            .unwrap();
        drop(log);
        let mut file = std::fs::OpenOptions::new()
            .append(true)
            .open(session.jsonl())
            .unwrap();
        writeln!(file, "{{").unwrap();
        std::fs::read(session.jsonl()).unwrap()
    };
    let log = session.log();
    let error = resume_coding(
        &session.actor,
        &log,
        Arc::clone(&session.services),
        "thread-1",
        LIMIT,
    )
    .await;
    assert!(matches!(error, Err(Exit::Fail(ActorError::Schema(_)))));
    assert_eq!(std::fs::read(session.jsonl()).unwrap(), bytes);
    assert_eq!(session.model_calls.load(Ordering::SeqCst), 0);
}
