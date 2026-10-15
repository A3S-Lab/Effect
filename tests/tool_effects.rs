#![cfg(test)]

//! One fault campaign exercised through both authoritative log adapters.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use a3s_effect::*;
use serde_json::{json, Value};
use tokio_util::sync::CancellationToken;

const THREAD: &str = "receipt-thread";
const LIMIT: u32 = 32;

struct ScriptModel(Mutex<VecDeque<ModelDecision>>);
impl Completion for ScriptModel {
    fn complete(
        &self,
        _request: CompletionRequest,
        _cancel: CancellationToken,
    ) -> a3s_effect::coding::BoxFuture<Result<ModelDecision, ActorError>> {
        let result = self
            .0
            .lock()
            .unwrap()
            .pop_front()
            .unwrap_or(ModelDecision::Text {
                text: "done".into(),
            });
        Box::pin(async move { Ok(result) })
    }
}
struct NoCompact;
impl Compactor for NoCompact {
    fn compact(
        &self,
        _messages: &[String],
    ) -> a3s_effect::coding::BoxFuture<Result<String, ActorError>> {
        Box::pin(async { Ok("summary".into()) })
    }
}

struct HostTools {
    executions: Arc<AtomicUsize>,
    parallel: bool,
}
impl ToolRunner for HostTools {
    fn run(
        &self,
        _call: ToolCall,
        _cancel: CancellationToken,
    ) -> a3s_effect::coding::BoxFuture<Result<Value, ActorError>> {
        unreachable!("the typed adapter executes only its prepared effect")
    }

    fn prepare(
        self: Arc<Self>,
        request: ToolPreparationRequest,
        _cancel: CancellationToken,
    ) -> a3s_effect::coding::BoxFuture<Result<PreparedToolExecution, ActorError>> {
        Box::pin(async move {
            let mut identity = request.identity;
            identity.admission = Some(ToolAdmissionIdentityV1 {
                run_id: "exact-run".into(),
                thread_id: THREAD.into(),
                capability_binding_digest: format!("sha256:{}", "a".repeat(64)),
            });
            let original_args_digest = tool_args_digest(&request.call.args)?;
            let mut effective = request.call.args.clone();
            if effective.get("hook_rewrite") == Some(&json!(true)) {
                effective["path"] = json!("hook-resolved.txt");
            }
            let prepared = ToolPreparedV1 {
                schema: TOOL_PREPARED_SCHEMA.into(),
                identity,
                tool_name: request.call.name,
                original_args_digest,
                effective_args_digest: tool_args_digest(&effective)?,
                recovery: ToolRecoveryPolicy::NeverReplay,
                baseline: Some(json!({"version": 1})),
            };
            let for_execute = prepared.clone();
            let calls = Arc::clone(&self.executions);
            let execute = Effect::from_async_observed(move |_services, cancel| {
                let calls = Arc::clone(&calls);
                let prepared = for_execute.clone();
                let args = effective.clone();
                async move {
                    calls.fetch_add(1, Ordering::SeqCst);
                    if let Some(file) = args.get("fixture_file").and_then(Value::as_str) {
                        std::fs::write(file, b"independent workspace effect").unwrap();
                    }
                    if args.get("runner_error") == Some(&json!(true)) {
                        return Err(Exit::Fail(ActorError::Handler {
                            key: "host".into(),
                            message: "after possible effect".into(),
                        }));
                    }
                    if args.get("cancel_after_write") == Some(&json!(true)) {
                        cancel.cancel();
                    }
                    let path = args["path"].as_str().unwrap_or("output.txt").to_string();
                    let mut result = ToolExecutionResultV1 {
                        schema: TOOL_RESULT_SCHEMA.into(),
                        identity: prepared.identity.clone(),
                        prepared_digest: prepared.digest().map_err(Exit::Fail)?,
                        outcome: if args.get("exit_failure") == Some(&json!(true)) {
                            ToolExecutionOutcome::Failed
                        } else {
                            ToolExecutionOutcome::Succeeded
                        },
                        exit_code: Some(if args.get("exit_failure") == Some(&json!(true)) {
                            7
                        } else {
                            0
                        }),
                        output: json!(format!("observed {path}")),
                        effects: ToolEffectObservationV1::complete(vec![ToolPathChangeV1 {
                            path,
                            before: None,
                            after: ToolFileIdentityV1 {
                                kind: ToolFileKind::File,
                                content_digest: Some(format!("sha256:{}", "b".repeat(64))),
                                mode: Some(0o644),
                            },
                        }]),
                    };
                    if args.get("bad_identity") == Some(&json!(true)) {
                        result.identity.turn += 1;
                    }
                    Ok(result)
                }
            });
            Ok(PreparedToolExecution { prepared, execute })
        })
    }
    fn can_parallelize(&self, _calls: &[ToolCall]) -> bool {
        self.parallel
    }
}

struct Case {
    log: Arc<dyn LogStore>,
    dir: Option<tempfile::TempDir>,
}
impl Case {
    fn memory() -> Self {
        Self {
            log: Arc::new(MemoryLog::new()),
            dir: None,
        }
    }
    fn file() -> Self {
        let dir = tempfile::tempdir().unwrap();
        Self {
            log: Arc::new(FileLog::open(dir.path()).unwrap()),
            dir: Some(dir),
        }
    }
    fn reopen(&self) -> Arc<dyn LogStore> {
        match &self.dir {
            Some(dir) => Arc::new(FileLog::open(dir.path()).unwrap()),
            None => Arc::clone(&self.log),
        }
    }
}

struct RefuseAppend {
    inner: Arc<dyn LogStore>,
    kind: &'static str,
}
impl LogStore for RefuseAppend {
    fn read(&self, thread: &str) -> Result<Vec<Fact>, ActorError> {
        self.inner.read(thread)
    }
    fn append(
        &self,
        thread: &str,
        facts: &[NewFact],
        cause: Option<&str>,
    ) -> Result<Vec<Fact>, ActorError> {
        if facts.iter().any(|fact| fact.kind == self.kind) {
            return Err(ActorError::Handler {
                key: self.kind.into(),
                message: "injected append refusal".into(),
            });
        }
        self.inner.append(thread, facts, cause)
    }
}

struct CommitTerminalPrefix {
    inner: Arc<dyn LogStore>,
}
impl LogStore for CommitTerminalPrefix {
    fn read(&self, thread: &str) -> Result<Vec<Fact>, ActorError> {
        self.inner.read(thread)
    }
    fn append(
        &self,
        thread: &str,
        facts: &[NewFact],
        cause: Option<&str>,
    ) -> Result<Vec<Fact>, ActorError> {
        if facts.len() > 1 && facts.iter().all(|fact| fact.kind == "tool.result") {
            self.inner.append(thread, &facts[..1], cause)?;
            return Err(ActorError::Handler {
                key: "terminal-prefix".into(),
                message: "injected IO failure after one committed terminal".into(),
            });
        }
        self.inner.append(thread, facts, cause)
    }
}

fn call(id: &str, args: Value) -> ToolCall {
    ToolCall {
        id: id.into(),
        name: "host-tool".into(),
        args,
        needs_confirmation: false,
        text: None,
        reasoning: None,
    }
}
fn fixture(
    calls: Vec<ToolCall>,
    parallel: bool,
) -> (
    Actor<CodingServices, CodingView>,
    Arc<CodingServices>,
    Arc<AtomicUsize>,
) {
    let mut calls = calls.into_iter();
    let first = calls.next().unwrap();
    let model = ScriptModel(Mutex::new(VecDeque::from([
        ModelDecision::Tool {
            call: first,
            also: calls.collect(),
        },
        ModelDecision::Text {
            text: "done".into(),
        },
    ])));
    let executions = Arc::new(AtomicUsize::new(0));
    let services = Arc::new(CodingServices {
        completion: Arc::new(model),
        tools: Arc::new(HostTools {
            executions: Arc::clone(&executions),
            parallel,
        }),
        compactor: Arc::new(NoCompact),
    });
    let config = HarnessConfig::new(20, usize::MAX, LIMIT, 1, vec![], vec![]).unwrap();
    (coding_actor(config), services, executions)
}
fn receipts(facts: &[Fact]) -> Vec<ToolExecutionResultV1> {
    facts
        .iter()
        .filter_map(|fact| fact.payload.get("receipt"))
        .map(|value| serde_json::from_value(value.clone()).unwrap())
        .collect()
}

async fn append_fault_campaign(case: Case) {
    let workspace = tempfile::tempdir().unwrap();
    let file = workspace.path().join("effect.txt");
    let (actor, services, count) = fixture(
        vec![call("one", json!({"path":"one.txt", "fixture_file":file}))],
        false,
    );
    let refuse = RefuseAppend {
        inner: Arc::clone(&case.log),
        kind: "tool.prepared",
    };
    let result = ingest_coding(
        &actor,
        &refuse,
        Arc::clone(&services),
        THREAD,
        message_fact("m1", "do it"),
        LIMIT,
    )
    .await;
    assert!(
        matches!(result, Err(Exit::Fail(ActorError::Handler { key, .. })) if key == "tool.prepared")
    );
    assert_eq!(
        count.load(Ordering::SeqCst),
        0,
        "prepare append failure must execute zero tools"
    );
    assert!(!file.exists());
    assert!(case
        .log
        .read(THREAD)
        .unwrap()
        .iter()
        .all(|fact| fact.kind != "tool.prepared"));

    let refuse = RefuseAppend {
        inner: Arc::clone(&case.log),
        kind: "tool.result",
    };
    let result = resume_coding(&actor, &refuse, Arc::clone(&services), THREAD, LIMIT).await;
    assert!(
        matches!(result, Err(Exit::Fail(ActorError::Handler { key, .. })) if key == "tool.result")
    );
    assert_eq!(count.load(Ordering::SeqCst), 1);
    assert_eq!(
        std::fs::read(&file).unwrap(),
        b"independent workspace effect"
    );
    let stored = case.log.read(THREAD).unwrap();
    assert!(stored.iter().any(|fact| fact.kind == "tool.prepared"));
    assert!(stored.iter().all(|fact| fact.kind != "tool.result"));
    let reopened = case.reopen();
    let settled = resume_coding(
        &actor,
        reopened.as_ref(),
        Arc::clone(&services),
        THREAD,
        LIMIT,
    )
    .await
    .unwrap();
    assert_eq!(
        count.load(Ordering::SeqCst),
        1,
        "unknown attempt must never execute again"
    );
    let results = receipts(&settled.log);
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].outcome, ToolExecutionOutcome::Unknown);
    assert_eq!(
        results[0].effects.completeness,
        ToolObservationCompleteness::Incomplete
    );
    let again = resume_coding(&actor, reopened.as_ref(), services, THREAD, LIMIT)
        .await
        .unwrap();
    assert_eq!(again.steps, 0);
    assert_eq!(count.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn append_faults_do_not_start_or_repeat_effects_on_both_logs() {
    append_fault_campaign(Case::memory()).await;
    append_fault_campaign(Case::file()).await;
}

async fn terminal_prefix_campaign(case: Case, parallel: bool) {
    let workspace = tempfile::tempdir().unwrap();
    let files: Vec<_> = (0..3)
        .map(|index| workspace.path().join(format!("effect-{index}.txt")))
        .collect();
    let calls = files
        .iter()
        .enumerate()
        .map(|(index, file)| {
            call(
                &format!("call-{index}"),
                json!({"path":format!("effect-{index}.txt"), "fixture_file":file}),
            )
        })
        .collect();
    let (actor, services, count) = fixture(calls, parallel);
    let prefix = CommitTerminalPrefix {
        inner: Arc::clone(&case.log),
    };
    let result = ingest_coding(
        &actor,
        &prefix,
        Arc::clone(&services),
        THREAD,
        message_fact("m1", "perform three effects"),
        LIMIT,
    )
    .await;
    assert!(
        matches!(result, Err(Exit::Fail(ActorError::Handler { key, .. })) if key == "terminal-prefix")
    );
    assert_eq!(count.load(Ordering::SeqCst), 3);
    for file in &files {
        assert_eq!(
            std::fs::read(file).unwrap(),
            b"independent workspace effect"
        );
    }
    let committed = case.log.read(THREAD).unwrap();
    let first = receipts(&committed);
    assert_eq!(first.len(), 1);
    assert_eq!(first[0].outcome, ToolExecutionOutcome::Succeeded);
    assert!(committed
        .iter()
        .find(|fact| fact.kind == "tool.result")
        .unwrap()
        .cause
        .as_deref()
        .unwrap()
        .starts_with("tool:"));

    let reopened = case.reopen();
    let settled = resume_coding(
        &actor,
        reopened.as_ref(),
        Arc::clone(&services),
        THREAD,
        LIMIT,
    )
    .await
    .unwrap();
    let results = receipts(&settled.log);
    assert_eq!(
        results.len(),
        3,
        "a committed cause must not mask remaining preparations"
    );
    assert_eq!(results[0].outcome, ToolExecutionOutcome::Succeeded);
    assert!(results[1..]
        .iter()
        .all(|receipt| receipt.outcome == ToolExecutionOutcome::Unknown));
    assert_eq!(
        count.load(Ordering::SeqCst),
        3,
        "recovery must never repeat an effect"
    );
    let again = resume_coding(&actor, reopened.as_ref(), services, THREAD, LIMIT)
        .await
        .unwrap();
    assert_eq!(again.steps, 0);
    assert_eq!(again.log.len(), settled.log.len());
    assert_eq!(count.load(Ordering::SeqCst), 3);
}

#[tokio::test]
async fn committed_terminal_prefix_does_not_hide_prepared_siblings_on_both_logs() {
    for parallel in [false, true] {
        terminal_prefix_campaign(Case::memory(), parallel).await;
        terminal_prefix_campaign(Case::file(), parallel).await;
    }
}

async fn prepared_without_execution_campaign(case: Case) {
    let (actor, services, count) = fixture(vec![call("one", json!({"path":"one.txt"}))], false);
    let decision = ModelDecision::Tool {
        call: call("one", json!({"path":"one.txt"})),
        also: vec![],
    };
    case.log
        .append(
            THREAD,
            &[
                message_fact("m1", "do it"),
                NewFact {
                    kind: "model.turn".into(),
                    key: "model:1:0".into(),
                    payload: serde_json::to_value(decision).unwrap(),
                },
            ],
            None,
        )
        .unwrap();
    // This model decision is already present in the manually built crash log.
    services
        .completion
        .complete(
            CompletionRequest {
                system: vec![],
                tools: vec![],
                summary: String::new(),
                messages: vec![],
            },
            CancellationToken::new(),
        )
        .await
        .unwrap();
    let prepared = Arc::clone(&services.tools)
        .prepare(
            ToolPreparationRequest {
                identity: ToolInvocationIdentityV1 {
                    turn: 1,
                    cycle: 0,
                    tool_call_id: "one".into(),
                    batch_index: 0,
                    admission: None,
                },
                call: call("one", json!({"path":"one.txt"})),
            },
            CancellationToken::new(),
        )
        .await
        .unwrap();
    case.log
        .append(
            THREAD,
            &[NewFact {
                kind: "tool.prepared".into(),
                key: "prepared-only".into(),
                payload: json!({"prepared": prepared.prepared}),
            }],
            None,
        )
        .unwrap();
    let reopened = case.reopen();
    let settled = resume_coding(&actor, reopened.as_ref(), services, THREAD, LIMIT)
        .await
        .unwrap();
    assert_eq!(count.load(Ordering::SeqCst), 0);
    assert_eq!(
        receipts(&settled.log)[0].outcome,
        ToolExecutionOutcome::Unknown
    );
}
#[tokio::test]
async fn prepared_only_is_unknown_without_execution_on_both_logs() {
    prepared_without_execution_campaign(Case::memory()).await;
    prepared_without_execution_campaign(Case::file()).await;
}

async fn partial_batch_campaign(case: Case, parallel: bool) {
    let (actor, services, count) = fixture(
        vec![
            call("first", json!({"path":"first.txt", "exit_failure":true})),
            call("second", json!({"path":"second.txt", "runner_error":true})),
        ],
        parallel,
    );
    let settled = ingest_coding(
        &actor,
        case.log.as_ref(),
        Arc::clone(&services),
        THREAD,
        message_fact("m1", "do it"),
        LIMIT,
    )
    .await
    .unwrap();
    let results = receipts(&settled.log);
    assert_eq!(results.len(), 2);
    assert_eq!(results[0].outcome, ToolExecutionOutcome::Failed);
    assert_eq!(results[0].exit_code, Some(7));
    assert_eq!(results[0].effects.changes[0].path, "first.txt");
    assert_eq!(results[1].outcome, ToolExecutionOutcome::Unknown);
    let reopened = case.reopen();
    let again = resume_coding(&actor, reopened.as_ref(), services, THREAD, LIMIT)
        .await
        .unwrap();
    assert_eq!(again.steps, 0);
    assert_eq!(count.load(Ordering::SeqCst), 2);
    assert_eq!(receipts(&again.log), results);
}
#[tokio::test]
async fn partial_sequential_and_parallel_results_survive_on_both_logs() {
    for parallel in [false, true] {
        partial_batch_campaign(Case::memory(), parallel).await;
        partial_batch_campaign(Case::file(), parallel).await;
    }
}

async fn cancellation_campaign(case: Case) {
    let (actor, services, count) = fixture(
        vec![
            call(
                "first",
                json!({"path":"first.txt", "cancel_after_write":true}),
            ),
            call("second", json!({"path":"second.txt"})),
        ],
        false,
    );
    let settled = ingest_coding_with_cancel(
        &actor,
        case.log.as_ref(),
        services,
        THREAD,
        message_fact("m1", "do it"),
        LIMIT,
        &CancellationToken::new(),
    )
    .await
    .unwrap();
    let results = receipts(&settled.log);
    assert_eq!(count.load(Ordering::SeqCst), 1);
    assert_eq!(results[0].outcome, ToolExecutionOutcome::Succeeded);
    assert_eq!(results[0].effects.changes[0].path, "first.txt");
    assert_eq!(results[1].outcome, ToolExecutionOutcome::Cancelled);
    assert!(results[1].effects.changes.is_empty());
}
#[tokio::test]
async fn late_cancel_preserves_known_effect_and_skips_unstarted_sibling_on_both_logs() {
    cancellation_campaign(Case::memory()).await;
    cancellation_campaign(Case::file()).await;
}

async fn identity_campaign(case: Case) {
    let (actor, services, count) = fixture(
        vec![call(
            "one",
            json!({"path":"before.txt", "hook_rewrite":true}),
        )],
        false,
    );
    let settled = ingest_coding(
        &actor,
        case.log.as_ref(),
        Arc::clone(&services),
        THREAD,
        message_fact("m1", "do it"),
        LIMIT,
    )
    .await
    .unwrap();
    let first = receipts(&settled.log).remove(0);
    let prepared: ToolPreparedV1 = serde_json::from_value(
        settled
            .log
            .iter()
            .find(|fact| fact.kind == "tool.prepared")
            .unwrap()
            .payload["prepared"]
            .clone(),
    )
    .unwrap();
    assert_ne!(
        prepared.original_args_digest,
        prepared.effective_args_digest
    );
    assert_eq!(first.effects.changes[0].path, "hook-resolved.txt");
    assert_eq!(
        first.identity.admission.as_ref().unwrap().run_id,
        "exact-run"
    );
    // A new narrative turn does not erase the earlier immutable identity.
    let again = ingest_coding(
        &actor,
        case.log.as_ref(),
        Arc::clone(&services),
        THREAD,
        message_fact("m2", "again"),
        LIMIT,
    )
    .await
    .unwrap();
    assert!(again.view.schema_error.is_none());
    assert_eq!(count.load(Ordering::SeqCst), 1);
    let mut foreign = first;
    foreign.identity.turn += 1;
    let result = foreign.clone();
    case.log.append(THREAD, &[NewFact { kind: "tool.result".into(), key: "foreign".into(), payload: json!({
        "toolCallId":"one", "ok":true, "output":result.output.as_str().unwrap(), "receipt":foreign,
    }) }], None).unwrap();
    let reopened = case.reopen();
    let refused = resume_coding(&actor, reopened.as_ref(), services, THREAD, LIMIT)
        .await
        .unwrap();
    assert!(refused
        .view
        .schema_error
        .as_deref()
        .unwrap()
        .contains("tool_effect_identity_mismatch"));
}
#[tokio::test]
async fn effective_args_and_foreign_terminal_identity_are_checked_on_both_logs() {
    identity_campaign(Case::memory()).await;
    identity_campaign(Case::file()).await;
}

async fn namespace_campaign(case: Case) {
    let (_, services, count) = fixture(vec![call("one", json!({"path":"one.txt"}))], false);
    let config = HarnessConfig::new(20, usize::MAX, LIMIT, 1, vec![], vec![]).unwrap();
    let actor = compose_coding_actor(
        "composed",
        vec![with_key_namespace("host", coding_scheduler(config))],
    );
    let settled = ingest_coding(
        &actor,
        case.log.as_ref(),
        services,
        THREAD,
        message_fact("m1", "do it"),
        LIMIT,
    )
    .await
    .unwrap();
    assert_eq!(count.load(Ordering::SeqCst), 1);
    let prepared = settled
        .log
        .iter()
        .find(|fact| fact.kind == "tool.prepared")
        .unwrap();
    let result = settled
        .log
        .iter()
        .find(|fact| fact.kind == "tool.result")
        .unwrap();
    assert!(prepared.seq < result.seq);
    assert!(prepared.cause.is_none());
    assert!(result.cause.as_deref().unwrap().starts_with("host/tool:"));
}

#[tokio::test]
async fn namespaced_composition_keeps_preparation_and_the_deferred_effect_on_both_logs() {
    namespace_campaign(Case::memory()).await;
    namespace_campaign(Case::file()).await;
}

async fn distinct_intents_campaign(case: Case) {
    let (_, services, count) = fixture(vec![call("same", json!({"path":"same.txt"}))], false);
    let same_call = call("same", json!({"path":"same.txt"}));
    let completion = Arc::new(ScriptModel(Mutex::new(VecDeque::from([
        ModelDecision::Tool {
            call: same_call.clone(),
            also: vec![],
        },
        ModelDecision::Tool {
            call: same_call,
            also: vec![],
        },
        ModelDecision::Text {
            text: "done".into(),
        },
    ]))));
    let services = Arc::new(CodingServices {
        completion,
        tools: Arc::clone(&services.tools),
        compactor: Arc::clone(&services.compactor),
    });
    let config = HarnessConfig::new(20, usize::MAX, LIMIT, 1, vec![], vec![]).unwrap();
    let actor = coding_actor(config);
    let settled = ingest_coding(
        &actor,
        case.log.as_ref(),
        Arc::clone(&services),
        THREAD,
        message_fact("m1", "two intents"),
        LIMIT,
    )
    .await
    .unwrap();
    assert!(settled.view.schema_error.is_none());
    assert_eq!(count.load(Ordering::SeqCst), 2);
    let results = receipts(&settled.log);
    assert_ne!(results[0].identity, results[1].identity);
    assert_eq!(
        results[0].identity.tool_call_id,
        results[1].identity.tool_call_id
    );
    let reopened = case.reopen();
    resume_coding(&actor, reopened.as_ref(), services, THREAD, LIMIT)
        .await
        .unwrap();
    assert_eq!(count.load(Ordering::SeqCst), 2);
}
#[tokio::test]
async fn identical_arguments_with_distinct_intents_execute_separately_on_both_logs() {
    distinct_intents_campaign(Case::memory()).await;
    distinct_intents_campaign(Case::file()).await;
}

async fn malformed_receipt_campaign(case: Case) {
    let (actor, services, count) = fixture(vec![call("one", json!({"path":"one.txt"}))], false);
    let settled = ingest_coding(
        &actor,
        case.log.as_ref(),
        Arc::clone(&services),
        THREAD,
        message_fact("m1", "do it"),
        LIMIT,
    )
    .await
    .unwrap();
    let mut receipt = serde_json::to_value(receipts(&settled.log).remove(0)).unwrap();
    receipt["schema"] = json!("future.v2");
    case.log.append(THREAD, &[NewFact { kind: "tool.result".into(), key: "future-result".into(),
        payload: json!({"toolCallId":"one", "ok":true, "output":"old compatibility text", "receipt":receipt}) }], None).unwrap();
    let reopened = case.reopen();
    let refused = resume_coding(&actor, reopened.as_ref(), services, THREAD, LIMIT)
        .await
        .unwrap();
    assert!(refused
        .view
        .schema_error
        .as_deref()
        .unwrap()
        .contains("tool_effect_schema_unsupported"));
    assert_eq!(count.load(Ordering::SeqCst), 1);
}
#[tokio::test]
async fn unknown_receipt_version_is_never_downgraded_to_legacy_on_both_logs() {
    malformed_receipt_campaign(Case::memory()).await;
    malformed_receipt_campaign(Case::file()).await;
}

async fn legacy_unsettled_campaign(case: Case) {
    let (actor, services, count) = fixture(vec![call("one", json!({"path":"one.txt"}))], false);
    services
        .completion
        .complete(
            CompletionRequest {
                system: vec![],
                tools: vec![],
                summary: String::new(),
                messages: vec![],
            },
            CancellationToken::new(),
        )
        .await
        .unwrap();
    case.log
        .append(
            THREAD,
            &[
                message_fact("m1", "old operation"),
                NewFact {
                    kind: "model.turn".into(),
                    key: "model:1:0".into(),
                    payload: serde_json::to_value(ModelDecision::Tool {
                        call: call("one", json!({"path":"one.txt"})),
                        also: vec![],
                    })
                    .unwrap(),
                },
            ],
            None,
        )
        .unwrap();
    let reopened = case.reopen();
    let settled = resume_coding(&actor, reopened.as_ref(), services, THREAD, LIMIT)
        .await
        .unwrap();
    assert_eq!(count.load(Ordering::SeqCst), 0);
    let result = settled
        .log
        .iter()
        .find(|fact| fact.kind == "tool.result")
        .unwrap();
    assert_eq!(result.payload["unresolvedLegacy"], true);
    assert_eq!(result.payload["ok"], false);
    assert!(result.payload.get("receipt").is_none());
    assert!(settled.log.iter().all(|fact| fact.kind != "tool.prepared"));
}
#[tokio::test]
async fn unsettled_legacy_call_is_never_replayed_or_promoted_on_both_logs() {
    legacy_unsettled_campaign(Case::memory()).await;
    legacy_unsettled_campaign(Case::file()).await;
}

async fn malformed_sibling_campaign(case: Case) {
    let (actor, services, count) = fixture(
        vec![
            call("first", json!({"path":"first.txt"})),
            call("second", json!({"path":"second.txt", "bad_identity":true})),
        ],
        true,
    );
    let settled = ingest_coding(
        &actor,
        case.log.as_ref(),
        services,
        THREAD,
        message_fact("m1", "two effects"),
        LIMIT,
    )
    .await
    .unwrap();
    let results = receipts(&settled.log);
    assert_eq!(count.load(Ordering::SeqCst), 2);
    assert_eq!(results[0].outcome, ToolExecutionOutcome::Succeeded);
    assert_eq!(results[0].effects.changes[0].path, "first.txt");
    assert_eq!(results[1].outcome, ToolExecutionOutcome::Unknown);
    assert!(results[1]
        .output
        .as_str()
        .unwrap()
        .contains("tool_effect_identity_mismatch"));
}
#[tokio::test]
async fn malformed_sibling_receipt_cannot_discard_a_known_result_on_both_logs() {
    malformed_sibling_campaign(Case::memory()).await;
    malformed_sibling_campaign(Case::file()).await;
}
