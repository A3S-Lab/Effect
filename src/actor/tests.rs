#![cfg(test)]

use super::{append_input, ingest, ingest_with_cancel};
use crate::coding::BoxFuture;
use crate::{
    coding_actor, message_fact, ActorError, CodingServices, Compactor, Completion,
    CompletionRequest, Exit, Fact, FileLog, HarnessConfig, LogStore, MemoryLog, ModelDecision,
    NewFact, ToolCall, ToolRunner, ToolSpec,
};
use serde_json::{json, Value};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use tokio_util::sync::CancellationToken;

struct CountedModel(Arc<AtomicUsize>);

impl Completion for CountedModel {
    fn complete(
        &self,
        _request: CompletionRequest,
        _cancel: CancellationToken,
    ) -> BoxFuture<Result<ModelDecision, ActorError>> {
        let call = self.0.fetch_add(1, Ordering::SeqCst);
        Box::pin(async move {
            Ok(if call == 0 {
                ModelDecision::Tool {
                    call: ToolCall {
                        id: "original-read".into(),
                        name: "read".into(),
                        args: json!({ "path": "original.txt" }),
                        needs_confirmation: false,
                        text: None,
                        reasoning: None,
                    },
                    also: Vec::new(),
                }
            } else {
                ModelDecision::Text {
                    text: "original turn complete".into(),
                }
            })
        })
    }
}

struct CountedTools(Arc<AtomicUsize>);

impl ToolRunner for CountedTools {
    fn run(
        &self,
        _call: ToolCall,
        _cancel: CancellationToken,
    ) -> BoxFuture<Result<Value, ActorError>> {
        self.0.fetch_add(1, Ordering::SeqCst);
        Box::pin(async { Ok(json!({ "content": "original bytes" })) })
    }
}

struct CountedCompactor(Arc<AtomicUsize>);

impl Compactor for CountedCompactor {
    fn compact(&self, _messages: &[String]) -> BoxFuture<Result<String, ActorError>> {
        self.0.fetch_add(1, Ordering::SeqCst);
        Box::pin(async { Ok("original summary".into()) })
    }
}

struct FailedAppend {
    original: Arc<dyn LogStore>,
    reads: AtomicUsize,
    appends: AtomicUsize,
}

impl FailedAppend {
    fn error() -> ActorError {
        ActorError::Handler {
            key: "append-input-fault".into(),
            message: "injected original log append failure".into(),
        }
    }
}

impl LogStore for FailedAppend {
    fn read(&self, thread_id: &str) -> Result<Vec<Fact>, ActorError> {
        self.reads.fetch_add(1, Ordering::SeqCst);
        self.original.read(thread_id)
    }

    fn append(
        &self,
        _thread_id: &str,
        _facts: &[NewFact],
        _cause: Option<&str>,
    ) -> Result<Vec<Fact>, ActorError> {
        self.appends.fetch_add(1, Ordering::SeqCst);
        Err(Self::error())
    }
}

/// An opaque Code-shaped transport fixture, not a Code admission proof.
/// Code owns recovery-unproved validation and key derivation; this layer only
/// proves that the caller's exact input reaches the original log unchanged.
fn recovery_input() -> NewFact {
    NewFact {
        kind: "recovery.unproved".into(),
        key: format!("recovery-unproved:sha256:{}", "1".repeat(64)),
        payload: json!({
            "schema": "a3s.code.recovery-unproved.v1",
            "session_id": "original-thread",
            "thread_id": "original-thread",
            "source_run_id": "original-run",
            "source_checkpoint_schema_version": 1,
            "source_checkpoint_content_digest": format!("sha256:{}", "2".repeat(64)),
            "source_capability_binding_digest": null,
            "reason": "cache_evidence_missing"
        }),
    }
}

async fn prove_append_only_input(log: Arc<dyn LogStore>, reopen: impl Fn() -> Arc<dyn LogStore>) {
    let model_calls = Arc::new(AtomicUsize::new(0));
    let tool_calls = Arc::new(AtomicUsize::new(0));
    let compactor_calls = Arc::new(AtomicUsize::new(0));
    let services = Arc::new(CodingServices {
        completion: Arc::new(CountedModel(Arc::clone(&model_calls))),
        tools: Arc::new(CountedTools(Arc::clone(&tool_calls))),
        compactor: Arc::new(CountedCompactor(Arc::clone(&compactor_calls))),
    });
    let config = HarnessConfig::new(
        4,
        1_000_000,
        32,
        1,
        Vec::new(),
        vec![ToolSpec {
            name: "read".into(),
            description: "Read original bytes".into(),
        }],
    )
    .unwrap();
    let limit = config.step_limit();
    let actor = coding_actor(config);
    let thread = "original-thread";
    let settled = ingest(
        &actor,
        log.as_ref(),
        Arc::clone(&services),
        thread,
        message_fact("original-message", "exercise the original actor writer"),
        limit,
    )
    .await
    .unwrap();
    assert_eq!(model_calls.load(Ordering::SeqCst), 2);
    assert_eq!(tool_calls.load(Ordering::SeqCst), 1);
    assert_eq!(compactor_calls.load(Ordering::SeqCst), 0);
    assert_eq!(
        settled
            .log
            .iter()
            .filter(|fact| fact.kind == "tool.prepared")
            .count(),
        1
    );
    assert_eq!(
        settled
            .log
            .iter()
            .filter(|fact| fact.kind == "tool.result")
            .count(),
        1
    );

    // The original actor above proves that these probes and the existing
    // preparation/result writer are live. Measure only append-only work below.
    model_calls.store(0, Ordering::SeqCst);
    tool_calls.store(0, Ordering::SeqCst);
    compactor_calls.store(0, Ordering::SeqCst);

    append_input(
        log.as_ref(),
        thread,
        message_fact("waiting-message", "do not dispatch the waiting model"),
    )
    .unwrap();
    let original = log.read(thread).unwrap();
    assert!(
        !actor.project(&original).unwrap().1.is_empty(),
        "a resumed actor would execute work; append-only must leave it pending"
    );
    let input = recovery_input();
    append_input(log.as_ref(), thread, input.clone()).unwrap();
    let reopened = reopen();
    let retained = reopened.read(thread).unwrap();
    assert_eq!(retained.len(), original.len() + 1);
    assert_eq!(&retained[..original.len()], original.as_slice());
    let appended = retained.last().unwrap();
    assert_eq!(appended.kind, input.kind);
    assert_eq!(appended.key, input.key);
    assert_eq!(appended.payload, input.payload);
    assert_eq!(appended.cause, None);
    assert_eq!(appended.seq, original.last().unwrap().seq + 1);
    assert!(reopened.read("different-thread").unwrap().is_empty());

    append_input(reopened.as_ref(), thread, input.clone()).unwrap();
    assert_eq!(reopen().read(thread).unwrap(), retained);
    for incompatible in [
        NewFact {
            payload: json!({ "different": true }),
            ..input.clone()
        },
        NewFact {
            kind: "different.kind".into(),
            ..input.clone()
        },
    ] {
        assert_eq!(
            append_input(reopened.as_ref(), thread, incompatible),
            Err(ActorError::DuplicateFact {
                key: input.key.clone()
            })
        );
        assert_eq!(reopen().read(thread).unwrap(), retained);
    }
    assert_eq!(
        append_input(reopened.as_ref(), "../outside", input.clone()),
        Err(ActorError::BadThreadId {
            thread_id: "../outside".into()
        })
    );
    let failed = FailedAppend {
        original: reopened,
        reads: AtomicUsize::new(0),
        appends: AtomicUsize::new(0),
    };
    assert_eq!(
        append_input(&failed, thread, input),
        Err(FailedAppend::error())
    );
    let failure = ingest_with_cancel(
        &actor,
        &failed,
        services,
        thread,
        message_fact(
            "failed-ingestion",
            "the original append failure must stop dispatch",
        ),
        limit,
        &CancellationToken::new(),
    )
    .await;
    assert!(matches!(failure, Err(Exit::Fail(error)) if error == FailedAppend::error()));
    assert_eq!(failed.appends.load(Ordering::SeqCst), 2);
    assert_eq!(failed.reads.load(Ordering::SeqCst), 0);
    assert_eq!(reopen().read(thread).unwrap(), retained);
    assert_eq!(model_calls.load(Ordering::SeqCst), 0);
    assert_eq!(tool_calls.load(Ordering::SeqCst), 0);
    assert_eq!(compactor_calls.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn append_only_input_preserves_original_memory_log_without_dispatch() {
    let log: Arc<dyn LogStore> = Arc::new(MemoryLog::new());
    let reopened = Arc::clone(&log);
    prove_append_only_input(log, move || Arc::clone(&reopened)).await;
}

#[tokio::test]
async fn append_only_input_preserves_reopened_file_log_without_dispatch() {
    let root = tempfile::tempdir().unwrap();
    let path = root.path().join("facts");
    let log: Arc<dyn LogStore> = Arc::new(FileLog::open(&path).unwrap());
    prove_append_only_input(log, || Arc::new(FileLog::open(&path).unwrap())).await;
}
