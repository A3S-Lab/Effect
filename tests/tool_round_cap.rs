use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use a3s_effect::{
    coding_actor, ingest_coding, message_fact, ActorError, CodingServices, Compactor, Completion,
    CompletionRequest, HarnessConfig, MemoryLog, ModelDecision, ToolRunner, ToolSpec,
};

struct RecordingModel {
    tool_counts: Mutex<Vec<usize>>,
    messages: Mutex<Vec<Vec<String>>>,
    calls: AtomicUsize,
}

impl Completion for RecordingModel {
    fn complete(
        &self,
        request: CompletionRequest,
    ) -> a3s_effect::coding::BoxFuture<Result<ModelDecision, ActorError>> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        self.tool_counts.lock().unwrap().push(request.tools.len());
        self.messages.lock().unwrap().push(request.messages.clone());
        Box::pin(async {
            Ok(ModelDecision::Text {
                text: "done".into(),
            })
        })
    }
}

struct NoTools;

impl ToolRunner for NoTools {
    fn run(
        &self,
        _call: a3s_effect::ToolCall,
    ) -> a3s_effect::coding::BoxFuture<Result<serde_json::Value, ActorError>> {
        Box::pin(async { Ok(serde_json::json!("unused")) })
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

#[tokio::test]
async fn tool_round_cap_completes_once_with_an_empty_tool_list() {
    let model = Arc::new(RecordingModel {
        tool_counts: Mutex::new(Vec::new()),
        messages: Mutex::new(Vec::new()),
        calls: AtomicUsize::new(0),
    });
    let config = HarnessConfig::new(
        4,
        10_000,
        8,
        1,
        vec!["system".into()],
        vec![ToolSpec {
            name: "read".into(),
            description: "Read a file".into(),
        }],
    )
    .unwrap()
    .with_tool_round_cap(0);
    let actor = coding_actor(config);
    let services = Arc::new(CodingServices {
        completion: model.clone(),
        tools: Arc::new(NoTools),
        compactor: Arc::new(NoCompact),
    });
    let log = MemoryLog::new();
    let settled = ingest_coding(
        &actor,
        &log,
        services,
        "thread-1",
        message_fact("m1", "hello"),
        8,
    )
    .await
    .expect("ingest");
    assert_eq!(settled.view.assistant.as_deref(), Some("done"));
    assert_eq!(model.calls.load(Ordering::SeqCst), 1);
    assert_eq!(model.tool_counts.lock().unwrap().as_slice(), &[0]);
    let messages = model.messages.lock().unwrap().clone();
    assert_eq!(messages, vec![vec!["user\nhello".to_string()]]);
    assert!(messages
        .iter()
        .flatten()
        .all(|line| !line.contains("TOOL_BUDGET_FINALIZATION")));
}
