#![cfg(test)]

use std::sync::{Arc, Mutex};

use super::*;
use crate::coding::{
    coding_scheduler, message_fact, Compactor, Completion, CompletionRequest, ModelDecision,
    ToolCall, ToolRunner,
};
use crate::{component, resume, Effect, Exit, MemoryLog};

struct Model(Mutex<Vec<Vec<String>>>);
impl Completion for Model {
    fn complete(
        &self,
        request: CompletionRequest,
        _cancel: tokio_util::sync::CancellationToken,
    ) -> crate::coding::BoxFuture<Result<ModelDecision, ActorError>> {
        self.0.lock().unwrap().push(request.system);
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
        _call: ToolCall,
        _cancel: tokio_util::sync::CancellationToken,
    ) -> crate::coding::BoxFuture<Result<serde_json::Value, ActorError>> {
        Box::pin(async { panic!("this contract must not invoke a Tool") })
    }
}

struct NoCompact;
impl Compactor for NoCompact {
    fn compact(
        &self,
        _messages: &[String],
    ) -> crate::coding::BoxFuture<Result<String, ActorError>> {
        Box::pin(async { panic!("this contract must not compact") })
    }
}

#[tokio::test]
async fn composed_inference_uses_ordered_current_fact_derived_system_ports() {
    let model = Arc::new(Model(Mutex::new(Vec::new())));
    let services = Arc::new(CodingServices {
        completion: model.clone(),
        tools: Arc::new(NoTools),
        compactor: Arc::new(NoCompact),
    });
    let config =
        HarnessConfig::new(4, 1000, 16, 1, vec!["scheduler fallback".into()], vec![]).unwrap();
    let dynamic = component(
        String::new,
        |state, fact| {
            if fact.kind == "user.message" {
                fact.payload["text"].as_str().unwrap().to_owned()
            } else {
                state
            }
        },
        |state| {
            (
                CodingView {
                    system: vec![format!("host:{state}")],
                    ..CodingView::empty()
                },
                Vec::new(),
            )
        },
    );
    let actor = compose_coding_actor(
        "ports",
        vec![
            system(vec!["first".into()]),
            infer(vec![dynamic]),
            system(vec!["last".into()]),
            coding_scheduler(config),
        ],
    );
    let log = MemoryLog::default();
    crate::ingest(
        &actor,
        &log,
        services.clone(),
        "t",
        message_fact("one", "one"),
        16,
    )
    .await
    .unwrap();
    crate::ingest(
        &actor,
        &log,
        services.clone(),
        "t",
        message_fact("two", "two"),
        16,
    )
    .await
    .unwrap();
    assert_eq!(
        *model.0.lock().unwrap(),
        [
            vec!["first", "host:one", "last"],
            vec!["first", "host:two", "last"]
        ]
    );
    resume(&actor, &log, services, "t", 16).await.unwrap();
    assert_eq!(
        model.0.lock().unwrap().len(),
        2,
        "resume repeated an already recorded inference"
    );
}

#[tokio::test]
async fn service_mapping_keeps_cancellation_and_original_trace() {
    let effect = Effect::<u32, ActorError, u32>::from_async_observed(|value, cancel| async move {
        if cancel.is_cancelled() {
            Err(Exit::Interrupt)
        } else {
            Ok(*value)
        }
    })
    .with_span("mapped")
    .map_services(|outer: Arc<(u32,)>| Arc::new(outer.0));
    let (result, trace) = effect.clone().run(Arc::new((7,))).await;
    assert_eq!(result.unwrap(), 7);
    assert_eq!(trace[0].name, "mapped");
    let cancel = tokio_util::sync::CancellationToken::new();
    cancel.cancel();
    let (result, _) = effect.run_with_cancel(Arc::new((7,)), cancel).await;
    assert!(matches!(result, Err(Exit::Interrupt)));
}
