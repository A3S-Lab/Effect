//! The actor loop's contract with cancellation: a transition that completed
//! and appended is durable work and settles; a transition that never started
//! because the token fired refuses with [`Exit::Interrupt`].

use a3s_effect::{
    coding_actor, component, ingest_with_cancel, message_fact, Actor, ActorError, CodingServices,
    Compactor, Completion, CompletionRequest, Effect, Exit, HarnessConfig, MemoryLog,
    ModelDecision, NewFact, ToolCall, ToolRunner, Transition,
};
use serde_json::Value;
use std::sync::Arc;
use tokio_util::sync::CancellationToken;

/// Model that never answers; only the harness's own cancel handling ends the
/// turn, which is exactly the interactive interruption shape.
struct HangingModel;

impl Completion for HangingModel {
    fn complete(
        &self,
        _request: CompletionRequest,
        _cancel: tokio_util::sync::CancellationToken,
    ) -> a3s_effect::coding::BoxFuture<Result<ModelDecision, ActorError>> {
        Box::pin(std::future::pending())
    }
}

struct IdleTools;

impl ToolRunner for IdleTools {
    fn run(
        &self,
        _call: ToolCall,
        _cancel: tokio_util::sync::CancellationToken,
    ) -> a3s_effect::coding::BoxFuture<Result<Value, ActorError>> {
        Box::pin(std::future::pending())
    }
}

struct IdleCompactor;

impl Compactor for IdleCompactor {
    fn compact(
        &self,
        _messages: &[String],
    ) -> a3s_effect::coding::BoxFuture<Result<String, ActorError>> {
        Box::pin(std::future::pending())
    }
}

fn services() -> Arc<CodingServices> {
    Arc::new(CodingServices {
        completion: Arc::new(HangingModel),
        tools: Arc::new(IdleTools),
        compactor: Arc::new(IdleCompactor),
    })
}

#[tokio::test]
async fn a_cancelled_model_turn_settles_with_the_interrupted_answer() {
    let config = HarnessConfig::new(4, 1_000_000, 32, 2, Vec::new(), Vec::new()).unwrap();
    let actor = coding_actor(config.clone());
    let log = MemoryLog::new();
    let cancel = CancellationToken::new();
    cancel.cancel();
    let settled = ingest_with_cancel(
        &actor,
        &log,
        services(),
        "thread-1",
        message_fact("m1", "hello"),
        config.step_limit(),
        &cancel,
    )
    .await
    .expect("the interrupted turn settles instead of erroring");
    assert_eq!(
        settled.view.assistant.as_deref(),
        Some("(Response interrupted by the user.)")
    );
    assert_eq!(settled.steps, 1, "the interrupt fact is one durable step");
}

#[tokio::test]
async fn an_unobserved_transition_still_refuses_to_start_once_cancelled() {
    let cancel_outer = CancellationToken::new();
    let canceller = cancel_outer.clone();
    // The first transition cancels the host token and lands a durable fact;
    // the second only becomes enabled after that fact.
    let tick = component(
        || false,
        |done, fact| done || fact.kind == "tick",
        move |done| {
            if *done {
                return ((), Vec::new());
            }
            let canceller = canceller.clone();
            (
                (),
                vec![Transition {
                    key: "self-cancelling".into(),
                    run: Effect::from_async_observed(move |_services: Arc<()>, cancel| {
                        let canceller = canceller.clone();
                        async move {
                            let _ = cancel;
                            canceller.cancel();
                            Ok(vec![NewFact {
                                kind: "tick".into(),
                                key: "tick".into(),
                                payload: serde_json::json!({}),
                            }])
                        }
                    })
                    .into(),
                }],
            )
        },
    );
    let pending = component(
        || false,
        |armed, fact| armed || fact.kind == "tick",
        move |armed| {
            if !*armed {
                return ((), Vec::new());
            }
            (
                (),
                vec![Transition {
                    key: "never-starts".into(),
                    run: Effect::from_async(move |_services: Arc<()>, _cancel| {
                        Box::pin(async move {
                            unreachable!("from_async refuses to start once cancelled")
                        })
                    })
                    .into(),
                }],
            )
        },
    );
    let actor = Actor::new("cancel", vec![tick, pending], |_views| ());
    let log = MemoryLog::new();
    let settled = ingest_with_cancel(
        &actor,
        &log,
        Arc::new(()),
        "thread-1",
        message_fact("m1", "go"),
        16,
        &cancel_outer,
    )
    .await;
    assert!(
        matches!(settled, Err(Exit::Interrupt)),
        "the refused transition still interrupts the loop"
    );
}
