use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use a3s_effect::{ActorError, Effect, Exit, Schedule};

#[tokio::test]
async fn constructing_an_effect_does_not_run_it() {
    let calls = Arc::new(AtomicUsize::new(0));
    let seen = Arc::clone(&calls);
    let effect = Effect::<usize, ActorError, ()>::from_async(move |_services, _cancel| {
        let seen = Arc::clone(&seen);
        async move {
            seen.fetch_add(1, Ordering::SeqCst);
            Ok(7)
        }
    });
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    let (result, _) = effect.run(Arc::new(())).await;
    assert_eq!(result.unwrap(), 7);
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn catch_fail_handles_expected_errors_and_lets_defects_through() {
    let failed = Effect::<usize, ActorError, ()>::fail(ActorError::Config("missing".into()));
    let recovered = failed.catch_fail(|error| Effect::succeed(error.to_string().len()));
    let (result, _) = recovered.run(Arc::new(())).await;
    assert!(result.unwrap() > 0);

    let defect = Effect::<usize, ActorError, ()>::die("boom");
    let untouched = defect.catch_fail(|_| Effect::succeed(1));
    let (result, _) = untouched.run(Arc::new(())).await;
    assert!(matches!(result, Err(Exit::Die(message)) if message == "boom"));
}

#[tokio::test]
async fn retry_repeats_expected_failures_and_stops_on_defects() {
    let attempts = Arc::new(AtomicUsize::new(0));
    let seen = Arc::clone(&attempts);
    let flaky = Effect::<&'static str, ActorError, ()>::from_async(move |_services, _cancel| {
        let seen = Arc::clone(&seen);
        async move {
            let n = seen.fetch_add(1, Ordering::SeqCst);
            if n == 0 {
                Err(Exit::Fail(ActorError::Config("once".into())))
            } else {
                Ok("ok")
            }
        }
    });
    let (result, trace) = flaky
        .retry(Schedule {
            remaining: 1,
            delay: Duration::ZERO,
        })
        .run(Arc::new(()))
        .await;
    assert_eq!(result.unwrap(), "ok");
    assert_eq!(attempts.load(Ordering::SeqCst), 2);
    assert!(trace
        .iter()
        .any(|event| event.name == "retry" && event.outcome == "fail"));

    let defects = Arc::new(AtomicUsize::new(0));
    let seen = Arc::clone(&defects);
    let die = Effect::<(), ActorError, ()>::from_async(move |_services, _cancel| {
        let seen = Arc::clone(&seen);
        async move {
            seen.fetch_add(1, Ordering::SeqCst);
            Err(Exit::Die("no".into()))
        }
    });
    let (result, _) = die
        .retry(Schedule {
            remaining: 3,
            delay: Duration::ZERO,
        })
        .run(Arc::new(()))
        .await;
    assert!(matches!(result, Err(Exit::Die(_))));
    assert_eq!(defects.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn bracket_releases_on_success_failure_and_cancellation() {
    let released = Arc::new(AtomicUsize::new(0));

    let acquire = {
        let released = Arc::clone(&released);
        Effect::<Arc<AtomicUsize>, ActorError, ()>::from_async(move |_services, _cancel| {
            let released = Arc::clone(&released);
            async move { Ok(released) }
        })
    };
    let release_count = Arc::clone(&released);
    let success = acquire.bracket(
        move |_| {
            release_count.fetch_add(1, Ordering::SeqCst);
        },
        |_resource| Effect::succeed("done"),
    );
    let (result, _) = success.run(Arc::new(())).await;
    assert_eq!(result.unwrap(), "done");
    assert_eq!(released.load(Ordering::SeqCst), 1);

    let released = Arc::new(AtomicUsize::new(0));
    let acquire = {
        let released = Arc::clone(&released);
        Effect::<Arc<AtomicUsize>, ActorError, ()>::from_async(move |_services, _cancel| {
            let released = Arc::clone(&released);
            async move { Ok(released) }
        })
    };
    let release_count = Arc::clone(&released);
    let failed = acquire.bracket(
        move |_| {
            release_count.fetch_add(1, Ordering::SeqCst);
        },
        |_resource| Effect::<(), ActorError, ()>::fail(ActorError::Config("use".into())),
    );
    let (result, _) = failed.run(Arc::new(())).await;
    assert!(matches!(result, Err(Exit::Fail(_))));
    assert_eq!(released.load(Ordering::SeqCst), 1);

    let released = Arc::new(AtomicUsize::new(0));
    let acquire = {
        let released = Arc::clone(&released);
        Effect::<Arc<AtomicUsize>, ActorError, ()>::from_async(move |_services, _cancel| {
            let released = Arc::clone(&released);
            async move { Ok(released) }
        })
    };
    let release_count = Arc::clone(&released);
    let hanging = acquire.bracket(
        move |_| {
            release_count.fetch_add(1, Ordering::SeqCst);
        },
        |_resource| {
            Effect::<(), ActorError, ()>::from_async(|_services, _cancel| async {
                std::future::pending::<()>().await;
                Ok(())
            })
        },
    );
    let (result, _) = hanging
        .timeout(Duration::from_millis(20))
        .run(Arc::new(()))
        .await;
    assert!(matches!(result, Err(Exit::Interrupt)));
    assert_eq!(released.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn zip_par_cancels_the_sibling_after_a_failure() {
    let cancelled = Arc::new(AtomicUsize::new(0));
    let seen = Arc::clone(&cancelled);
    let slow = Effect::<(), ActorError, ()>::from_async(move |_services, cancel| {
        let seen = Arc::clone(&seen);
        async move {
            cancel.cancelled().await;
            seen.fetch_add(1, Ordering::SeqCst);
            Err(Exit::Interrupt)
        }
    });
    let fast = Effect::<(), ActorError, ()>::fail(ActorError::Config("left".into()));
    let (result, _) = fast.zip_par(&slow).run(Arc::new(())).await;
    assert!(matches!(result, Err(Exit::Fail(ActorError::Config(message))) if message == "left"));
    assert_eq!(cancelled.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn provide_swaps_the_service_environment() {
    #[derive(Clone)]
    struct ModelId(u8);

    let effect = Effect::<u8, ActorError, ModelId>::from_async(|services, _cancel| {
        let id = services.0;
        async move { Ok(id) }
    });
    let provided = effect.provide(ModelId(4));
    let (result, trace) = provided.with_span("model").run(Arc::new(())).await;
    assert_eq!(result.unwrap(), 4);
    assert!(trace
        .iter()
        .any(|event| event.name == "model" && event.outcome == "ok"));
}

#[tokio::test]
async fn config_is_rejected_before_an_actor_exists() {
    let error = a3s_effect::HarnessConfig::new(1, 10, 0, 1, Vec::new(), Vec::new()).unwrap_err();
    assert!(matches!(error, ActorError::Config(_)));
}
