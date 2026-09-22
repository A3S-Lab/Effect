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

fn count() -> (Arc<AtomicUsize>, Arc<AtomicUsize>) {
    let calls = Arc::new(AtomicUsize::new(0));
    (Arc::clone(&calls), calls)
}

fn interrupt_effect() -> Effect<i32, ActorError, ()> {
    Effect::from_async(|_services, _cancel| async { Err(Exit::Interrupt) })
}

#[tokio::test]
async fn map_and_and_then_transform_success_and_skip_other_exits() {
    let (seen, calls) = count();
    let (mapped, _) = Effect::<i32, ActorError, ()>::succeed(2)
        .map(move |n| {
            seen.fetch_add(1, Ordering::SeqCst);
            n + 3
        })
        .run(Arc::new(()))
        .await;
    assert_eq!(mapped.unwrap(), 5);
    assert_eq!(calls.load(Ordering::SeqCst), 1);

    let (seen, calls) = count();
    let (bound, _) = Effect::<i32, ActorError, ()>::succeed(2)
        .and_then(move |n| {
            seen.fetch_add(1, Ordering::SeqCst);
            Effect::succeed(n + 3)
        })
        .run(Arc::new(()))
        .await;
    assert_eq!(bound.unwrap(), 5);
    assert_eq!(calls.load(Ordering::SeqCst), 1);

    let failed = Effect::<i32, ActorError, ()>::fail(ActorError::Config("x".into()));
    let (seen, calls) = count();
    let (mapped, _) = failed
        .map(move |n| {
            seen.fetch_add(1, Ordering::SeqCst);
            n + 1
        })
        .run(Arc::new(()))
        .await;
    assert_eq!(mapped, Err(Exit::Fail(ActorError::Config("x".into()))));
    assert_eq!(calls.load(Ordering::SeqCst), 0);

    let (seen, calls) = count();
    let (bound, _) = failed
        .and_then(move |n| {
            seen.fetch_add(1, Ordering::SeqCst);
            Effect::succeed(n + 1)
        })
        .run(Arc::new(()))
        .await;
    assert_eq!(bound, Err(Exit::Fail(ActorError::Config("x".into()))));
    assert_eq!(calls.load(Ordering::SeqCst), 0);

    let died = Effect::<i32, ActorError, ()>::die("d");
    let (seen, calls) = count();
    let (mapped, _) = died
        .map(move |n| {
            seen.fetch_add(1, Ordering::SeqCst);
            n + 1
        })
        .run(Arc::new(()))
        .await;
    assert!(matches!(mapped, Err(Exit::Die(message)) if message == "d"));
    assert_eq!(calls.load(Ordering::SeqCst), 0);

    let (seen, calls) = count();
    let (bound, _) = died
        .and_then(move |_n| {
            seen.fetch_add(1, Ordering::SeqCst);
            Effect::succeed(1)
        })
        .run(Arc::new(()))
        .await;
    assert!(matches!(bound, Err(Exit::Die(message)) if message == "d"));
    assert_eq!(calls.load(Ordering::SeqCst), 0);

    let (seen, calls) = count();
    let (mapped, _) = interrupt_effect()
        .map(move |n| {
            seen.fetch_add(1, Ordering::SeqCst);
            n + 1
        })
        .run(Arc::new(()))
        .await;
    assert!(matches!(mapped, Err(Exit::Interrupt)));
    assert_eq!(calls.load(Ordering::SeqCst), 0);

    let (seen, calls) = count();
    let (bound, _) = interrupt_effect()
        .and_then(move |_n| {
            seen.fetch_add(1, Ordering::SeqCst);
            Effect::succeed(1)
        })
        .run(Arc::new(()))
        .await;
    assert!(matches!(bound, Err(Exit::Interrupt)));
    assert_eq!(calls.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn race_returns_the_first_result_and_awaits_the_loser() {
    let (seen, calls) = count();
    let left = Effect::<&'static str, ActorError, ()>::from_async(|_services, _cancel| async {
        Ok("left")
    });
    let right = Effect::<&'static str, ActorError, ()>::from_async(move |_services, cancel| {
        let seen = Arc::clone(&seen);
        async move {
            cancel.cancelled().await;
            seen.fetch_add(1, Ordering::SeqCst);
            Err(Exit::Interrupt)
        }
    });
    let (result, _) = left.race(&right).run(Arc::new(())).await;
    assert_eq!(result.unwrap(), "left");
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn timeout_returns_interrupt_without_running_the_child_continuation() {
    let (seen, calls) = count();
    let child = Effect::<(), ActorError, ()>::from_async(move |_services, cancel| {
        let seen = Arc::clone(&seen);
        async move {
            cancel.cancelled().await;
            seen.fetch_add(1, Ordering::SeqCst);
            Err(Exit::Interrupt)
        }
    });
    let (result, _) = child
        .timeout(Duration::from_millis(20))
        .run(Arc::new(()))
        .await;
    assert!(matches!(result, Err(Exit::Interrupt)));
    // `timeout` drops the child inside `tokio::time::timeout` before `cancel`,
    // and it does not await that child.
    assert_eq!(calls.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn with_span_records_fail_die_and_interrupt() {
    let (failed, trace) = Effect::<(), ActorError, ()>::fail(ActorError::Config("x".into()))
        .with_span("mark")
        .run(Arc::new(()))
        .await;
    assert!(matches!(failed, Err(Exit::Fail(_))));
    assert!(trace
        .iter()
        .any(|event| event.name == "mark" && event.outcome == "fail"));

    let (died, trace) = Effect::<(), ActorError, ()>::die("d")
        .with_span("mark")
        .run(Arc::new(()))
        .await;
    assert!(matches!(died, Err(Exit::Die(message)) if message == "d"));
    assert!(trace
        .iter()
        .any(|event| event.name == "mark" && event.outcome == "die"));

    let (interrupted, trace) = interrupt_effect().with_span("mark").run(Arc::new(())).await;
    assert!(matches!(interrupted, Err(Exit::Interrupt)));
    assert!(trace
        .iter()
        .any(|event| event.name == "mark" && event.outcome == "interrupt"));
}

#[tokio::test]
async fn retry_with_no_remaining_runs_once_and_leaves_interrupt() {
    let (seen, calls) = count();
    let failing = Effect::<(), ActorError, ()>::from_async(move |_services, _cancel| {
        let seen = Arc::clone(&seen);
        async move {
            seen.fetch_add(1, Ordering::SeqCst);
            Err(Exit::Fail(ActorError::Config("once".into())))
        }
    });
    let (result, _) = failing
        .retry(Schedule {
            remaining: 0,
            delay: Duration::ZERO,
        })
        .run(Arc::new(()))
        .await;
    assert_eq!(result, Err(Exit::Fail(ActorError::Config("once".into()))));
    assert_eq!(calls.load(Ordering::SeqCst), 1);

    let (seen, calls) = count();
    let interrupting = Effect::<(), ActorError, ()>::from_async(move |_services, _cancel| {
        let seen = Arc::clone(&seen);
        async move {
            seen.fetch_add(1, Ordering::SeqCst);
            Err(Exit::Interrupt)
        }
    });
    let (result, trace) = interrupting
        .retry(Schedule {
            remaining: 2,
            delay: Duration::ZERO,
        })
        .run(Arc::new(()))
        .await;
    assert!(matches!(result, Err(Exit::Interrupt)));
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert!(trace.iter().all(|event| event.name != "retry"));
}

#[tokio::test]
async fn catch_fail_leaves_interrupt_unchanged() {
    let (seen, calls) = count();
    let (result, _) = interrupt_effect()
        .catch_fail(move |_| {
            seen.fetch_add(1, Ordering::SeqCst);
            Effect::succeed(1)
        })
        .run(Arc::new(()))
        .await;
    assert!(matches!(result, Err(Exit::Interrupt)));
    assert_eq!(calls.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn infer_schedule_runs_a_defect_once() {
    let model_attempts = 3_u32;
    let (seen, calls) = count();
    let effect = Effect::<(), ActorError, ()>::from_async(move |_services, _cancel| {
        let seen = Arc::clone(&seen);
        async move {
            seen.fetch_add(1, Ordering::SeqCst);
            Err(Exit::Die("m".into()))
        }
    });
    let (result, _) = effect
        .retry(Schedule {
            remaining: model_attempts.saturating_sub(1),
            delay: Duration::ZERO,
        })
        .run(Arc::new(()))
        .await;
    assert!(matches!(result, Err(Exit::Die(message)) if message == "m"));
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}
