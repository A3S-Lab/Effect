//! Programs as values.
//!
//! An [`Effect`] describes a program: the value it produces, the expected
//! error it can return, and the services it needs. Constructing one does not
//! run it. [`Effect::run`] is the edge where a description becomes a result.
//!
//! The operators follow the Effect v4 onboarding split: typed expected
//! failures, retries as a schedule, structured concurrency that cancels the
//! sibling it no longer needs, resource release on every exit including
//! cancellation, services passed in the type, and spans recorded by the
//! runtime.

use std::future::Future;
use std::marker::PhantomData;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use tokio_util::sync::CancellationToken;

use crate::exit::Exit;

pub type BoxFuture<T> = Pin<Box<dyn Future<Output = T> + Send>>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TraceEvent {
    pub name: String,
    pub outcome: &'static str,
}

pub struct RunCtx<S> {
    pub services: Arc<S>,
    pub cancel: CancellationToken,
    trace: Arc<Mutex<Vec<TraceEvent>>>,
}

impl<S> Clone for RunCtx<S> {
    fn clone(&self) -> Self {
        Self {
            services: Arc::clone(&self.services),
            cancel: self.cancel.clone(),
            trace: Arc::clone(&self.trace),
        }
    }
}

impl<S> RunCtx<S> {
    pub fn child(&self) -> Self {
        Self {
            services: Arc::clone(&self.services),
            cancel: self.cancel.child_token(),
            trace: Arc::clone(&self.trace),
        }
    }

    fn record(&self, name: impl Into<String>, outcome: &'static str) {
        if let Ok(mut events) = self.trace.lock() {
            events.push(TraceEvent {
                name: name.into(),
                outcome,
            });
        }
    }
}

/// A description of a program. `A` is the success value, `E` is the expected
/// error, and `S` is the service environment required to run it.
pub struct Effect<A, E, S> {
    run: Arc<dyn Fn(RunCtx<S>) -> BoxFuture<Result<A, Exit<E>>> + Send + Sync>,
    _marker: PhantomData<fn() -> (A, E)>,
}

impl<A, E, S> Clone for Effect<A, E, S> {
    fn clone(&self) -> Self {
        Self {
            run: Arc::clone(&self.run),
            _marker: PhantomData,
        }
    }
}

impl<A, E, S> Effect<A, E, S>
where
    A: Send + Sync + 'static,
    E: Send + Sync + 'static,
    S: Send + Sync + 'static,
{
    fn new<F>(run: F) -> Self
    where
        F: Fn(RunCtx<S>) -> BoxFuture<Result<A, Exit<E>>> + Send + Sync + 'static,
    {
        Self {
            run: Arc::new(run),
            _marker: PhantomData,
        }
    }

    pub fn succeed(value: A) -> Self
    where
        A: Clone,
    {
        let value = Arc::new(value);
        Self::new(move |_ctx| {
            let value = Arc::clone(&value);
            Box::pin(async move { Ok((*value).clone()) })
        })
    }

    pub fn fail(error: E) -> Self
    where
        E: Clone,
    {
        let error = Arc::new(error);
        Self::new(move |_ctx| {
            let error = Arc::clone(&error);
            Box::pin(async move { Err(Exit::Fail((*error).clone())) })
        })
    }

    pub fn die(message: impl Into<String>) -> Self {
        let message = Arc::new(message.into());
        Self::new(move |_ctx| {
            let message = Arc::clone(&message);
            Box::pin(async move { Err(Exit::Die((*message).clone())) })
        })
    }

    /// Build a program from services and a cancellation token. The function
    /// runs only when [`Effect::run`] is called.
    pub fn from_async<F, Fut>(f: F) -> Self
    where
        F: Fn(Arc<S>, CancellationToken) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<A, Exit<E>>> + Send + 'static,
    {
        let f = Arc::new(f);
        Self::new(move |ctx| {
            if ctx.cancel.is_cancelled() {
                return Box::pin(async { Err(Exit::Interrupt) });
            }
            let fut = f(Arc::clone(&ctx.services), ctx.cancel.clone());
            Box::pin(fut)
        })
    }

    pub fn map<B, F>(&self, f: F) -> Effect<B, E, S>
    where
        B: Send + Sync + 'static,
        F: Fn(A) -> B + Send + Sync + 'static,
    {
        let run = Arc::clone(&self.run);
        let f = Arc::new(f);
        Effect::new(move |ctx| {
            let run = Arc::clone(&run);
            let f = Arc::clone(&f);
            Box::pin(async move { run(ctx).await.map(|value| f(value)) })
        })
    }

    pub fn and_then<B, F>(&self, f: F) -> Effect<B, E, S>
    where
        B: Send + Sync + 'static,
        F: Fn(A) -> Effect<B, E, S> + Send + Sync + 'static,
    {
        let run = Arc::clone(&self.run);
        let f = Arc::new(f);
        Effect::new(move |ctx| {
            let run = Arc::clone(&run);
            let f = Arc::clone(&f);
            Box::pin(async move {
                match run(ctx.clone()).await {
                    Ok(value) => f(value).run_in(ctx).await,
                    Err(error) => Err(error),
                }
            })
        })
    }

    /// Handle an expected failure. Defects and interruption pass through.
    pub fn catch_fail<F>(&self, f: F) -> Self
    where
        F: Fn(E) -> Effect<A, E, S> + Send + Sync + 'static,
    {
        let run = Arc::clone(&self.run);
        let f = Arc::new(f);
        Self::new(move |ctx| {
            let run = Arc::clone(&run);
            let f = Arc::clone(&f);
            Box::pin(async move {
                match run(ctx.clone()).await {
                    Err(Exit::Fail(error)) => f(error).run_in(ctx).await,
                    other => other,
                }
            })
        })
    }

    /// Retry expected failures. Interruption and defects are not retried.
    pub fn retry(&self, schedule: Schedule) -> Self
    where
        E: Clone,
    {
        let run = Arc::clone(&self.run);
        Self::new(move |ctx| {
            let run = Arc::clone(&run);
            let mut schedule = schedule;
            Box::pin(async move {
                loop {
                    if ctx.cancel.is_cancelled() {
                        return Err(Exit::Interrupt);
                    }
                    match run(ctx.clone()).await {
                        Ok(value) => return Ok(value),
                        Err(Exit::Fail(_error)) if schedule.remaining > 0 => {
                            schedule.remaining -= 1;
                            ctx.record("retry", "fail");
                            tokio::select! {
                                _ = ctx.cancel.cancelled() => return Err(Exit::Interrupt),
                                _ = tokio::time::sleep(schedule.delay) => {}
                            }
                        }
                        Err(error) => return Err(error),
                    }
                }
            })
        })
    }

    pub fn timeout(&self, duration: Duration) -> Self {
        let run = Arc::clone(&self.run);
        Self::new(move |ctx| {
            let run = Arc::clone(&run);
            let child = ctx.child();
            Box::pin(async move {
                let child_cancel = child.cancel.clone();
                let fut = run(child);
                match tokio::time::timeout(duration, fut).await {
                    Ok(result) => result,
                    Err(_elapsed) => {
                        child_cancel.cancel();
                        Err(Exit::Interrupt)
                    }
                }
            })
        })
    }

    pub fn with_span(&self, name: impl Into<String>) -> Self {
        let name = name.into();
        let run = Arc::clone(&self.run);
        Self::new(move |ctx| {
            let run = Arc::clone(&run);
            let name = name.clone();
            let fut = run(ctx.clone());
            Box::pin(async move {
                let result = fut.await;
                let outcome = match &result {
                    Ok(_) => "ok",
                    Err(Exit::Fail(_)) => "fail",
                    Err(Exit::Die(_)) => "die",
                    Err(Exit::Interrupt) => "interrupt",
                };
                ctx.record(name, outcome);
                result
            })
        })
    }

    /// Run `left` and `right` together. The first expected failure or defect
    /// cancels the sibling and waits for that sibling to finish.
    pub fn zip_par<B>(&self, right: &Effect<B, E, S>) -> Effect<(A, B), E, S>
    where
        B: Send + Sync + 'static,
    {
        let left_run = Arc::clone(&self.run);
        let right_run = Arc::clone(&right.run);
        Effect::new(move |ctx| {
            let left_run = Arc::clone(&left_run);
            let right_run = Arc::clone(&right_run);
            let left_ctx = ctx.child();
            let right_ctx = ctx.child();
            Box::pin(async move {
                let left = left_run(left_ctx.clone());
                let right = right_run(right_ctx.clone());
                tokio::pin!(left, right);
                tokio::select! {
                    left_result = &mut left => match left_result {
                        Ok(left_value) => match right.await {
                            Ok(right_value) => Ok((left_value, right_value)),
                            Err(error) => Err(error),
                        },
                        Err(error) => {
                            right_ctx.cancel.cancel();
                            let _ = right.await;
                            Err(error)
                        }
                    },
                    right_result = &mut right => match right_result {
                        Ok(right_value) => match left.await {
                            Ok(left_value) => Ok((left_value, right_value)),
                            Err(error) => Err(error),
                        },
                        Err(error) => {
                            left_ctx.cancel.cancel();
                            let _ = left.await;
                            Err(error)
                        }
                    },
                }
            })
        })
    }

    /// First success or failure wins. The other side is cancelled and awaited.
    pub fn race(&self, other: &Self) -> Self {
        let left_run = Arc::clone(&self.run);
        let right_run = Arc::clone(&other.run);
        Self::new(move |ctx| {
            let left_run = Arc::clone(&left_run);
            let right_run = Arc::clone(&right_run);
            let left_ctx = ctx.child();
            let right_ctx = ctx.child();
            Box::pin(async move {
                let left = left_run(left_ctx.clone());
                let right = right_run(right_ctx.clone());
                tokio::pin!(left, right);
                tokio::select! {
                    left_result = &mut left => {
                        right_ctx.cancel.cancel();
                        let _ = right.await;
                        left_result
                    }
                    right_result = &mut right => {
                        left_ctx.cancel.cancel();
                        let _ = left.await;
                        right_result
                    }
                }
            })
        })
    }

    /// Acquire a resource, use it, and release it on success, expected
    /// failure, and cancellation. Dropping the running future releases it too.
    pub fn bracket<B, F, R>(&self, release: R, body: F) -> Effect<B, E, S>
    where
        A: Clone,
        B: Send + Sync + 'static,
        F: Fn(A) -> Effect<B, E, S> + Send + Sync + 'static,
        R: Fn(A) + Send + Sync + 'static,
    {
        let acquire = Arc::clone(&self.run);
        let release: Arc<dyn Fn(A) + Send + Sync> = Arc::new(release);
        let body = Arc::new(body);
        Effect::new(move |ctx| {
            let acquire = Arc::clone(&acquire);
            let release = Arc::clone(&release);
            let body = Arc::clone(&body);
            Box::pin(async move {
                let resource = match acquire(ctx.clone()).await {
                    Ok(resource) => resource,
                    Err(error) => return Err(error),
                };
                let guard = ReleaseOnce::new(resource.clone(), Arc::clone(&release));
                let result = body(resource).run_in(ctx).await;
                guard.release_now();
                result
            })
        })
    }

    /// Replace the service environment. The resulting program no longer
    /// requires `S`; the caller supplies a different environment at the edge.
    pub fn provide<S2>(self, services: S) -> Effect<A, E, S2>
    where
        S2: Send + Sync + 'static,
    {
        let services = Arc::new(services);
        let run = self.run;
        Effect::new(move |ctx| {
            let run = Arc::clone(&run);
            let services = Arc::clone(&services);
            let ctx = RunCtx {
                services,
                cancel: ctx.cancel,
                trace: ctx.trace,
            };
            run(ctx)
        })
    }

    pub async fn run(self, services: Arc<S>) -> (Result<A, Exit<E>>, Vec<TraceEvent>) {
        let trace = Arc::new(Mutex::new(Vec::new()));
        let ctx = RunCtx {
            services,
            cancel: CancellationToken::new(),
            trace: Arc::clone(&trace),
        };
        let result = self.run_in(ctx).await;
        let events = trace
            .lock()
            .map(|events| events.clone())
            .unwrap_or_default();
        (result, events)
    }

    pub(crate) async fn run_in(&self, ctx: RunCtx<S>) -> Result<A, Exit<E>> {
        (self.run)(ctx).await
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Schedule {
    pub remaining: u32,
    pub delay: Duration,
}

struct ReleaseOnce<T> {
    value: Mutex<Option<T>>,
    release: Arc<dyn Fn(T) + Send + Sync>,
}

impl<T> ReleaseOnce<T> {
    fn new(value: T, release: Arc<dyn Fn(T) + Send + Sync>) -> Self {
        Self {
            value: Mutex::new(Some(value)),
            release,
        }
    }

    fn release_now(&self) {
        if let Ok(mut slot) = self.value.lock() {
            if let Some(value) = slot.take() {
                (self.release)(value);
            }
        }
    }
}

impl<T> Drop for ReleaseOnce<T> {
    fn drop(&mut self) {
        self.release_now();
    }
}
