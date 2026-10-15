//! An actor is a fold over an immutable log.
//!
//! Each component is a Moore machine: `step` consumes one fact, `output`
//! returns the view and the transitions that state enables. The runtime is
//! the edge that runs those transitions. A transition that has already left
//! a `cause` in the log is not selected again, so resuming a thread continues
//! unfinished work and does not repeat finished work.

use std::sync::Arc;

use tokio_util::sync::CancellationToken;

use crate::effect::{Effect, TraceEvent};
use crate::error::ActorError;
use crate::exit::Exit;
use crate::fact::{cut_log, Fact, LogCut, NewFact};
use crate::log::LogStore;

pub struct Transition<S> {
    pub key: String,
    pub run: TransitionProgram<S>,
}

/// A preparation value and the still-unexecuted continuation it authorizes.
/// Only the actor appends these facts; preparing never acquires tool effects.
pub struct PreparedTransition<S> {
    pub facts: Vec<NewFact>,
    pub run: Effect<Vec<NewFact>, ActorError, S>,
}

pub enum TransitionProgram<S> {
    Immediate(Effect<Vec<NewFact>, ActorError, S>),
    Prepared(Effect<PreparedTransition<S>, ActorError, S>),
}

impl<S> From<Effect<Vec<NewFact>, ActorError, S>> for TransitionProgram<S> {
    fn from(run: Effect<Vec<NewFact>, ActorError, S>) -> Self {
        Self::Immediate(run)
    }
}

impl<S: Send + Sync + 'static> TransitionProgram<S> {
    /// Project the same environment through preparation and its continuation.
    /// This preserves the system binding installed by composed harnesses.
    pub fn map_services(self, project: impl Fn(Arc<S>) -> Arc<S> + Send + Sync + 'static) -> Self {
        let project = Arc::new(project);
        match self {
            Self::Immediate(run) => {
                Self::Immediate(run.map_services(move |services| project(services)))
            }
            Self::Prepared(prepare) => {
                let continuation_project = Arc::clone(&project);
                Self::Prepared(
                    prepare
                        .map(move |prepared| {
                            let project = Arc::clone(&continuation_project);
                            PreparedTransition {
                                facts: prepared.facts,
                                run: prepared.run.map_services(move |services| project(services)),
                            }
                        })
                        .map_services(move |services| project(services)),
                )
            }
        }
    }
}

type Projection<S, V> =
    dyn Fn(&[Fact]) -> Result<(V, Vec<Transition<S>>), ActorError> + Send + Sync;

pub struct ErasedComponent<S, V> {
    pub(crate) project: Arc<Projection<S, V>>,
}

pub fn component<S, V, St, I, Step, Out>(
    initial: I,
    step: Step,
    output: Out,
) -> ErasedComponent<S, V>
where
    S: Send + Sync + 'static,
    V: 'static,
    St: Send + 'static,
    I: Fn() -> St + Send + Sync + 'static,
    Step: Fn(St, &Fact) -> St + Send + Sync + 'static,
    Out: Fn(&St) -> (V, Vec<Transition<S>>) + Send + Sync + 'static,
{
    ErasedComponent {
        project: Arc::new(move |facts| {
            let mut state = initial();
            for fact in facts {
                state = step(state, fact);
            }
            Ok(output(&state))
        }),
    }
}

pub struct Actor<S, V> {
    pub name: &'static str,
    components: Vec<ErasedComponent<S, V>>,
    merge: Arc<dyn Fn(Vec<V>) -> V + Send + Sync>,
}

impl<S, V> Actor<S, V>
where
    S: Send + Sync + 'static,
    V: 'static,
{
    pub fn new(
        name: &'static str,
        components: Vec<ErasedComponent<S, V>>,
        merge: impl Fn(Vec<V>) -> V + Send + Sync + 'static,
    ) -> Self {
        Self {
            name,
            components,
            merge: Arc::new(merge),
        }
    }

    fn project(&self, facts: &[Fact]) -> Result<(V, Vec<Transition<S>>), ActorError> {
        let mut views = Vec::with_capacity(self.components.len());
        let mut transitions = Vec::new();
        for component in &self.components {
            let (view, enabled) = (component.project)(facts)?;
            views.push(view);
            transitions.extend(enabled);
        }
        let mut seen = std::collections::BTreeSet::new();
        for transition in &transitions {
            if !seen.insert(transition.key.clone()) {
                return Err(ActorError::DuplicateTransition {
                    key: transition.key.clone(),
                });
            }
        }
        let causes: std::collections::BTreeSet<&str> = facts
            .iter()
            .filter_map(|fact| fact.cause.as_deref())
            .collect();
        transitions.retain(|transition| !causes.contains(transition.key.as_str()));
        transitions.sort_by(|left, right| left.key.cmp(&right.key));
        Ok(((self.merge)(views), transitions))
    }
}

pub struct Settlement<V> {
    pub view: V,
    pub log: Vec<Fact>,
    pub steps: u32,
    pub cut: LogCut,
    pub trace: Vec<TraceEvent>,
}

pub async fn resume<S, V>(
    actor: &Actor<S, V>,
    log: &dyn LogStore,
    services: Arc<S>,
    thread_id: &str,
    limit: u32,
) -> Result<Settlement<V>, Exit<ActorError>>
where
    S: Send + Sync + 'static,
    V: 'static,
{
    resume_with_cancel(
        actor,
        log,
        services,
        thread_id,
        limit,
        &CancellationToken::new(),
    )
    .await
}

pub async fn resume_with_cancel<S, V>(
    actor: &Actor<S, V>,
    log: &dyn LogStore,
    services: Arc<S>,
    thread_id: &str,
    limit: u32,
    cancel: &CancellationToken,
) -> Result<Settlement<V>, Exit<ActorError>>
where
    S: Send + Sync + 'static,
    V: 'static,
{
    let mut steps = 0;
    let mut trace_events = Vec::new();
    loop {
        let facts = match log.read(thread_id) {
            Ok(facts) => facts,
            Err(error) => return Err(Exit::Fail(error)),
        };
        let (view, transitions) = match actor.project(&facts) {
            Ok(projected) => projected,
            Err(error) => return Err(Exit::Fail(error)),
        };
        let Some(next) = transitions.into_iter().next() else {
            return Ok(Settlement {
                cut: cut_log(&facts),
                view,
                log: facts,
                steps,
                trace: trace_events,
            });
        };
        if steps >= limit {
            return Err(Exit::Fail(ActorError::StepLimit { limit }));
        }
        // The transition records its own cancel facts. Do not select around
        // this await: dropping it first would skip those facts, and a later
        // resume would run the same tool again.
        let run = match next.run {
            TransitionProgram::Immediate(run) => run,
            TransitionProgram::Prepared(prepare) => {
                let (prepared, span) = prepare
                    .with_span(format!("preparation:{}", next.key))
                    .run_with_cancel(Arc::clone(&services), cancel.clone())
                    .await;
                trace_events.extend(span);
                let prepared = prepared?;
                // Preparation deliberately has no execution cause. On recovery
                // the component sees the obligation and chooses reconciliation,
                // while only this selected continuation may execute the tool.
                log.append(thread_id, &prepared.facts, None)
                    .map_err(Exit::Fail)?;
                prepared.run
            }
        };
        let (result, span) = run
            .with_span(format!("transition:{}", next.key))
            .run_with_cancel(Arc::clone(&services), cancel.clone())
            .await;
        trace_events.extend(span);
        let produced = match result {
            Ok(produced) => produced,
            Err(error) => return Err(error),
        };
        if let Err(error) = log.append(thread_id, &produced, Some(next.key.as_str())) {
            return Err(Exit::Fail(error));
        }
        steps += 1;
        // A transition that completed and appended is durable work: settle
        // with it, never discard it. Cancellation is carried by the
        // transitions themselves — observed effects record their own
        // interrupt facts and quiesce the actor, and `Effect::from_async`
        // refuses to start once cancelled, which still leaves the loop with
        // [`Exit::Interrupt`].
    }
}

pub async fn ingest<S, V>(
    actor: &Actor<S, V>,
    log: &dyn LogStore,
    services: Arc<S>,
    thread_id: &str,
    fact: NewFact,
    limit: u32,
) -> Result<Settlement<V>, Exit<ActorError>>
where
    S: Send + Sync + 'static,
    V: 'static,
{
    ingest_with_cancel(
        actor,
        log,
        services,
        thread_id,
        fact,
        limit,
        &CancellationToken::new(),
    )
    .await
}

/// Append one host input to the existing actor log without resuming it.
///
/// The caller retains the original log and thread binding. This entry point
/// does not project components, select transitions, or acquire any services;
/// preparation and terminal facts still come from the actor's transition path.
pub fn append_input(log: &dyn LogStore, thread_id: &str, fact: NewFact) -> Result<(), ActorError> {
    log.append(thread_id, &[fact], None).map(|_| ())
}

pub async fn ingest_with_cancel<S, V>(
    actor: &Actor<S, V>,
    log: &dyn LogStore,
    services: Arc<S>,
    thread_id: &str,
    fact: NewFact,
    limit: u32,
    cancel: &CancellationToken,
) -> Result<Settlement<V>, Exit<ActorError>>
where
    S: Send + Sync + 'static,
    V: 'static,
{
    append_input(log, thread_id, fact).map_err(Exit::Fail)?;
    resume_with_cancel(actor, log, services, thread_id, limit, cancel).await
}

#[cfg(test)]
#[path = "actor/tests.rs"]
mod tests;
