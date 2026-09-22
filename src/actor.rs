//! An actor is a fold over an immutable log.
//!
//! Each component is a Moore machine: `step` consumes one fact, `output`
//! returns the view and the transitions that state enables. The runtime is
//! the edge that runs those transitions. A transition that has already left
//! a `cause` in the log is not selected again, so resuming a thread continues
//! unfinished work and does not repeat finished work.

use std::sync::Arc;

use crate::effect::{Effect, TraceEvent};
use crate::error::ActorError;
use crate::exit::Exit;
use crate::fact::{cut_log, Fact, LogCut, NewFact};
use crate::log::LogStore;

pub struct Transition<S> {
    pub key: String,
    pub run: Effect<Vec<NewFact>, ActorError, S>,
}

pub struct ErasedComponent<S, V> {
    project: Arc<dyn Fn(&[Fact]) -> Result<(V, Vec<Transition<S>>), ActorError> + Send + Sync>,
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
        let (result, span) = next
            .run
            .with_span(format!("transition:{}", next.key))
            .run(Arc::clone(&services))
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
    log.append(thread_id, &[fact], None).map_err(Exit::Fail)?;
    resume(actor, log, services, thread_id, limit).await
}
