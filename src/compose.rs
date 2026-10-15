//! Meta Harness composition: mount Moore components on one fact log.
//!
//! Hosts assemble `system`, `tools`, `budget`, `compact`, and `infer([...])`
//! via an ordered `components: [...]` list. The product default remains
//! [`coding_actor`](crate::coding::coding_actor), which is sugar for the stock
//! tree. Nested `infer` mounts namespace child transition keys so siblings
//! cannot collide (`DuplicateTransition`). The stock Infer part mounts the
//! scheduler **without** a key prefix so existing fact-log causes stay stable.

use std::sync::Arc;

use crate::actor::{component, Actor, ErasedComponent, Transition};
use crate::coding::{
    merge_coding_view, CodingServices, CodingView, Completion, CompletionRequest, HarnessConfig,
    ModelDecision, ToolSpec,
};
use crate::error::ActorError;

/// Documented merge slots for the coding Meta Harness view.
///
/// Same shape as [`CodingView`]. Prefer this name in composition APIs.
pub type HarnessView = CodingView;

/// Prefix every enabled transition key.
pub fn namespace_transitions<S>(
    prefix: &str,
    transitions: Vec<Transition<S>>,
) -> Vec<Transition<S>> {
    let prefix = prefix.trim_matches('/');
    transitions
        .into_iter()
        .map(|transition| Transition {
            key: if prefix.is_empty() {
                transition.key
            } else {
                format!("{prefix}/{}", transition.key)
            },
            run: transition.run,
        })
        .collect()
}

/// Wrap a component so every transition it enables carries `prefix/`.
pub fn with_key_namespace<S, V>(
    prefix: impl Into<String>,
    inner: ErasedComponent<S, V>,
) -> ErasedComponent<S, V>
where
    S: Send + Sync + 'static,
    V: 'static,
{
    let prefix = prefix.into();
    let project = inner.project;
    ErasedComponent {
        project: Arc::new(move |facts| {
            let (view, transitions) = (project)(facts)?;
            Ok((view, namespace_transitions(&prefix, transitions)))
        }),
    }
}

/// Mount several components as one. Views merge left-to-right with `merge`;
/// transitions are concatenated then namespaced under `prefix`.
pub fn mount<S, V>(
    prefix: impl Into<String>,
    children: Vec<ErasedComponent<S, V>>,
    merge: impl Fn(Vec<V>) -> V + Send + Sync + 'static,
) -> ErasedComponent<S, V>
where
    S: Send + Sync + 'static,
    V: 'static,
{
    let prefix = prefix.into();
    let merge = Arc::new(merge);
    let children = Arc::new(children);
    ErasedComponent {
        project: Arc::new(move |facts| {
            let mut views = Vec::with_capacity(children.len());
            let mut transitions = Vec::new();
            for child in children.iter() {
                let (view, enabled) = (child.project)(facts)?;
                views.push(view);
                transitions.extend(enabled);
            }
            let view = (merge)(views);
            Ok((view, namespace_transitions(&prefix, transitions)))
        }),
    }
}

/// Nested `infer([...])` with a key-space prefix.
///
/// Use for host-authored subtrees. The stock Meta Harness Infer part mounts
/// the scheduler without this prefix so persisted cause keys stay stable.
pub fn infer(
    children: Vec<ErasedComponent<CodingServices, HarnessView>>,
) -> ErasedComponent<CodingServices, HarnessView> {
    mount("infer", children, merge_coding_view)
}

/// System-prompt slot component.
pub fn system(prompts: Vec<String>) -> ErasedComponent<CodingServices, HarnessView> {
    component(
        || (),
        |state, _fact| state,
        move |_state| {
            (
                CodingView {
                    system: prompts.clone(),
                    ..CodingView::empty()
                },
                Vec::new(),
            )
        },
    )
}

/// Tool-catalog slot component.
pub fn tools(specs: Vec<ToolSpec>) -> ErasedComponent<CodingServices, HarnessView> {
    component(
        || (),
        |state, _fact| state,
        move |_state| {
            (
                CodingView {
                    tools: specs.clone(),
                    ..CodingView::empty()
                },
                Vec::new(),
            )
        },
    )
}

/// First-class budget policy slot (compose-tree presence + view observability).
///
/// The stock infer scheduler still enforces the numeric limit from
/// [`HarnessConfig`].
pub fn budget(limit: u32) -> ErasedComponent<CodingServices, HarnessView> {
    component(
        move || limit,
        |state, _fact| state,
        move |state| {
            (
                CodingView {
                    tool_budget: Some(*state),
                    ..CodingView::empty()
                },
                Vec::new(),
            )
        },
    )
}

/// First-class compaction threshold slot.
pub fn compact(after_chars: usize) -> ErasedComponent<CodingServices, HarnessView> {
    component(
        move || after_chars,
        |state, _fact| state,
        move |state| {
            (
                CodingView {
                    compact_after_chars: Some(*state),
                    ..CodingView::empty()
                },
                Vec::new(),
            )
        },
    )
}

/// Declarative Meta Harness recipe admitted by Code / SDKs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MetaHarnessSpec {
    pub name: &'static str,
    pub budget: u32,
    pub compact_after_chars: usize,
    pub step_limit: u32,
    pub model_attempts: u32,
    pub system: Vec<String>,
    pub tools: Vec<ToolSpec>,
    pub tool_round_cap: Option<u32>,
    /// Ordered part ids. Empty means the stock tree
    /// `system + tools + budget + compact + infer(scheduler)`.
    pub parts: Vec<HarnessPartId>,
}

/// Stock part identifiers for compose recipes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HarnessPartId {
    System,
    Tools,
    Budget,
    Compact,
    Infer,
}

impl Default for MetaHarnessSpec {
    fn default() -> Self {
        Self {
            name: "a3s-code",
            budget: 8,
            compact_after_chars: 1_000_000,
            step_limit: 32,
            model_attempts: 2,
            system: Vec::new(),
            tools: Vec::new(),
            tool_round_cap: None,
            parts: Vec::new(),
        }
    }
}

impl MetaHarnessSpec {
    pub fn into_config(self) -> Result<HarnessConfig, ActorError> {
        let mut config = HarnessConfig::new(
            self.budget,
            self.compact_after_chars,
            self.step_limit,
            self.model_attempts,
            self.system,
            self.tools,
        )?;
        if let Some(cap) = self.tool_round_cap {
            config = config.with_tool_round_cap(cap);
        }
        Ok(config)
    }

    /// The recipe's fields over a session config. Token compaction, which a
    /// recipe cannot name, stays from `base`; a character threshold that
    /// differs from `base` is an explicit choice and replaces it.
    pub fn into_config_over(self, base: &HarnessConfig) -> Result<HarnessConfig, ActorError> {
        let keeps_token_compact = self.compact_after_chars == base.compact_after_chars();
        let config = self.into_config()?;
        Ok(if keeps_token_compact {
            config.with_token_compact_of(base)
        } else {
            config
        })
    }

    pub fn resolved_parts(&self) -> Vec<HarnessPartId> {
        if self.parts.is_empty() {
            vec![
                HarnessPartId::System,
                HarnessPartId::Tools,
                HarnessPartId::Budget,
                HarnessPartId::Compact,
                HarnessPartId::Infer,
            ]
        } else {
            self.parts.clone()
        }
    }
}

/// Build an actor from an ordered list of already-constructed components.
pub fn compose_coding_actor(
    name: &'static str,
    components: Vec<ErasedComponent<CodingServices, HarnessView>>,
) -> Actor<CodingServices, HarnessView> {
    let mounted = mount("", components, merge_coding_view);
    let bound = ErasedComponent {
        project: Arc::new(move |facts| {
            let (view, transitions) = (mounted.project)(facts)?;
            let system = Arc::new(view.system.clone());
            let transitions = transitions
                .into_iter()
                .map(|transition| {
                    let system = Arc::clone(&system);
                    Transition {
                        key: transition.key,
                        run: transition
                            .run
                            .map_services(move |services: Arc<CodingServices>| {
                                Arc::new(CodingServices {
                                    completion: Arc::new(ProjectedCompletion {
                                        inner: Arc::clone(&services.completion),
                                        system: Arc::clone(&system),
                                    }),
                                    tools: Arc::clone(&services.tools),
                                    compactor: Arc::clone(&services.compactor),
                                })
                            }),
                    }
                })
                .collect();
            Ok((view, transitions))
        }),
    };
    Actor::new(name, vec![bound], merge_coding_view)
}

struct ProjectedCompletion {
    inner: Arc<dyn Completion>,
    system: Arc<Vec<String>>,
}

impl Completion for ProjectedCompletion {
    fn complete(
        &self,
        mut request: CompletionRequest,
        cancel: tokio_util::sync::CancellationToken,
    ) -> crate::coding::BoxFuture<Result<ModelDecision, ActorError>> {
        request.system = self.system.as_ref().clone();
        self.inner.complete(request, cancel)
    }

    fn reported_prompt_tokens(&self) -> Option<u64> {
        self.inner.reported_prompt_tokens()
    }
}

#[cfg(test)]
#[path = "compose_execution_tests.rs"]
mod execution_tests;

fn compose_meta_harness_with<F>(
    spec: MetaHarnessSpec,
    mut make_scheduler: F,
) -> Actor<CodingServices, HarnessView>
where
    F: FnMut() -> ErasedComponent<CodingServices, HarnessView>,
{
    let name = spec.name;
    let parts = spec.resolved_parts();
    let system_prompts = spec.system.clone();
    let tool_specs = spec.tools.clone();
    let budget_limit = spec.budget;
    let compact_after = spec.compact_after_chars;

    let mut components = Vec::new();
    for part in parts {
        match part {
            HarnessPartId::System => components.push(system(system_prompts.clone())),
            HarnessPartId::Tools => components.push(tools(tool_specs.clone())),
            HarnessPartId::Budget => components.push(budget(budget_limit)),
            HarnessPartId::Compact => components.push(compact(compact_after)),
            // Stock Infer mounts the scheduler without a key prefix so cause
            // keys (`infer:1:0`, `compact:1`, …) remain stable across resumes.
            HarnessPartId::Infer => components.push(make_scheduler()),
        }
    }
    compose_coding_actor(name, components)
}

/// Admitted harness graph: stock coding actor or a composed tree.
pub struct HarnessGraph {
    actor: Actor<CodingServices, HarnessView>,
}

impl HarnessGraph {
    pub fn from_actor(actor: Actor<CodingServices, HarnessView>) -> Self {
        Self { actor }
    }

    pub fn coding(config: HarnessConfig) -> Self {
        Self {
            actor: crate::coding::coding_actor(config),
        }
    }

    /// Compose from a declarative spec. `make_scheduler` is called once per
    /// `Infer` part (stock recipes include Infer exactly once).
    pub fn from_spec<F>(spec: MetaHarnessSpec, mut make_scheduler: F) -> Self
    where
        F: FnMut() -> ErasedComponent<CodingServices, HarnessView>,
    {
        Self {
            actor: compose_meta_harness_with(spec, &mut make_scheduler),
        }
    }

    pub fn actor(&self) -> &Actor<CodingServices, HarnessView> {
        &self.actor
    }

    pub fn into_actor(self) -> Actor<CodingServices, HarnessView> {
        self.actor
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::coding::coding_scheduler;
    use crate::effect::Effect;

    #[test]
    fn namespace_prefixes_keys() {
        let transitions = namespace_transitions(
            "infer",
            vec![Transition::<()> {
                key: "compact:1".into(),
                run: Effect::succeed(Vec::new()).into(),
            }],
        );
        assert_eq!(transitions[0].key, "infer/compact:1");
    }

    #[test]
    fn stock_spec_parts_include_budget_and_compact() {
        let spec = MetaHarnessSpec::default();
        assert_eq!(
            spec.resolved_parts(),
            vec![
                HarnessPartId::System,
                HarnessPartId::Tools,
                HarnessPartId::Budget,
                HarnessPartId::Compact,
                HarnessPartId::Infer,
            ]
        );
    }

    #[test]
    fn compose_meta_harness_builds() {
        let config = HarnessConfig::new(
            2,
            100,
            8,
            1,
            vec!["sys".into()],
            vec![ToolSpec {
                name: "read".into(),
                description: "r".into(),
            }],
        )
        .unwrap();
        let spec = MetaHarnessSpec {
            name: "test",
            budget: 2,
            compact_after_chars: 100,
            step_limit: 8,
            model_attempts: 1,
            system: vec!["sys".into()],
            tools: config.tools().to_vec(),
            tool_round_cap: None,
            parts: Vec::new(),
        };
        let graph = HarnessGraph::from_spec(spec, || coding_scheduler(config.clone()));
        assert_eq!(graph.actor().name, "test");
    }

    #[test]
    fn spec_over_a_session_config_keeps_token_compaction_unless_chars_are_explicit() {
        let base = HarnessConfig::new(4, 100, 8, 1, Vec::new(), Vec::new())
            .unwrap()
            .with_token_compact(800, 200);
        let spec = |chars| MetaHarnessSpec {
            budget: 4,
            compact_after_chars: chars,
            step_limit: 8,
            model_attempts: 1,
            ..MetaHarnessSpec::default()
        };
        assert_eq!(
            spec(base.compact_after_chars())
                .into_config_over(&base)
                .unwrap(),
            base
        );
        let explicit = spec(5_000).into_config_over(&base).unwrap();
        assert_eq!(
            explicit,
            HarnessConfig::new(4, 5_000, 8, 1, Vec::new(), Vec::new()).unwrap()
        );
    }

    #[test]
    fn custom_compose_includes_host_component() {
        let host = component(
            || (),
            |state, _fact| state,
            |_state| {
                (
                    CodingView {
                        system: vec!["host-slot".into()],
                        ..CodingView::empty()
                    },
                    Vec::new(),
                )
            },
        );
        let actor = compose_coding_actor(
            "custom",
            vec![
                system(vec!["base".into()]),
                tools(Vec::new()),
                budget(1),
                compact(64),
                host,
            ],
        );
        assert_eq!(actor.name, "custom");
    }
}
