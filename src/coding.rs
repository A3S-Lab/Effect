//! Coding harness projected from the log.
//!
//! The scheduler does not keep a parked question or a tool confirmation in
//! process memory. Those waits are phases of the fold. The runtime runs a
//! transition only when the fold enables one. Answering or confirming appends
//! a fact, and the next `resume` derives the tool or model call from that fact.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use serde::Deserialize;
use serde_json::{json, Value};

use crate::actor::{component, ingest, resume, Actor, ErasedComponent, Settlement, Transition};
use crate::effect::{Effect, Schedule};
use crate::error::ActorError;
use crate::exit::Exit;
use crate::fact::{Fact, NewFact};

pub type BoxFuture<T> = Pin<Box<dyn Future<Output = T> + Send>>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolSpec {
    pub name: String,
    pub description: String,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub args: Value,
    pub needs_confirmation: bool,
    /// Prose that accompanied the call. The fold does not branch on it.
    /// Replaying it is what lets the next model call see its own plan.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    /// Provider reasoning that accompanied the call. The fold does not branch
    /// on it. Absent on logs written before this field existed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<String>,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ModelDecision {
    Text {
        text: String,
    },
    Tool {
        call: ToolCall,
    },
    Question {
        question_id: String,
        question: String,
        allow_free_text: bool,
        /// Options the host renders after the log is reopened. Absent JSON
        /// fields decode as an empty list.
        #[serde(default)]
        options: Vec<String>,
    },
}

#[derive(Debug, Clone)]
pub struct CompletionRequest {
    pub system: Vec<String>,
    pub tools: Vec<ToolSpec>,
    pub summary: String,
    pub messages: Vec<String>,
}

pub trait Completion: Send + Sync {
    fn complete(&self, request: CompletionRequest) -> BoxFuture<Result<ModelDecision, ActorError>>;
}

pub trait ToolRunner: Send + Sync {
    fn run(&self, call: ToolCall) -> BoxFuture<Result<Value, ActorError>>;
}

pub trait Compactor: Send + Sync {
    fn compact(&self, messages: &[String]) -> BoxFuture<Result<String, ActorError>>;
}

#[derive(Clone)]
pub struct CodingServices {
    pub completion: Arc<dyn Completion>,
    pub tools: Arc<dyn ToolRunner>,
    pub compactor: Arc<dyn Compactor>,
}

/// Configuration checked before an actor exists. A zero step limit or a zero
/// model attempt count cannot construct a harness.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HarnessConfig {
    budget: u32,
    compact_after_chars: usize,
    step_limit: u32,
    model_attempts: u32,
    system: Vec<String>,
    tools: Vec<ToolSpec>,
    /// After this many successful tool results in the current turn, the next
    /// completion request carries an empty tool list. `None` does not cap.
    tool_round_cap: Option<u32>,
}

impl HarnessConfig {
    pub fn new(
        budget: u32,
        compact_after_chars: usize,
        step_limit: u32,
        model_attempts: u32,
        system: Vec<String>,
        tools: Vec<ToolSpec>,
    ) -> Result<Self, ActorError> {
        if step_limit == 0 {
            return Err(ActorError::Config("step_limit must be at least 1".into()));
        }
        if model_attempts == 0 {
            return Err(ActorError::Config(
                "model_attempts must be at least 1".into(),
            ));
        }
        Ok(Self {
            budget,
            compact_after_chars,
            step_limit,
            model_attempts,
            system,
            tools,
            tool_round_cap: None,
        })
    }

    /// One later completion sees no tools once this many tool results exist.
    /// The caller does not inject a user message to force that completion.
    pub fn with_tool_round_cap(mut self, cap: u32) -> Self {
        self.tool_round_cap = Some(cap);
        self
    }

    pub fn step_limit(&self) -> u32 {
        self.step_limit
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct PendingConfirmation {
    pub tool_call_id: String,
    pub name: String,
    pub args: Value,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PendingQuestion {
    pub question_id: String,
    pub question: String,
    pub allow_free_text: bool,
    pub options: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CodingPhase {
    Idle,
    Infer,
    Compact,
    Confirm,
    Question,
    Tool,
    Deny,
    Done,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CodingView {
    pub system: Vec<String>,
    pub tools: Vec<ToolSpec>,
    pub pending_confirmation: Option<PendingConfirmation>,
    pub pending_question: Option<PendingQuestion>,
    pub assistant: Option<String>,
    pub phase: CodingPhase,
    pub schema_error: Option<String>,
}

impl CodingView {
    fn empty() -> Self {
        Self {
            system: Vec::new(),
            tools: Vec::new(),
            pending_confirmation: None,
            pending_question: None,
            assistant: None,
            phase: CodingPhase::Idle,
            schema_error: None,
        }
    }
}

pub fn merge_coding_view(views: Vec<CodingView>) -> CodingView {
    CodingView {
        system: views.iter().flat_map(|view| view.system.clone()).collect(),
        tools: views.iter().flat_map(|view| view.tools.clone()).collect(),
        pending_confirmation: views
            .iter()
            .find_map(|view| view.pending_confirmation.clone()),
        pending_question: views.iter().find_map(|view| view.pending_question.clone()),
        assistant: views.iter().rev().find_map(|view| view.assistant.clone()),
        phase: views
            .iter()
            .rev()
            .find(|view| view.phase != CodingPhase::Idle)
            .map(|view| view.phase)
            .unwrap_or(CodingPhase::Idle),
        schema_error: views.iter().find_map(|view| view.schema_error.clone()),
    }
}

#[derive(Debug, Clone)]
struct SchedulerState {
    turn: u64,
    cycle: u64,
    phase: CodingPhase,
    messages: Vec<String>,
    summary: String,
    tool_runs: u32,
    compacted_turn: u64,
    assistant: Option<String>,
    pending_call: Option<ToolCall>,
    pending_question: Option<PendingQuestion>,
    schema_error: Option<String>,
}

impl SchedulerState {
    fn new() -> Self {
        Self {
            turn: 0,
            cycle: 0,
            phase: CodingPhase::Idle,
            messages: Vec::new(),
            summary: String::new(),
            tool_runs: 0,
            compacted_turn: 0,
            assistant: None,
            pending_call: None,
            pending_question: None,
            schema_error: None,
        }
    }

    fn size(&self) -> usize {
        self.summary.len() + self.messages.iter().map(String::len).sum::<usize>()
    }
}

fn after_input(state: SchedulerState, config: &HarnessConfig) -> SchedulerState {
    let needs_compact =
        state.size() >= config.compact_after_chars && state.compacted_turn != state.turn;
    SchedulerState {
        phase: if needs_compact {
            CodingPhase::Compact
        } else {
            CodingPhase::Infer
        },
        assistant: None,
        pending_call: None,
        pending_question: None,
        ..state
    }
}

fn on_model(mut state: SchedulerState, config: &HarnessConfig, payload: &Value) -> SchedulerState {
    let decided = match serde_json::from_value::<ModelDecision>(payload.clone()) {
        Ok(decided) => decided,
        Err(error) => {
            state.phase = CodingPhase::Done;
            state.schema_error = Some(error.to_string());
            state.pending_call = None;
            return state;
        }
    };
    match decided {
        ModelDecision::Text { text } => SchedulerState {
            phase: CodingPhase::Done,
            assistant: Some(text),
            pending_call: None,
            ..state
        },
        ModelDecision::Question {
            question_id,
            question,
            allow_free_text,
            options,
        } => SchedulerState {
            phase: CodingPhase::Question,
            pending_question: Some(PendingQuestion {
                question_id,
                question,
                allow_free_text,
                options,
            }),
            pending_call: None,
            ..state
        },
        ModelDecision::Tool { call } => {
            if state.tool_runs >= config.budget {
                SchedulerState {
                    phase: CodingPhase::Deny,
                    pending_call: Some(call),
                    ..state
                }
            } else if call.needs_confirmation {
                SchedulerState {
                    phase: CodingPhase::Confirm,
                    pending_call: Some(call),
                    ..state
                }
            } else {
                SchedulerState {
                    phase: CodingPhase::Tool,
                    pending_call: Some(call),
                    ..state
                }
            }
        }
    }
}

#[derive(Deserialize)]
struct TextPayload {
    text: String,
}

#[derive(Deserialize)]
struct SummaryPayload {
    summary: String,
}

#[derive(Deserialize)]
struct ConfirmationPayload {
    tool_call_id: String,
    approved: bool,
}

#[derive(Deserialize)]
struct ToolResultPayload {
    ok: bool,
    output: String,
}

fn step(mut state: SchedulerState, fact: &Fact, config: &HarnessConfig) -> SchedulerState {
    match fact.kind.as_str() {
        "user.message" => match serde_json::from_value::<TextPayload>(fact.payload.clone()) {
            Ok(payload) => after_input(
                SchedulerState {
                    turn: state.turn + 1,
                    cycle: 0,
                    messages: {
                        let mut messages = state.messages.clone();
                        messages.push(format!("user\n{}", payload.text));
                        messages
                    },
                    tool_runs: 0,
                    ..state
                },
                config,
            ),
            Err(error) => {
                state.phase = CodingPhase::Done;
                state.schema_error = Some(error.to_string());
                state
            }
        },
        "compaction.done" => match serde_json::from_value::<SummaryPayload>(fact.payload.clone()) {
            Ok(payload) => SchedulerState {
                summary: payload.summary,
                compacted_turn: state.turn,
                phase: CodingPhase::Infer,
                messages: Vec::new(),
                ..state
            },
            Err(error) => {
                state.phase = CodingPhase::Done;
                state.schema_error = Some(error.to_string());
                state
            }
        },
        "model.turn" => on_model(state, config, &fact.payload),
        "confirmation.answered" => {
            let Some(call) = state.pending_call.clone() else {
                return state;
            };
            match serde_json::from_value::<ConfirmationPayload>(fact.payload.clone()) {
                Ok(payload) if payload.tool_call_id == call.id && payload.approved => {
                    SchedulerState {
                        phase: CodingPhase::Tool,
                        pending_question: None,
                        ..state
                    }
                }
                Ok(payload) if payload.tool_call_id == call.id => SchedulerState {
                    phase: CodingPhase::Done,
                    assistant: Some("denied".into()),
                    pending_call: None,
                    ..state
                },
                Ok(_) => state,
                Err(error) => {
                    state.schema_error = Some(error.to_string());
                    state.phase = CodingPhase::Done;
                    state
                }
            }
        }
        "tool.result" => match serde_json::from_value::<ToolResultPayload>(fact.payload.clone()) {
            Ok(payload) => after_input(
                SchedulerState {
                    tool_runs: state.tool_runs + u32::from(payload.ok),
                    cycle: state.cycle + 1,
                    messages: {
                        let mut messages = state.messages.clone();
                        messages.push(format!("tool\n{}", payload.output));
                        messages
                    },
                    pending_call: None,
                    ..state
                },
                config,
            ),
            Err(error) => {
                state.schema_error = Some(error.to_string());
                state.phase = CodingPhase::Done;
                state
            }
        },
        "question.answered" => match serde_json::from_value::<TextPayload>(fact.payload.clone()) {
            Ok(payload) => after_input(
                SchedulerState {
                    cycle: state.cycle + 1,
                    messages: {
                        let mut messages = state.messages.clone();
                        messages.push(format!("user\n{}", payload.text));
                        messages
                    },
                    pending_question: None,
                    ..state
                },
                config,
            ),
            Err(error) => {
                state.schema_error = Some(error.to_string());
                state.phase = CodingPhase::Done;
                state
            }
        },
        "budget.denied" => SchedulerState {
            phase: CodingPhase::Done,
            assistant: Some("budget".into()),
            pending_call: None,
            ..state
        },
        _ => state,
    }
}

fn view_of(state: &SchedulerState) -> CodingView {
    CodingView {
        pending_confirmation: match (&state.phase, &state.pending_call) {
            (CodingPhase::Confirm, Some(call)) => Some(PendingConfirmation {
                tool_call_id: call.id.clone(),
                name: call.name.clone(),
                args: call.args.clone(),
            }),
            _ => None,
        },
        pending_question: match state.phase {
            CodingPhase::Question => state.pending_question.clone(),
            _ => None,
        },
        assistant: state.assistant.clone(),
        phase: state.phase,
        schema_error: state.schema_error.clone(),
        ..CodingView::empty()
    }
}

fn transitions_of(
    state: &SchedulerState,
    config: &HarnessConfig,
) -> Vec<Transition<CodingServices>> {
    match state.phase {
        CodingPhase::Compact => {
            let mut messages = state.messages.clone();
            if !state.summary.is_empty() {
                messages.insert(0, state.summary.clone());
            }
            let turn = state.turn;
            vec![Transition {
                key: format!("compact:{turn}"),
                run: Effect::from_async(move |services: Arc<CodingServices>, _cancel| {
                    let messages = messages.clone();
                    async move {
                        let summary = services.compactor.compact(&messages).await?;
                        Ok(vec![NewFact {
                            kind: "compaction.done".into(),
                            key: format!("compaction:{turn}"),
                            payload: json!({ "summary": summary }),
                        }])
                    }
                }),
            }]
        }
        CodingPhase::Infer => {
            let tools = match config.tool_round_cap {
                Some(cap) if state.tool_runs >= cap => Vec::new(),
                _ => config.tools.clone(),
            };
            let request = CompletionRequest {
                system: config.system.clone(),
                tools,
                summary: state.summary.clone(),
                messages: state.messages.clone(),
            };
            let turn = state.turn;
            let cycle = state.cycle;
            let attempts = config.model_attempts;
            vec![Transition {
                key: format!("infer:{turn}:{cycle}"),
                run: Effect::from_async(move |services: Arc<CodingServices>, _cancel| {
                    let request = request.clone();
                    async move {
                        let decided = match services.completion.complete(request).await {
                            Ok(decided) => decided,
                            Err(ActorError::Defect(message)) => return Err(Exit::Die(message)),
                            Err(error) => return Err(Exit::Fail(error)),
                        };
                        Ok(vec![NewFact {
                            kind: "model.turn".into(),
                            key: format!("model:{turn}:{cycle}"),
                            payload: serde_json::to_value(&decided)
                                .map_err(|error| Exit::Die(error.to_string()))?,
                        }])
                    }
                })
                .retry(Schedule {
                    remaining: attempts.saturating_sub(1),
                    delay: std::time::Duration::ZERO,
                }),
            }]
        }
        CodingPhase::Tool => {
            let Some(call) = state.pending_call.clone() else {
                return Vec::new();
            };
            vec![Transition {
                key: format!("tool:{}", call.id),
                run: Effect::from_async(move |services: Arc<CodingServices>, _cancel| {
                    let call = call.clone();
                    async move {
                        let output = services.tools.run(call.clone()).await?;
                        let output = match output {
                            Value::String(text) => text,
                            other => other.to_string(),
                        };
                        Ok(vec![NewFact {
                            kind: "tool.result".into(),
                            key: format!("tool-result:{}", call.id),
                            payload: json!({ "toolCallId": call.id, "ok": true, "output": output }),
                        }])
                    }
                }),
            }]
        }
        CodingPhase::Deny => {
            let Some(call) = state.pending_call.clone() else {
                return Vec::new();
            };
            vec![Transition {
                key: format!("budget:{}", call.id),
                run: Effect::succeed(vec![NewFact {
                    kind: "budget.denied".into(),
                    key: format!("budget-denied:{}", call.id),
                    payload: json!({ "toolCallId": call.id }),
                }]),
            }]
        }
        CodingPhase::Idle | CodingPhase::Confirm | CodingPhase::Question | CodingPhase::Done => {
            Vec::new()
        }
    }
}

fn scheduler(config: HarnessConfig) -> ErasedComponent<CodingServices, CodingView> {
    let step_config = config.clone();
    let output_config = config;
    component(
        SchedulerState::new,
        move |state, fact| step(state, fact, &step_config),
        move |state| (view_of(state), transitions_of(state, &output_config)),
    )
}

fn instructions(config: &HarnessConfig) -> ErasedComponent<CodingServices, CodingView> {
    let system = config.system.clone();
    component(
        || (),
        |state, _fact| state,
        move |_state| {
            (
                CodingView {
                    system: system.clone(),
                    ..CodingView::empty()
                },
                Vec::new(),
            )
        },
    )
}

fn catalog(config: &HarnessConfig) -> ErasedComponent<CodingServices, CodingView> {
    let tools = config.tools.clone();
    component(
        || (),
        |state, _fact| state,
        move |_state| {
            (
                CodingView {
                    tools: tools.clone(),
                    ..CodingView::empty()
                },
                Vec::new(),
            )
        },
    )
}

pub fn coding_actor(config: HarnessConfig) -> Actor<CodingServices, CodingView> {
    let system = instructions(&config);
    let tools = catalog(&config);
    let schedule = scheduler(config);
    Actor::new("a3s-code", vec![system, tools, schedule], merge_coding_view)
}

pub fn message_fact(key: impl Into<String>, text: impl Into<String>) -> NewFact {
    NewFact {
        kind: "user.message".into(),
        key: key.into(),
        payload: json!({ "text": text.into() }),
    }
}

pub fn confirm_fact(
    key: impl Into<String>,
    tool_call_id: impl Into<String>,
    approved: bool,
) -> NewFact {
    NewFact {
        kind: "confirmation.answered".into(),
        key: key.into(),
        payload: json!({ "tool_call_id": tool_call_id.into(), "approved": approved }),
    }
}

pub fn answer_fact(key: impl Into<String>, text: impl Into<String>) -> NewFact {
    NewFact {
        kind: "question.answered".into(),
        key: key.into(),
        payload: json!({ "text": text.into() }),
    }
}

pub async fn ingest_coding(
    actor: &Actor<CodingServices, CodingView>,
    log: &dyn crate::log::LogStore,
    services: Arc<CodingServices>,
    thread_id: &str,
    fact: NewFact,
    limit: u32,
) -> Result<Settlement<CodingView>, Exit<ActorError>> {
    ingest(actor, log, services, thread_id, fact, limit).await
}

pub async fn resume_coding(
    actor: &Actor<CodingServices, CodingView>,
    log: &dyn crate::log::LogStore,
    services: Arc<CodingServices>,
    thread_id: &str,
    limit: u32,
) -> Result<Settlement<CodingView>, Exit<ActorError>> {
    resume(actor, log, services, thread_id, limit).await
}
