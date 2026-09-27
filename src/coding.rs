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
        /// Further calls from the same model message. Empty on older logs.
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        also: Vec<ToolCall>,
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

    /// Provider-reported prompt tokens for the completion that just returned.
    /// `None` when the provider omitted usage. The infer transition copies this
    /// onto the `model.turn` fact so a later fold can compact from it.
    fn reported_prompt_tokens(&self) -> Option<u64> {
        None
    }
}

pub trait ToolRunner: Send + Sync {
    fn run(&self, call: ToolCall) -> BoxFuture<Result<Value, ActorError>>;

    /// True when every call in the batch can run at the same time.
    /// The default keeps calls in order, one after another.
    fn can_parallelize(&self, _calls: &[ToolCall]) -> bool {
        false
    }
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
    /// Compact when the previous model turn reported at least this many prompt
    /// tokens. `None` keeps the character threshold.
    compact_after_tokens: Option<u64>,
    /// Recent transcript to keep verbatim, in estimated tokens. `0` keeps only
    /// the latest tool call and its result.
    compact_keep_tokens: usize,
    /// After this many identical tool calls, the next one is not executed.
    /// The model receives a tool result that asks it to change approach.
    /// `None` does not check.
    duplicate_threshold: Option<u32>,
    /// Wait this long before replaying a failed model call. `ZERO` retries
    /// immediately. The host sets a jittered delay for provider failures.
    model_retry_delay: std::time::Duration,
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
            compact_after_tokens: None,
            compact_keep_tokens: 0,
            duplicate_threshold: None,
            model_retry_delay: std::time::Duration::ZERO,
        })
    }

    /// Compact from provider prompt tokens and keep a recent token budget.
    /// The character threshold no longer fires on this config.
    pub fn with_token_compact(mut self, after_tokens: u64, keep_tokens: usize) -> Self {
        self.compact_after_tokens = Some(after_tokens.max(1));
        self.compact_keep_tokens = keep_tokens.max(1);
        self.compact_after_chars = usize::MAX;
        self
    }

    /// Keep this many estimated tokens of recent transcript when a character
    /// threshold fires. `0` keeps only the latest tool pair.
    pub fn with_keep_tokens(mut self, keep_tokens: usize) -> Self {
        self.compact_keep_tokens = keep_tokens;
        self
    }

    /// Soft stop for an identical tool call. The call is not executed.
    pub fn with_duplicate_threshold(mut self, threshold: u32) -> Self {
        self.duplicate_threshold = Some(threshold);
        self
    }

    /// Delay before a failed model call is tried again.
    pub fn with_model_retry_delay(mut self, delay: std::time::Duration) -> Self {
        self.model_retry_delay = delay;
        self
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

    pub fn budget(&self) -> u32 {
        self.budget
    }

    pub fn compact_after_chars(&self) -> usize {
        self.compact_after_chars
    }

    pub fn model_attempts(&self) -> u32 {
        self.model_attempts
    }

    pub fn system(&self) -> &[String] {
        &self.system
    }

    pub fn tools(&self) -> &[ToolSpec] {
        &self.tools
    }

    pub fn tool_round_cap(&self) -> Option<u32> {
        self.tool_round_cap
    }

    pub fn model_retry_delay(&self) -> std::time::Duration {
        self.model_retry_delay
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
    /// Composed budget policy slot (`budget(...)` part).
    pub tool_budget: Option<u32>,
    /// Composed compaction threshold slot (`compact(...)` part).
    pub compact_after_chars: Option<usize>,
}

impl CodingView {
    pub fn empty() -> Self {
        Self {
            system: Vec::new(),
            tools: Vec::new(),
            pending_confirmation: None,
            pending_question: None,
            assistant: None,
            phase: CodingPhase::Idle,
            schema_error: None,
            tool_budget: None,
            compact_after_chars: None,
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
        tool_budget: views.iter().find_map(|view| view.tool_budget),
        compact_after_chars: views.iter().find_map(|view| view.compact_after_chars),
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
    /// Calls from the same model message that have not run yet.
    queued: Vec<ToolCall>,
    pending_question: Option<PendingQuestion>,
    schema_error: Option<String>,
    prompt_tokens: Option<u64>,
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
            queued: Vec::new(),
            pending_question: None,
            schema_error: None,
            prompt_tokens: None,
        }
    }

    fn size(&self) -> usize {
        self.summary.len() + self.messages.iter().map(String::len).sum::<usize>()
    }
}

fn after_input(state: SchedulerState, config: &HarnessConfig) -> SchedulerState {
    let (prefix, _) = split_compact_tail(&state.messages, config.compact_keep_tokens);
    let token_pressure = config
        .compact_after_tokens
        .is_some_and(|limit| state.prompt_tokens.is_some_and(|used| used >= limit));
    let char_pressure =
        config.compact_after_tokens.is_none() && state.size() >= config.compact_after_chars;
    let needs_compact = !prefix.is_empty()
        && (token_pressure || char_pressure)
        && state.compacted_turn != state.turn;
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

fn prompt_tokens_of(payload: &Value) -> Option<u64> {
    payload
        .get("prompt_tokens")
        .and_then(|value| value.as_u64())
        .filter(|tokens| *tokens > 0)
}

fn on_model(mut state: SchedulerState, config: &HarnessConfig, payload: &Value) -> SchedulerState {
    let prompt_tokens = prompt_tokens_of(payload).or(state.prompt_tokens);
    state.prompt_tokens = prompt_tokens;
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
        ModelDecision::Tool { call, also } => {
            let mut calls = Vec::with_capacity(1 + also.len());
            calls.push(call);
            calls.extend(also);
            let mut messages = state.messages.clone();
            for call in &calls {
                messages.push(format!("assistant\ntool {}\n{}", call.name, call.args));
            }
            let Some(first) = calls.first().cloned() else {
                return SchedulerState {
                    phase: CodingPhase::Done,
                    schema_error: Some("model tool decision had no call".into()),
                    ..state
                };
            };
            let queued = calls.into_iter().skip(1).collect();
            let recorded = SchedulerState {
                messages,
                queued,
                ..state
            };
            if recorded.tool_runs >= config.budget {
                SchedulerState {
                    phase: CodingPhase::Deny,
                    pending_call: Some(first),
                    ..recorded
                }
            } else if first.needs_confirmation {
                SchedulerState {
                    phase: CodingPhase::Confirm,
                    pending_call: Some(first),
                    ..recorded
                }
            } else {
                SchedulerState {
                    phase: CodingPhase::Tool,
                    pending_call: Some(first),
                    ..recorded
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
    #[serde(default)]
    kept: Vec<String>,
}

fn estimate_tokens(text: &str) -> usize {
    text.len().div_ceil(4).max(1)
}

fn is_tool_result_line(message: &str) -> bool {
    message.starts_with("tool\n") || message.starts_with("tool-error\n")
}

fn tool_result_line(ok: bool, output: &str) -> String {
    if ok {
        format!("tool\n{output}")
    } else {
        format!("tool-error\n{output}")
    }
}

fn promote_queued(mut state: SchedulerState) -> SchedulerState {
    if state.queued.is_empty() {
        state.pending_call = None;
        return state;
    }
    let next = state.queued.remove(0);
    state.phase = if next.needs_confirmation {
        CodingPhase::Confirm
    } else {
        CodingPhase::Tool
    };
    state.pending_call = Some(next);
    state.pending_question = None;
    state
}

/// Older lines are summarized. `keep_tokens == 0` retains only the newest tool
/// call and its result. A positive budget walks backward by tool pairs and
/// keeps every recent pair that still fits, always including the newest pair.
fn split_compact_tail(messages: &[String], keep_tokens: usize) -> (Vec<String>, Vec<String>) {
    if messages.is_empty() {
        return (Vec::new(), Vec::new());
    }
    if keep_tokens == 0 {
        let Some(tool_at) = messages
            .iter()
            .rposition(|message| is_tool_result_line(message))
        else {
            return (messages.to_vec(), Vec::new());
        };
        let start = if tool_at > 0 && messages[tool_at - 1].starts_with("assistant\ntool ") {
            tool_at - 1
        } else {
            tool_at
        };
        return (messages[..start].to_vec(), messages[start..].to_vec());
    }
    let mut units = Vec::new();
    let mut index = 0;
    while index < messages.len() {
        if index + 1 < messages.len()
            && messages[index].starts_with("assistant\ntool ")
            && is_tool_result_line(&messages[index + 1])
        {
            units.push((index, index + 2));
            index += 2;
        } else {
            units.push((index, index + 1));
            index += 1;
        }
    }
    let mut start_unit = units.len() - 1;
    let mut used = messages[units[start_unit].0..units[start_unit].1]
        .iter()
        .map(|line| estimate_tokens(line))
        .sum::<usize>();
    while start_unit > 0 {
        let previous = start_unit - 1;
        let extra = messages[units[previous].0..units[previous].1]
            .iter()
            .map(|line| estimate_tokens(line))
            .sum::<usize>();
        if used.saturating_add(extra) > keep_tokens {
            break;
        }
        used = used.saturating_add(extra);
        start_unit = previous;
    }
    let start = units[start_unit].0;
    (messages[..start].to_vec(), messages[start..].to_vec())
}

fn duplicate_nudge(messages: &[String], call: &ToolCall, threshold: Option<u32>) -> Option<String> {
    let threshold = threshold.filter(|value| *value > 0)?;
    let line = format!("assistant\ntool {}\n{}", call.name, call.args);
    let count = messages
        .iter()
        .filter(|message| message.as_str() == line)
        .count();
    if count <= threshold as usize {
        return None;
    }
    Some(format!(
        "Tool '{}' has been called {} times with identical arguments. Change your approach instead of repeating this call.",
        call.name, count
    ))
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
    /// When set, this result counts toward the tool ceiling even if `ok` is false.
    /// Absent logs count only successful results, which is the older behavior.
    #[serde(default, rename = "countRun")]
    count_run: Option<bool>,
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
                messages: payload.kept,
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
                Ok(payload) if payload.tool_call_id == call.id => {
                    if state.queued.is_empty() {
                        SchedulerState {
                            phase: CodingPhase::Done,
                            assistant: Some("denied".into()),
                            pending_call: None,
                            ..state
                        }
                    } else {
                        let mut continued = SchedulerState {
                            cycle: state.cycle + 1,
                            messages: {
                                let mut messages = state.messages.clone();
                                messages.push(tool_result_line(false, "denied"));
                                messages
                            },
                            tool_runs: state.tool_runs.saturating_add(1),
                            ..state
                        };
                        continued = promote_queued(continued);
                        continued
                    }
                }
                Ok(_) => state,
                Err(error) => {
                    state.schema_error = Some(error.to_string());
                    state.phase = CodingPhase::Done;
                    state
                }
            }
        }
        "tool.result" => match serde_json::from_value::<ToolResultPayload>(fact.payload.clone()) {
            Ok(payload) => {
                let counts = payload.count_run.unwrap_or(payload.ok);
                let folded = SchedulerState {
                    tool_runs: state.tool_runs + u32::from(counts),
                    cycle: state.cycle + 1,
                    messages: {
                        let mut messages = state.messages.clone();
                        messages.push(tool_result_line(payload.ok, &payload.output));
                        messages
                    },
                    ..state
                };
                let folded = promote_queued(folded);
                if folded.pending_call.is_none() {
                    after_input(folded, config)
                } else {
                    folded
                }
            }
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
            queued: Vec::new(),
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
            let (prefix, kept) = split_compact_tail(&state.messages, config.compact_keep_tokens);
            let mut to_summarize = prefix;
            if !state.summary.is_empty() {
                to_summarize.insert(0, state.summary.clone());
            }
            let turn = state.turn;
            vec![Transition {
                key: format!("compact:{turn}"),
                run: Effect::from_async(move |services: Arc<CodingServices>, _cancel| {
                    let to_summarize = to_summarize.clone();
                    let kept = kept.clone();
                    async move {
                        let summary = services.compactor.compact(&to_summarize).await?;
                        Ok(vec![NewFact {
                            kind: "compaction.done".into(),
                            key: format!("compaction:{turn}"),
                            payload: json!({ "summary": summary, "kept": kept }),
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
            let retry_delay = config.model_retry_delay;
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
                        let mut payload = serde_json::to_value(&decided)
                            .map_err(|error| Exit::Die(error.to_string()))?;
                        if let Some(tokens) = services.completion.reported_prompt_tokens() {
                            payload["prompt_tokens"] = json!(tokens);
                        }
                        Ok(vec![NewFact {
                            kind: "model.turn".into(),
                            key: format!("model:{turn}:{cycle}"),
                            payload,
                        }])
                    }
                })
                .retry(Schedule {
                    remaining: attempts.saturating_sub(1),
                    delay: retry_delay,
                }),
            }]
        }
        CodingPhase::Tool => {
            let Some(call) = state.pending_call.clone() else {
                return Vec::new();
            };
            let batch = ready_tool_batch(&call, &state.queued);
            let turn = state.turn;
            let cycle = state.cycle;
            let messages = state.messages.clone();
            let threshold = config.duplicate_threshold;
            let key = batch
                .iter()
                .map(|call| call.id.as_str())
                .collect::<Vec<_>>()
                .join("+");
            vec![Transition {
                key: format!("tool:{turn}:{cycle}:{key}"),
                run: Effect::from_async(move |services: Arc<CodingServices>, _cancel| {
                    let batch = batch.clone();
                    let messages = messages.clone();
                    async move {
                        run_tool_batch(services, batch, &messages, threshold, turn, cycle).await
                    }
                }),
            }]
        }
        CodingPhase::Deny => {
            let Some(call) = state.pending_call.clone() else {
                return Vec::new();
            };
            vec![Transition {
                key: format!("budget:{}:{}:{}", state.turn, state.cycle, call.id),
                run: Effect::succeed(vec![NewFact {
                    kind: "budget.denied".into(),
                    key: format!("budget-denied:{}:{}:{}", state.turn, state.cycle, call.id),
                    payload: json!({ "toolCallId": call.id }),
                }]),
            }]
        }
        CodingPhase::Idle | CodingPhase::Confirm | CodingPhase::Question | CodingPhase::Done => {
            Vec::new()
        }
    }
}

fn ready_tool_batch(pending: &ToolCall, queued: &[ToolCall]) -> Vec<ToolCall> {
    let mut batch = vec![pending.clone()];
    for call in queued {
        if call.needs_confirmation {
            break;
        }
        batch.push(call.clone());
    }
    batch
}

fn tool_output_text(value: &Value) -> String {
    match value {
        Value::String(text) => text.clone(),
        other => other.to_string(),
    }
}

fn tool_result_fact(
    turn: u64,
    cycle: u64,
    id: &str,
    ok: bool,
    output: String,
    count_run: bool,
) -> NewFact {
    let mut payload = json!({ "toolCallId": id, "ok": ok, "output": output });
    if count_run {
        payload["countRun"] = json!(true);
    }
    NewFact {
        kind: "tool.result".into(),
        key: format!("tool-result:{turn}:{cycle}:{id}"),
        payload,
    }
}

async fn run_tool_batch(
    services: Arc<CodingServices>,
    batch: Vec<ToolCall>,
    messages: &[String],
    threshold: Option<u32>,
    turn: u64,
    cycle: u64,
) -> Result<Vec<NewFact>, Exit<ActorError>> {
    let mut nudges = Vec::with_capacity(batch.len());
    let mut runnable = Vec::new();
    for call in &batch {
        if let Some(nudge) = duplicate_nudge(messages, call, threshold) {
            nudges.push(Some(nudge));
        } else {
            nudges.push(None);
            runnable.push(call.clone());
        }
    }
    let outputs = if runnable.len() > 1 && services.tools.can_parallelize(&runnable) {
        run_parallel(&services.tools, runnable).await?
    } else {
        let mut outputs = Vec::with_capacity(runnable.len());
        for call in runnable {
            outputs.push(services.tools.run(call).await.map_err(Exit::Fail)?);
        }
        outputs
    };
    let mut ran = outputs.into_iter();
    let mut facts = Vec::with_capacity(batch.len());
    for (call, nudge) in batch.iter().zip(nudges) {
        if let Some(nudge) = nudge {
            facts.push(tool_result_fact(turn, cycle, &call.id, false, nudge, true));
            continue;
        }
        let output = ran
            .next()
            .ok_or_else(|| Exit::Die("tool batch lost a result".into()))?;
        facts.push(tool_result_fact(
            turn,
            cycle,
            &call.id,
            true,
            tool_output_text(&output),
            false,
        ));
    }
    Ok(facts)
}

async fn run_parallel(
    tools: &Arc<dyn ToolRunner>,
    calls: Vec<ToolCall>,
) -> Result<Vec<Value>, Exit<ActorError>> {
    let mut set = tokio::task::JoinSet::new();
    for (index, call) in calls.into_iter().enumerate() {
        let tools = Arc::clone(tools);
        set.spawn(async move { (index, tools.run(call).await) });
    }
    let mut indexed = Vec::new();
    while let Some(joined) = set.join_next().await {
        indexed.push(joined.map_err(|error| Exit::Die(error.to_string()))?);
    }
    indexed.sort_by_key(|(index, _)| *index);
    indexed
        .into_iter()
        .map(|(_, result)| result.map_err(Exit::Fail))
        .collect()
}

/// Infer-loop scheduler (model / tool / compact / budget deny phases).
pub fn coding_scheduler(config: HarnessConfig) -> ErasedComponent<CodingServices, CodingView> {
    let step_config = config.clone();
    let output_config = config;
    component(
        SchedulerState::new,
        move |state, fact| step(state, fact, &step_config),
        move |state| (view_of(state), transitions_of(state, &output_config)),
    )
}

/// Stock Meta Harness: `system + tools + budget + compact + infer([scheduler])`.
///
/// Equivalent to the pre-composition three-component actor for control
/// behavior; budget/compact also appear as explicit tree parts.
pub fn coding_actor(config: HarnessConfig) -> Actor<CodingServices, CodingView> {
    let spec = crate::compose::MetaHarnessSpec {
        name: "a3s-code",
        budget: config.budget(),
        compact_after_chars: config.compact_after_chars(),
        step_limit: config.step_limit(),
        model_attempts: config.model_attempts(),
        system: config.system().to_vec(),
        tools: config.tools().to_vec(),
        tool_round_cap: config.tool_round_cap(),
        parts: Vec::new(),
    };
    crate::compose::HarnessGraph::from_spec(spec, || coding_scheduler(config.clone())).into_actor()
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
