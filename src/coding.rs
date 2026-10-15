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
use tokio_util::sync::CancellationToken;

use crate::actor::{
    component, ingest_with_cancel, resume_with_cancel, Actor, ErasedComponent, PreparedTransition,
    Settlement, Transition, TransitionProgram,
};
use crate::effect::{Effect, Schedule};
use crate::error::ActorError;
use crate::exit::Exit;
use crate::fact::{Fact, NewFact};

mod tool_effects;
pub use tool_effects::{
    tool_args_digest, ToolAdmissionIdentityV1, ToolEffectObservationV1, ToolExecutionOutcome,
    ToolExecutionResultV1, ToolFileIdentityV1, ToolFileKind, ToolInvocationIdentityV1,
    ToolObservationCompleteness, ToolPathChangeV1, ToolPreparedV1, ToolRecoveryPolicy,
    MAX_TOOL_EFFECT_CANONICAL_BYTES, MAX_TOOL_EFFECT_PATHS, TOOL_PREPARED_SCHEMA,
    TOOL_RESULT_SCHEMA,
};

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

/// Model call for the infer transition.
///
/// Cancellation ownership: the harness passes the host token into
/// [`Self::complete`] and never races the call. The implementation owns
/// mid-flight cancellation — the interactive completion returns a
/// `ModelDecision::Text` carrying the interrupted-answer sentence so the turn
/// settles with its transcript — and must return promptly once the token
/// fires.
pub trait Completion: Send + Sync {
    fn complete(
        &self,
        request: CompletionRequest,
        cancel: CancellationToken,
    ) -> BoxFuture<Result<ModelDecision, ActorError>>;

    /// Provider-reported prompt tokens for the completion that just returned.
    /// `None` when the provider omitted usage. The infer transition copies this
    /// onto the `model.turn` fact so a later fold can compact from it.
    fn reported_prompt_tokens(&self) -> Option<u64> {
        None
    }
}

/// Tool execution for the tool transition.
///
/// Cancellation ownership mirrors [`Completion`]: the host token is a
/// parameter, the runner observes it, and the harness never races the call.
/// A cancelled call returns a cancelled output value that lands as the
/// `tool.result` fact.
#[derive(Debug, Clone)]
pub struct ToolPreparationRequest {
    pub identity: ToolInvocationIdentityV1,
    pub call: ToolCall,
}

#[derive(Clone)]
pub struct PreparedToolExecution {
    pub prepared: ToolPreparedV1,
    pub execute: Effect<ToolExecutionResultV1, ActorError, CodingServices>,
}

pub trait ToolRunner: Send + Sync + 'static {
    fn run(
        &self,
        call: ToolCall,
        cancel: CancellationToken,
    ) -> BoxFuture<Result<Value, ActorError>>;

    /// Resolve input and authority without executing the tool. The actor persists
    /// the returned preparation before evaluating `execute`. Legacy adapters are
    /// supported conservatively: their effect observation is explicitly incomplete.
    fn prepare(
        self: Arc<Self>,
        request: ToolPreparationRequest,
        _cancel: CancellationToken,
    ) -> BoxFuture<Result<PreparedToolExecution, ActorError>> {
        Box::pin(async move {
            let digest = tool_args_digest(&request.call.args)?;
            let prepared = ToolPreparedV1 {
                schema: TOOL_PREPARED_SCHEMA.into(),
                identity: request.identity,
                tool_name: request.call.name.clone(),
                original_args_digest: digest.clone(),
                effective_args_digest: digest,
                recovery: ToolRecoveryPolicy::NeverReplay,
                baseline: None,
            };
            let execution_prepared = prepared.clone();
            let call = request.call;
            let execute = Effect::from_async_observed(move |_services, cancel| {
                let runner = Arc::clone(&self);
                let call = call.clone();
                let prepared = execution_prepared.clone();
                async move {
                    let output = runner.run(call, cancel.clone()).await.map_err(Exit::Fail)?;
                    Ok(ToolExecutionResultV1 {
                        schema: TOOL_RESULT_SCHEMA.into(),
                        identity: prepared.identity.clone(),
                        prepared_digest: prepared.digest().map_err(Exit::Fail)?,
                        outcome: if cancel.is_cancelled() {
                            ToolExecutionOutcome::Cancelled
                        } else {
                            ToolExecutionOutcome::Succeeded
                        },
                        exit_code: None,
                        output,
                        effects: ToolEffectObservationV1::incomplete(
                            "legacy runner did not observe effects",
                        ),
                    })
                }
            });
            Ok(PreparedToolExecution { prepared, execute })
        })
    }

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

    /// Take `base`'s token compaction, which a compose recipe does not carry.
    pub(crate) fn with_token_compact_of(mut self, base: &HarnessConfig) -> Self {
        if let Some(after_tokens) = base.compact_after_tokens {
            self = self.with_token_compact(after_tokens, base.compact_keep_tokens);
        }
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
    /// A compaction landed and no model call has reported since. Pressure is
    /// measured again only after the model has seen the compacted prompt.
    compacted_awaiting_model: bool,
    assistant: Option<String>,
    pending_call: Option<ToolCall>,
    /// Calls from the same model message that have not run yet.
    queued: Vec<ToolCall>,
    pending_question: Option<PendingQuestion>,
    schema_error: Option<String>,
    prompt_tokens: Option<u64>,
    /// Background-arrival lines folded while the protocol was mid-step.
    /// Flush points move them into `messages` (see [`flush_background`]).
    background_pending: Vec<String>,
    /// Set once [`BACKGROUND_PENDING_CAP`] overflowed; the flush renders one
    /// truncation marker instead of the dropped lines. Replay-deterministic.
    background_truncated: bool,
    preparations: Vec<ToolPreparedV1>,
    completed_invocations: Vec<ToolInvocationIdentityV1>,
    legacy_pending: bool,
}

/// Rendered when background arrivals exceeded [`BACKGROUND_PENDING_CAP`]
/// while the protocol was mid-step. The producer-side stream is separately
/// capped per task; this marker bounds the fold-side buffer.
const BACKGROUND_TRUNCATION_LINE: &str =
    "[background] …pending arrivals truncated (background pending cap reached)";

/// Buffered background arrivals are bounded: at most this many lines wait for
/// the next flush point.
const BACKGROUND_PENDING_CAP: usize = 64;

impl SchedulerState {
    fn new() -> Self {
        Self {
            turn: 0,
            cycle: 0,
            phase: CodingPhase::Idle,
            messages: Vec::new(),
            summary: String::new(),
            tool_runs: 0,
            compacted_awaiting_model: false,
            assistant: None,
            pending_call: None,
            queued: Vec::new(),
            pending_question: None,
            schema_error: None,
            prompt_tokens: None,
            background_pending: Vec::new(),
            background_truncated: false,
            preparations: Vec::new(),
            completed_invocations: Vec::new(),
            legacy_pending: false,
        }
    }

    fn size(&self) -> usize {
        self.summary.len()
            + self.messages.iter().map(String::len).sum::<usize>()
            + self
                .background_pending
                .iter()
                .map(String::len)
                .sum::<usize>()
    }
}

/// Move buffered background arrivals into `messages`.
///
/// Call only at protocol-safe points: when `messages` ends at a completed
/// step (a finished tool batch, an answered question, right before a new
/// user line) — never between an `assistant\ntool` line and its results,
/// and never before the `model.turn` fact of a model call that already ran
/// without them. Every call site satisfies both by construction; the fold
/// order of the log keeps replay deterministic.
fn flush_background(state: SchedulerState) -> SchedulerState {
    if state.background_pending.is_empty() && !state.background_truncated {
        return state;
    }
    let mut messages = state.messages.clone();
    if state.background_truncated {
        messages.push(BACKGROUND_TRUNCATION_LINE.to_string());
    }
    messages.extend(state.background_pending.iter().cloned());
    SchedulerState {
        messages,
        background_pending: Vec::new(),
        background_truncated: false,
        ..state
    }
}

/// Fold one `background.output` fact. While the protocol is mid-step — a
/// model call in flight, a tool batch or confirmation pending, a compaction
/// rewriting the transcript — the rendered line buffers; otherwise it lands
/// directly after the last completed step.
fn on_background_output(state: SchedulerState, fact: &Fact) -> SchedulerState {
    let payload = match serde_json::from_value::<BackgroundOutputPayload>(fact.payload.clone()) {
        Ok(payload) => payload,
        Err(error) => {
            return SchedulerState {
                phase: CodingPhase::Done,
                schema_error: Some(error.to_string()),
                ..state
            }
        }
    };
    let line = payload.render();
    let mid_step = state.pending_call.is_some()
        || !state.queued.is_empty()
        || matches!(
            state.phase,
            CodingPhase::Infer
                | CodingPhase::Tool
                | CodingPhase::Confirm
                | CodingPhase::Deny
                | CodingPhase::Compact
        );
    if !mid_step {
        let mut messages = state.messages.clone();
        messages.push(line);
        return SchedulerState { messages, ..state };
    }
    if state.background_pending.len() >= BACKGROUND_PENDING_CAP {
        return SchedulerState {
            background_truncated: true,
            ..state
        };
    }
    let mut background_pending = state.background_pending.clone();
    background_pending.push(line);
    SchedulerState {
        background_pending,
        ..state
    }
}

/// One emitted line of a Run-owned background task, appended by the task's
/// reader through the normal ingest path (`background.output` facts on the
/// session's one fact log).
#[derive(Deserialize)]
struct BackgroundOutputPayload {
    name: String,
    line: String,
    /// `truncated` marks the producer-side stream cap; `exited` carries the
    /// final disposition line. Absent for a plain output line.
    #[serde(default)]
    event: Option<String>,
    /// `stderr` when the line came from the task's standard error.
    #[serde(default)]
    stream: Option<String>,
}

impl BackgroundOutputPayload {
    fn render(&self) -> String {
        match self.event.as_deref() {
            Some("truncated") => format!(
                "[background:{}] …output truncated (recorded-line cap reached; the task keeps running but its output is no longer recorded)",
                self.name
            ),
            Some("exited") => format!("[background:{}] exited: {}", self.name, self.line),
            _ => {
                if self.stream.as_deref() == Some("stderr") {
                    format!("[background:{}] (stderr) {}", self.name, self.line)
                } else {
                    format!("[background:{}] {}", self.name, self.line)
                }
            }
        }
    }
}

fn after_input(state: SchedulerState, config: &HarnessConfig) -> SchedulerState {
    let (prefix, _) = split_compact_tail(&state.messages, config.compact_keep_tokens);
    let token_pressure = config
        .compact_after_tokens
        .is_some_and(|limit| state.prompt_tokens.is_some_and(|used| used >= limit));
    let char_pressure =
        config.compact_after_tokens.is_none() && state.size() >= config.compact_after_chars;
    let needs_compact =
        !prefix.is_empty() && (token_pressure || char_pressure) && !state.compacted_awaiting_model;
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
    state.compacted_awaiting_model = false;
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
        ModelDecision::Text { text } => {
            // The immutable model fact is also conversation history. Keep it
            // in the same fold so later inference and compaction see the reply.
            state.messages.push(format!("assistant\n{text}"));
            SchedulerState {
                phase: CodingPhase::Done,
                assistant: Some(text),
                pending_call: None,
                ..state
            }
        }
        ModelDecision::Question {
            question_id,
            question,
            allow_free_text,
            options,
        } => {
            state.messages.push(format!("assistant\n{question}"));
            SchedulerState {
                phase: CodingPhase::Question,
                pending_question: Some(PendingQuestion {
                    question_id,
                    question,
                    allow_free_text,
                    options,
                }),
                pending_call: None,
                ..state
            }
        }
        ModelDecision::Tool { call, also } => {
            let legacy_pending = match payload.get("tool_effect_protocol") {
                None => true,
                Some(value) if value.as_u64() == Some(1) => false,
                Some(_) => {
                    state.schema_error =
                        Some("tool_effect_schema_unsupported: model tool protocol".into());
                    state.phase = CodingPhase::Done;
                    return state;
                }
            };
            let mut calls = Vec::with_capacity(1 + also.len());
            calls.push(call);
            calls.extend(also);
            let mut ids = std::collections::BTreeSet::new();
            if calls.iter().any(|call| !ids.insert(call.id.as_str())) {
                state.schema_error = Some(
                    "tool_effect_identity_mismatch: duplicate tool call id in model batch".into(),
                );
                state.phase = CodingPhase::Done;
                return state;
            }
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
                legacy_pending,
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
/// result, plus the preceding line when that line is its own record.
/// A positive budget walks backward by scheduler regions. A run of
/// `assistant\ntool ` records followed by one or more results is one region,
/// so the tail cannot keep the last record glued to another call's result.
/// Records with no following result stay single lines.
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
    const RECORD_PREFIX: &str = "assistant\ntool ";
    let mut units = Vec::new();
    let mut index = 0;
    while index < messages.len() {
        if !messages[index].starts_with(RECORD_PREFIX) {
            units.push((index, index + 1));
            index += 1;
            continue;
        }
        let start = index;
        while index < messages.len() && messages[index].starts_with(RECORD_PREFIX) {
            index += 1;
        }
        let records = index - start;
        let mut results = 0;
        while index + results < messages.len() && is_tool_result_line(&messages[index + results]) {
            results += 1;
        }
        if results > 0 {
            units.push((start, index + results));
            index += results;
        } else {
            index = start;
            while index < start + records {
                units.push((index, index + 1));
                index += 1;
            }
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
    #[serde(default, rename = "toolCallId")]
    tool_call_id: Option<String>,
    #[serde(default)]
    receipt: Option<ToolExecutionResultV1>,
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
            Ok(payload) => {
                // Arrivals that belonged to the previous turn land before the
                // new human line, which is where the next model call reads them.
                let state = flush_background(state);
                after_input(
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
                )
            }
            Err(error) => {
                state.phase = CodingPhase::Done;
                state.schema_error = Some(error.to_string());
                state
            }
        },
        "compaction.done" => match serde_json::from_value::<SummaryPayload>(fact.payload.clone()) {
            Ok(payload) => SchedulerState {
                summary: payload.summary,
                compacted_awaiting_model: true,
                // The count before compaction no longer describes the prompt.
                prompt_tokens: None,
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
        "tool.prepared" => {
            let parsed = serde_json::from_value::<ToolPreparedV1>(
                fact.payload.get("prepared").cloned().unwrap_or(Value::Null),
            )
            .map_err(|error| ActorError::Schema(error.to_string()))
            .and_then(|prepared| {
                prepared.validate()?;
                let Some(call) = state
                    .pending_call
                    .as_ref()
                    .filter(|call| call.id == prepared.identity.tool_call_id)
                    .or_else(|| {
                        state
                            .queued
                            .iter()
                            .find(|call| call.id == prepared.identity.tool_call_id)
                    })
                else {
                    return Err(ActorError::Schema(
                        "tool_effect_identity_mismatch: preparation has no pending call".into(),
                    ));
                };
                let batch = ready_tool_batch(
                    state.pending_call.as_ref().ok_or_else(|| {
                        ActorError::Schema("tool_effect_identity_mismatch: no selected call".into())
                    })?,
                    &state.queued,
                );
                let index = usize::try_from(prepared.identity.batch_index).map_err(|_| {
                    ActorError::Schema("tool_effect_identity_mismatch: batch index overflow".into())
                })?;
                if batch
                    .get(index)
                    .is_none_or(|expected| expected.id != call.id)
                {
                    return Err(ActorError::Schema(
                        "tool_effect_identity_mismatch: wrong batch position".into(),
                    ));
                }
                // Every preparation in a batch binds its original batch cycle.
                if prepared.identity.turn != state.turn
                    || prepared.identity.cycle != state.cycle
                    || call.name != prepared.tool_name
                    || tool_args_digest(&call.args)? != prepared.original_args_digest
                {
                    return Err(ActorError::Schema(
                        "tool_effect_identity_mismatch: preparation differs from selected call"
                            .into(),
                    ));
                }
                if state
                    .preparations
                    .iter()
                    .any(|earlier| earlier.identity.matches_scheduler(&prepared.identity))
                    || state
                        .completed_invocations
                        .iter()
                        .any(|earlier| earlier.matches_scheduler(&prepared.identity))
                {
                    return Err(ActorError::Schema(
                        "tool_effect_identity_mismatch: duplicate preparation".into(),
                    ));
                }
                Ok(prepared)
            });
            match parsed {
                Ok(prepared) => state.preparations.push(prepared),
                Err(error) => {
                    state.schema_error = Some(error.to_string());
                    state.phase = CodingPhase::Done;
                }
            }
            state
        }
        "tool.result" if fact.payload.get("receipt") == Some(&Value::Null) => {
            state.schema_error = Some("tool_effect_schema_unsupported: null receipt".into());
            state.phase = CodingPhase::Done;
            state
        }
        "tool.result" => match serde_json::from_value::<ToolResultPayload>(fact.payload.clone()) {
            Ok(payload) => {
                if let Err(error) = validate_folded_result(&state, &payload) {
                    state.schema_error = Some(error.to_string());
                    state.phase = CodingPhase::Done;
                    return state;
                }
                if let Some(receipt) = &payload.receipt {
                    state
                        .preparations
                        .retain(|prepared| prepared.identity != receipt.identity);
                    state.completed_invocations.push(receipt.identity.clone());
                }
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
                    // The batch is complete: arrivals buffered while the tools
                    // ran are the next thing the model should read.
                    after_input(flush_background(folded), config)
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
            Ok(payload) => {
                // Same placement rule as `user.message`: buffered arrivals
                // precede the answer text the fold is about to append.
                let state = flush_background(state);
                after_input(
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
                )
            }
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
        "background.output" => on_background_output(state, fact),
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
    if state.schema_error.is_some() {
        return Vec::new();
    }
    match state.phase {
        CodingPhase::Compact => {
            let (prefix, kept) = split_compact_tail(&state.messages, config.compact_keep_tokens);
            let mut to_summarize = prefix;
            if !state.summary.is_empty() {
                to_summarize.insert(0, state.summary.clone());
            }
            let (turn, cycle) = (state.turn, state.cycle);
            vec![Transition {
                key: format!("compact:{turn}:{cycle}"),
                run: Effect::from_async_observed(move |services: Arc<CodingServices>, cancel| {
                    let to_summarize = to_summarize.clone();
                    let kept = kept.clone();
                    async move {
                        tokio::select! {
                            biased;
                            _ = cancel.cancelled() => Err(Exit::Interrupt),
                            summary = services.compactor.compact(&to_summarize) => {
                                let summary = summary?;
                                Ok(vec![NewFact {
                                    kind: "compaction.done".into(),
                                    key: format!("compaction:{turn}:{cycle}"),
                                    payload: json!({ "summary": summary, "kept": kept }),
                                }])
                            }
                        }
                    }
                })
                .into(),
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
                // Mid-flight cancellation belongs to the completion: an
                // observing completion lands its own interrupted decision and
                // session transcript, and racing it here with a select would
                // discard that work and leave the turn without its interrupt
                // answer. Only the not-yet-started turn is refused here.
                run: Effect::from_async_observed(move |services: Arc<CodingServices>, cancel| {
                    let request = request.clone();
                    async move {
                        if cancel.is_cancelled() {
                            return interrupted_model_turn(turn, cycle);
                        }
                        let decided = services.completion.complete(request, cancel.clone()).await;
                        let decided = match decided {
                            Ok(decided) => decided,
                            Err(ActorError::Defect(message)) => {
                                return Err(Exit::Die(message));
                            }
                            Err(error) => return Err(Exit::Fail(error)),
                        };
                        let mut payload = serde_json::to_value(&decided)
                            .map_err(|error| Exit::Die(error.to_string()))?;
                        // A protocol-one tool decision guarantees that no tool
                        // can start without the durable preparation boundary.
                        payload["tool_effect_protocol"] = json!(1);
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
                })
                .into(),
            }]
        }
        CodingPhase::Tool => {
            let Some(call) = state.pending_call.clone() else {
                return Vec::new();
            };
            let batch = ready_tool_batch(&call, &state.queued);
            let turn = state.turn;
            let cycle = state.cycle;
            // The same durable prepared state is ambiguous after process death.
            // Only the original selected continuation executes; replay records
            // Unknown, preserving accountability without repeating an operation.
            let unresolved: Vec<_> = state
                .preparations
                .iter()
                .filter(|prepared| {
                    prepared.identity.turn == turn
                        && batch
                            .iter()
                            .any(|call| call.id == prepared.identity.tool_call_id)
                })
                .cloned()
                .collect();
            if !unresolved.is_empty() {
                let key = format!("tool-reconcile:{turn}:{cycle}:{}", call.id);
                return vec![Transition {
                    key,
                    run: Effect::from_async_observed(move |_services, _cancel| {
                        let unresolved = unresolved.clone();
                        async move {
                            unresolved
                                .iter()
                                .map(|prepared| {
                                    let result = ToolExecutionResultV1::unknown(
                                        prepared,
                                        "prepared invocation has no committed terminal result",
                                    )
                                    .map_err(Exit::Fail)?;
                                    receipt_fact(prepared, result).map_err(Exit::Fail)
                                })
                                .collect()
                        }
                    })
                    .into(),
                }];
            }
            if state.legacy_pending {
                return vec![Transition {
                    key: format!("tool-legacy-unresolved:{turn}:{cycle}:{}", call.id),
                    run: Effect::succeed(batch.iter().map(|call| {
                        let mut fact = tool_result_fact(turn, cycle, &call.id, false,
                            "tool_effect_unresolved: legacy invocation has no authoritative execution evidence".into(), true);
                        fact.payload["unresolvedLegacy"] = json!(true);
                        fact
                    }).collect()).into(),
                }];
            }
            let messages = state.messages.clone();
            let threshold = config.duplicate_threshold;
            let key = batch
                .iter()
                .map(|call| call.id.as_str())
                .collect::<Vec<_>>()
                .join("+");
            vec![Transition {
                key: format!("tool:{turn}:{cycle}:{key}"),
                run: TransitionProgram::Prepared(Effect::from_async_observed(
                    move |services: Arc<CodingServices>, cancel| {
                        let batch = batch.clone();
                        let messages = messages.clone();
                        async move {
                            prepare_tool_batch(
                                services, batch, &messages, threshold, turn, cycle, cancel,
                            )
                            .await
                        }
                    },
                )),
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
                }])
                .into(),
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

fn interrupted_model_turn(turn: u64, cycle: u64) -> Result<Vec<NewFact>, Exit<ActorError>> {
    let decided = ModelDecision::Text {
        text: "(Response interrupted by the user.)".into(),
    };
    let payload = serde_json::to_value(&decided).map_err(|error| Exit::Die(error.to_string()))?;
    Ok(vec![NewFact {
        kind: "model.turn".into(),
        key: format!("model:{turn}:{cycle}"),
        payload,
    }])
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

fn receipt_fact(
    prepared: &ToolPreparedV1,
    result: ToolExecutionResultV1,
) -> Result<NewFact, ActorError> {
    result.ensure_prepared(prepared)?;
    let ok = result.outcome == ToolExecutionOutcome::Succeeded;
    let mut fact = tool_result_fact(
        prepared.identity.turn,
        prepared.identity.cycle,
        &prepared.identity.tool_call_id,
        ok,
        tool_output_text(&result.output),
        !ok,
    );
    fact.payload["receipt"] =
        serde_json::to_value(result).map_err(|error| ActorError::Schema(error.to_string()))?;
    Ok(fact)
}

fn validate_folded_result(
    state: &SchedulerState,
    payload: &ToolResultPayload,
) -> Result<(), ActorError> {
    if let Some(receipt) = &payload.receipt {
        receipt.validate()?;
    }
    let Some(call) = &state.pending_call else {
        return Err(ActorError::Schema(
            "tool_effect_identity_mismatch: result has no pending call".into(),
        ));
    };
    if payload
        .tool_call_id
        .as_ref()
        .is_some_and(|id| id != &call.id)
    {
        return Err(ActorError::Schema(
            "tool_effect_identity_mismatch: wrong pending call".into(),
        ));
    }
    if let Some(receipt) = &payload.receipt {
        let prepared = state
            .preparations
            .iter()
            .find(|prepared| prepared.identity == receipt.identity)
            .ok_or_else(|| {
                ActorError::Schema(
                    "tool_effect_identity_mismatch: result has no preparation".into(),
                )
            })?;
        receipt.ensure_prepared(prepared)?;
        if receipt.identity.turn != state.turn
            || receipt.identity.tool_call_id != call.id
            || payload.ok != (receipt.outcome == ToolExecutionOutcome::Succeeded)
            || payload.output != tool_output_text(&receipt.output)
        {
            return Err(ActorError::Schema(
                "tool_effect_identity_mismatch: terminal projection differs".into(),
            ));
        }
    } else if state.preparations.iter().any(|prepared| {
        prepared.identity.turn == state.turn && prepared.identity.tool_call_id == call.id
    }) {
        return Err(ActorError::Schema(
            "tool_effect_identity_mismatch: prepared call needs a typed result".into(),
        ));
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
async fn prepare_tool_batch(
    services: Arc<CodingServices>,
    batch: Vec<ToolCall>,
    messages: &[String],
    threshold: Option<u32>,
    turn: u64,
    cycle: u64,
    cancel: CancellationToken,
) -> Result<PreparedTransition<CodingServices>, Exit<ActorError>> {
    tokio::task::yield_now().await;
    let mut prepared_batch = Vec::with_capacity(batch.len());
    let mut accepted: Vec<&ToolCall> = Vec::new();
    for (index, call) in batch.iter().enumerate() {
        let identity = ToolInvocationIdentityV1 {
            turn,
            cycle,
            tool_call_id: call.id.clone(),
            batch_index: u32::try_from(index)
                .map_err(|_| Exit::Fail(ActorError::Schema("tool batch index overflow".into())))?,
            admission: None,
        };
        let nudge = if threshold.is_some()
            && accepted
                .iter()
                .any(|earlier| earlier.name == call.name && earlier.args == call.args)
        {
            Some(format!("Tool '{}' has been called 2 times with identical arguments. Change your approach instead of repeating this call.", call.name))
        } else {
            duplicate_nudge(messages, call, threshold)
        };
        accepted.push(call);
        let mut execution = Arc::clone(&services.tools)
            .prepare(
                ToolPreparationRequest {
                    identity: identity.clone(),
                    call: call.clone(),
                },
                cancel.clone(),
            )
            .await
            .map_err(Exit::Fail)?;
        execution.prepared.validate().map_err(Exit::Fail)?;
        if !execution.prepared.identity.matches_scheduler(&identity)
            || execution.prepared.tool_name != call.name
            || execution.prepared.original_args_digest
                != tool_args_digest(&call.args).map_err(Exit::Fail)?
        {
            return Err(Exit::Fail(ActorError::Schema(
                "tool_effect_identity_mismatch: runner changed selected intent".into(),
            )));
        }
        if let Some(nudge) = nudge {
            // A convergence refusal still binds the same admitted invocation;
            // only its execution continuation is replaced by a no-effect result.
            let result = ToolExecutionResultV1 {
                schema: TOOL_RESULT_SCHEMA.into(),
                identity: execution.prepared.identity.clone(),
                prepared_digest: execution.prepared.digest().map_err(Exit::Fail)?,
                outcome: ToolExecutionOutcome::Denied,
                exit_code: None,
                output: Value::String(nudge),
                effects: ToolEffectObservationV1::complete(Vec::new()),
            };
            execution.execute = Effect::succeed(result);
        }
        prepared_batch.push(execution);
    }
    let facts = prepared_batch
        .iter()
        .map(|execution| {
            Ok(NewFact {
                kind: "tool.prepared".into(),
                key: format!("tool-prepared:{}", execution.prepared.identity.key()?),
                payload: json!({ "prepared": execution.prepared }),
            })
        })
        .collect::<Result<Vec<_>, ActorError>>()
        .map_err(Exit::Fail)?;
    let parallel = batch.len() > 1 && services.tools.can_parallelize(&batch);
    let prepared_batch = Arc::new(prepared_batch);
    let run = Effect::from_async_observed(move |services: Arc<CodingServices>, cancel| {
        let prepared_batch = Arc::clone(&prepared_batch);
        async move {
            let results = if parallel {
                run_prepared_parallel(&prepared_batch, services, cancel).await
            } else {
                let mut results = Vec::with_capacity(prepared_batch.len());
                for execution in prepared_batch.iter() {
                    results.push(
                        run_prepared_one(execution, Arc::clone(&services), cancel.clone()).await,
                    );
                }
                results
            };
            prepared_batch
                .iter()
                .zip(results)
                .map(|(execution, result)| {
                    let result = match result {
                        Ok(result) => match result.ensure_prepared(&execution.prepared) {
                            Ok(()) => result,
                            Err(error) => ToolExecutionResultV1::unknown(
                                &execution.prepared,
                                format!("runner receipt was refused: {error}"),
                            )
                            .map_err(Exit::Fail)?,
                        },
                        Err(error) => ToolExecutionResultV1::unknown(
                            &execution.prepared,
                            format!("execution did not return an authoritative result: {error:?}"),
                        )
                        .map_err(Exit::Fail)?,
                    };
                    receipt_fact(&execution.prepared, result).map_err(Exit::Fail)
                })
                .collect()
        }
    });
    Ok(PreparedTransition { facts, run })
}

async fn run_prepared_one(
    execution: &PreparedToolExecution,
    services: Arc<CodingServices>,
    cancel: CancellationToken,
) -> Result<ToolExecutionResultV1, Exit<ActorError>> {
    if cancel.is_cancelled() {
        return Ok(ToolExecutionResultV1 {
            schema: TOOL_RESULT_SCHEMA.into(),
            identity: execution.prepared.identity.clone(),
            prepared_digest: execution.prepared.digest().map_err(Exit::Fail)?,
            outcome: ToolExecutionOutcome::Cancelled,
            exit_code: None,
            output: Value::String("cancelled".into()),
            effects: ToolEffectObservationV1::complete(Vec::new()),
        });
    }
    // Once execution started, its returned host evidence survives a late cancel.
    execution
        .execute
        .clone()
        .run_with_cancel(services, cancel)
        .await
        .0
}

async fn run_prepared_parallel(
    executions: &[PreparedToolExecution],
    services: Arc<CodingServices>,
    cancel: CancellationToken,
) -> Vec<Result<ToolExecutionResultV1, Exit<ActorError>>> {
    let mut set = tokio::task::JoinSet::new();
    let mut ids = std::collections::HashMap::new();
    for (index, execution) in executions.iter().cloned().enumerate() {
        let services = Arc::clone(&services);
        let cancel = cancel.clone();
        let task = set.spawn(async move { run_prepared_one(&execution, services, cancel).await });
        ids.insert(task.id(), index);
    }
    let mut results: Vec<_> = executions.iter().map(|_| None).collect();
    while let Some(joined) = set.join_next_with_id().await {
        match joined {
            Ok((id, result)) => {
                if let Some(index) = ids.remove(&id) {
                    results[index] = Some(result);
                }
            }
            Err(error) => {
                if let Some(index) = ids.remove(&error.id()) {
                    results[index] = Some(Err(Exit::Die(error.to_string())));
                }
            }
        }
    }
    results
        .into_iter()
        .map(|result| result.unwrap_or_else(|| Err(Exit::Die("tool batch lost a result".into()))))
        .collect()
}

// The existing batch overlap unit tests exercise the executor directly.
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
async fn run_tool_batch(
    services: Arc<CodingServices>,
    batch: Vec<ToolCall>,
    messages: &[String],
    threshold: Option<u32>,
    turn: u64,
    cycle: u64,
    cancel: CancellationToken,
) -> Result<Vec<NewFact>, Exit<ActorError>> {
    let prepared = prepare_tool_batch(
        Arc::clone(&services),
        batch,
        messages,
        threshold,
        turn,
        cycle,
        cancel.clone(),
    )
    .await?;
    prepared.run.run_with_cancel(services, cancel).await.0
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

/// One background-task output line as a fact on the session's one log.
///
/// `task_seq` counts emitted facts within the task and rides the fact KEY
/// (`bg:{name}:{task_seq}`), which is what makes keys unique per task
/// without coordinating with the log's own seq. `event` is
/// `Some("truncated" | "exited")` for the stream-cap marker and the final
/// disposition line; `stream` is `Some("stderr")` for standard-error lines.
pub fn background_output_fact(
    name: &str,
    line: &str,
    task_seq: u64,
    event: Option<&str>,
    stream: Option<&str>,
) -> NewFact {
    let mut payload = json!({ "name": name, "line": line });
    if let Some(event) = event {
        payload["event"] = json!(event);
    }
    if let Some(stream) = stream {
        payload["stream"] = json!(stream);
    }
    NewFact {
        kind: "background.output".into(),
        key: format!("bg:{name}:{task_seq}"),
        payload,
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
    ingest_coding_with_cancel(
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

pub async fn ingest_coding_with_cancel(
    actor: &Actor<CodingServices, CodingView>,
    log: &dyn crate::log::LogStore,
    services: Arc<CodingServices>,
    thread_id: &str,
    fact: NewFact,
    limit: u32,
    cancel: &CancellationToken,
) -> Result<Settlement<CodingView>, Exit<ActorError>> {
    ingest_with_cancel(actor, log, services, thread_id, fact, limit, cancel).await
}

pub async fn resume_coding(
    actor: &Actor<CodingServices, CodingView>,
    log: &dyn crate::log::LogStore,
    services: Arc<CodingServices>,
    thread_id: &str,
    limit: u32,
) -> Result<Settlement<CodingView>, Exit<ActorError>> {
    resume_coding_with_cancel(
        actor,
        log,
        services,
        thread_id,
        limit,
        &CancellationToken::new(),
    )
    .await
}

pub async fn resume_coding_with_cancel(
    actor: &Actor<CodingServices, CodingView>,
    log: &dyn crate::log::LogStore,
    services: Arc<CodingServices>,
    thread_id: &str,
    limit: u32,
    cancel: &CancellationToken,
) -> Result<Settlement<CodingView>, Exit<ActorError>> {
    resume_with_cancel(actor, log, services, thread_id, limit, cancel).await
}

#[cfg(test)]
mod batch_overlap_tests {
    //! WC-MEAS2: deterministic overlap oracles for the tool-batch executor.
    //! A batch the adapter marks parallel-safe must actually overlap — a
    //! serialization regression deadlocks on the barrier and fails the
    //! guarded timeout instead of passing silently (matrix #1's silent
    //! failure mode, pinned at the executor this time).

    use super::*;
    use serde_json::json;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;
    use std::time::{Duration, Instant};
    use tokio::sync::Barrier;

    struct NeverModel;

    impl Completion for NeverModel {
        fn complete(
            &self,
            _request: CompletionRequest,
            _cancel: CancellationToken,
        ) -> BoxFuture<Result<ModelDecision, ActorError>> {
            Box::pin(async {
                Ok(ModelDecision::Text {
                    text: "unused".into(),
                })
            })
        }
    }

    struct NeverCompact;

    impl Compactor for NeverCompact {
        fn compact(&self, _messages: &[String]) -> BoxFuture<Result<String, ActorError>> {
            Box::pin(async { Ok(String::new()) })
        }
    }

    struct BarrierInner {
        barrier: Barrier,
        entered: AtomicUsize,
    }

    struct BarrierTools {
        inner: Arc<BarrierInner>,
        allow_parallel: bool,
    }

    impl ToolRunner for BarrierTools {
        fn run(
            &self,
            call: ToolCall,
            _cancel: CancellationToken,
        ) -> BoxFuture<Result<Value, ActorError>> {
            let inner = Arc::clone(&self.inner);
            Box::pin(async move {
                inner.entered.fetch_add(1, Ordering::SeqCst);
                // Releases only when every party is in flight at once.
                let _ = inner.barrier.wait().await;
                Ok(json!(format!("passed-{}", call.id)))
            })
        }

        fn can_parallelize(&self, _calls: &[ToolCall]) -> bool {
            self.allow_parallel
        }
    }

    struct SleepTools {
        per_call: Duration,
        allow_parallel: bool,
    }

    impl ToolRunner for SleepTools {
        fn run(
            &self,
            call: ToolCall,
            _cancel: CancellationToken,
        ) -> BoxFuture<Result<Value, ActorError>> {
            let per_call = self.per_call;
            Box::pin(async move {
                tokio::time::sleep(per_call).await;
                Ok(json!(format!("slept-{}", call.id)))
            })
        }

        fn can_parallelize(&self, _calls: &[ToolCall]) -> bool {
            self.allow_parallel
        }
    }

    fn services(tools: Arc<dyn ToolRunner>) -> Arc<CodingServices> {
        Arc::new(CodingServices {
            completion: Arc::new(NeverModel),
            tools,
            compactor: Arc::new(NeverCompact),
        })
    }

    fn read_batch(count: usize) -> Vec<ToolCall> {
        (0..count)
            .map(|index| ToolCall {
                id: format!("r{index}"),
                name: "read".into(),
                args: json!({ "file_path": format!("f{index}.rs") }),
                needs_confirmation: false,
                text: None,
                reasoning: None,
            })
            .collect()
    }

    #[tokio::test]
    async fn parallel_batch_overlaps_or_fails_closed() {
        let parties = 4;
        let tools = Arc::new(BarrierTools {
            inner: Arc::new(BarrierInner {
                barrier: Barrier::new(parties),
                entered: AtomicUsize::new(0),
            }),
            allow_parallel: true,
        });
        let entered = Arc::clone(&tools.inner);
        let facts = tokio::time::timeout(
            Duration::from_secs(10),
            run_tool_batch(
                services(tools),
                read_batch(parties),
                &[],
                None,
                1,
                1,
                CancellationToken::new(),
            ),
        )
        .await
        .expect("SERIALIZED: the batch never overlapped — the executor ran calls one after another while the adapter marked them parallel-safe")
        .expect("batch executor failed");
        assert_eq!(facts.len(), parties);
        for (fact, index) in facts.iter().zip(0..) {
            assert_eq!(fact.kind, "tool.result");
            assert_eq!(fact.key, format!("tool-result:1:1:r{index}"));
            assert_eq!(fact.payload["output"], json!(format!("passed-r{index}")));
            assert_eq!(fact.payload["ok"], json!(true));
        }
        assert_eq!(
            entered.entered.load(Ordering::SeqCst),
            parties,
            "every party must have entered the barrier"
        );
    }

    #[tokio::test]
    async fn duplicate_call_twins_stay_parked_even_in_parallel_batches() {
        // The same call emitted twice must not run twice; the twin gets the
        // nudge fact while the two distinct calls still overlap. Only the
        // distinct calls are runnable, so the barrier counts those.
        let parties = 2;
        let tools = Arc::new(BarrierTools {
            inner: Arc::new(BarrierInner {
                barrier: Barrier::new(parties),
                entered: AtomicUsize::new(0),
            }),
            allow_parallel: true,
        });
        let mut batch = read_batch(parties);
        batch.push(batch[0].clone());
        let facts = tokio::time::timeout(
            Duration::from_secs(10),
            run_tool_batch(
                services(tools),
                batch,
                &[],
                Some(1),
                1,
                1,
                CancellationToken::new(),
            ),
        )
        .await
        .expect("SERIALIZED: distinct calls must overlap")
        .expect("batch executor failed");
        assert_eq!(facts.len(), 3);
        assert_eq!(facts[0].payload["ok"], json!(true));
        assert_eq!(facts[1].payload["ok"], json!(true));
        assert_eq!(facts[2].payload["ok"], json!(false));
        assert_eq!(
            facts[2].payload["countRun"],
            json!(true),
            "the parked twin still counts toward the budget"
        );
    }

    // Measurement, not assertion: the oracle above pins overlap
    // deterministically; this records the fan-out wall clock the evidence
    // ledger quotes (batch=8 disjoint reads shape, sleep=60ms stand-in).
    #[tokio::test]
    async fn parallel_batch_fanout_wall_clock_is_recorded() {
        let per_call = Duration::from_millis(60);
        let parallel_tools = Arc::new(SleepTools {
            per_call,
            allow_parallel: true,
        });
        let serial_tools = Arc::new(SleepTools {
            per_call,
            allow_parallel: false,
        });
        let batch = read_batch(8);

        let started = Instant::now();
        run_tool_batch(
            services(parallel_tools),
            batch.clone(),
            &[],
            None,
            1,
            1,
            CancellationToken::new(),
        )
        .await
        .expect("parallel arm failed");
        let parallel_ms = started.elapsed().as_millis();

        let started = Instant::now();
        run_tool_batch(
            services(serial_tools),
            batch,
            &[],
            None,
            1,
            1,
            CancellationToken::new(),
        )
        .await
        .expect("serial arm failed");
        let serial_ms = started.elapsed().as_millis();

        println!(
            "WCMEAS2 fan-out: batch=8 per_call=60ms parallel={parallel_ms}ms serial={serial_ms}ms"
        );
    }
}

#[cfg(test)]
mod background_fold_tests {
    //! WC-BG1 fold semantics: background arrivals are facts on the one log,
    //! and the fold places them where the next model call reads them — never
    //! between an `assistant\ntool` line and its results, never before the
    //! `model.turn` of a call that already ran without them.

    use super::*;

    fn config() -> HarnessConfig {
        HarnessConfig::new(16, 1_000_000, 8, 1, Vec::new(), Vec::new()).unwrap()
    }

    fn fact(seq: u64, new_fact: NewFact) -> Fact {
        Fact {
            seq,
            kind: new_fact.kind,
            key: new_fact.key,
            cause: None,
            payload: new_fact.payload,
        }
    }

    fn bg(seq: u64, name: &str, line: &str, task_seq: u64) -> Fact {
        fact(
            seq,
            background_output_fact(name, line, task_seq, None, None),
        )
    }

    fn tool_call_fact(seq: u64, name: &str, id: &str) -> Fact {
        fact(
            seq,
            NewFact {
                kind: "model.turn".into(),
                key: "model:1:0".into(),
                payload: serde_json::to_value(ModelDecision::Tool {
                    call: ToolCall {
                        id: id.into(),
                        needs_confirmation: false,
                        name: name.into(),
                        args: serde_json::json!({}),
                        text: None,
                        reasoning: None,
                    },
                    also: Vec::new(),
                })
                .unwrap(),
            },
        )
    }

    fn tool_result_fact(seq: u64) -> Fact {
        fact(
            seq,
            NewFact {
                kind: "tool.result".into(),
                key: "tr".into(),
                payload: serde_json::json!({ "ok": true, "output": "ok" }),
            },
        )
    }

    fn fold(facts: Vec<Fact>) -> SchedulerState {
        let mut state = SchedulerState::new();
        for item in facts {
            state = step(state, &item, &config());
        }
        state
    }

    #[test]
    fn arrival_during_tool_batch_lands_after_its_result() {
        let state = fold(vec![
            fact(1, message_fact("m-1", "start the watch")),
            tool_call_fact(2, "background", "call-1"),
            bg(3, "watch", "BUILD DONE", 1),
            tool_result_fact(4),
        ]);
        let tool_at = state
            .messages
            .iter()
            .position(|line| line.starts_with("assistant\ntool background"))
            .expect("assistant tool line present");
        let result_at = state
            .messages
            .iter()
            .position(|line| line.starts_with("tool\n"))
            .expect("tool result present");
        let bg_at = state
            .messages
            .iter()
            .position(|line| line == "[background:watch] BUILD DONE")
            .expect("arrival folded");
        // Protocol adjacency: the tool result directly follows its call, and
        // the arrival reads after the completed pair.
        assert_eq!(result_at, tool_at + 1, "tool result must be adjacent");
        assert!(bg_at > result_at, "arrival must follow the completed pair");
        assert!(state.background_pending.is_empty());
    }

    #[test]
    fn arrival_during_model_call_waits_for_the_next_completed_step() {
        // The arrival fact has a lower seq than the model.turn of the call
        // that was in flight, so it must not appear before that turn's line.
        let state = fold(vec![
            fact(1, message_fact("m-1", "watch")),
            bg(2, "watch", "early", 1),
            fact(
                3,
                NewFact {
                    kind: "model.turn".into(),
                    key: "model:1:0".into(),
                    payload: serde_json::to_value(ModelDecision::Text {
                        text: "answer without seeing it".into(),
                    })
                    .unwrap(),
                },
            ),
        ]);
        assert!(
            !state
                .messages
                .iter()
                .any(|line| line == "[background:watch] early"),
            "the in-flight call did not see the arrival; the fold must not claim it did"
        );
        assert_eq!(state.background_pending, vec!["[background:watch] early"]);
        // The next user input flushes the buffered line before itself.
        let state = step(state, &fact(4, message_fact("m-2", "next turn")), &config());
        let flushed_at = state
            .messages
            .iter()
            .position(|line| line == "[background:watch] early")
            .expect("flushed");
        let user_at = state
            .messages
            .iter()
            .position(|line| line == "user\nnext turn")
            .expect("new user line");
        assert!(flushed_at < user_at);
        assert!(state.background_pending.is_empty());
    }

    #[test]
    fn arrival_while_parked_lands_directly() {
        let state = fold(vec![
            fact(1, message_fact("m-1", "watch")),
            fact(
                2,
                NewFact {
                    kind: "model.turn".into(),
                    key: "model:1:0".into(),
                    payload: serde_json::to_value(ModelDecision::Text {
                        text: "done asking".into(),
                    })
                    .unwrap(),
                },
            ),
            bg(3, "watch", "late", 1),
        ]);
        assert_eq!(state.messages.last().unwrap(), "[background:watch] late");
        assert!(state.background_pending.is_empty());
    }

    #[test]
    fn pending_cap_records_one_truncation_marker() {
        let mut facts = vec![
            fact(1, message_fact("m-1", "watch")),
            tool_call_fact(2, "background", "call-1"),
        ];
        for index in 0..(BACKGROUND_PENDING_CAP + 10) {
            facts.push(bg(
                3 + index as u64,
                "watch",
                &format!("line-{index}"),
                index as u64 + 1,
            ));
        }
        facts.push(tool_result_fact(3 + (BACKGROUND_PENDING_CAP + 10) as u64));
        let state = fold(facts);
        let markers = state
            .messages
            .iter()
            .filter(|line| line.as_str() == BACKGROUND_TRUNCATION_LINE)
            .count();
        assert_eq!(markers, 1, "exactly one truncation marker");
        assert!(
            !state.messages.iter().any(|line| line.contains("line-70")),
            "overflow arrivals are dropped, not buffered unbounded"
        );
        assert!(state.messages.iter().any(|line| line.contains("line-0")));
        assert!(state.background_pending.is_empty());
    }

    #[test]
    fn exited_and_stderr_events_render_dispositions() {
        let state = fold(vec![
            fact(
                1,
                background_output_fact("watch", "exit status: 0", 9, Some("exited"), None),
            ),
            fact(
                2,
                background_output_fact("watch", "boom", 10, None, Some("stderr")),
            ),
        ]);
        assert!(state
            .messages
            .iter()
            .any(|line| line == "[background:watch] exited: exit status: 0"));
        assert!(state
            .messages
            .iter()
            .any(|line| line == "[background:watch] (stderr) boom"));
    }

    #[test]
    fn malformed_background_fact_fails_the_fold() {
        let state = fold(vec![Fact {
            seq: 1,
            kind: "background.output".into(),
            key: "bg:watch:1".into(),
            cause: None,
            payload: serde_json::json!({ "unexpected": true }),
        }]);
        assert!(state.schema_error.is_some());
        assert_eq!(state.phase, CodingPhase::Done);
    }
}

#[cfg(test)]
mod conversation_history_tests;
