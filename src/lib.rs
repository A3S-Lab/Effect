//! Actor framework for the A3S Code harness.
//!
//! The shape follows an immutable event log: a component folds facts into a
//! view and the transitions that view enables, and the runtime runs those
//! transitions until the fold enables nothing. Resuming a thread replays the
//! log instead of restoring process-local waits.
//!
//! The description of a transition follows the Effect v4 onboarding
//! philosophy, expressed in Rust. An [`Effect`] is a value that names its
//! success, its expected error, and the services it requires. Operators add
//! retries, timeouts, structured concurrency, resource release, and spans.
//! [`Effect::run`] is the only place a description executes.

pub mod actor;
pub mod coding;
pub mod compose;
pub mod effect;
pub mod error;
pub mod exit;
pub mod fact;
pub mod log;

pub use actor::{
    append_input, component, ingest, ingest_with_cancel, resume, resume_with_cancel, Actor,
    ErasedComponent, PreparedTransition, Settlement, Transition, TransitionProgram,
};
pub use coding::{
    answer_fact, background_output_fact, coding_actor, coding_scheduler, confirm_fact,
    ingest_coding, ingest_coding_with_cancel, merge_coding_view, message_fact, resume_coding,
    resume_coding_with_cancel, tool_args_digest, CodingPhase, CodingServices, CodingView,
    Compactor, Completion, CompletionRequest, HarnessConfig, ModelDecision, PendingConfirmation,
    PendingQuestion, PreparedToolExecution, ToolAdmissionIdentityV1, ToolCall,
    ToolEffectObservationV1, ToolExecutionOutcome, ToolExecutionResultV1, ToolFileIdentityV1,
    ToolFileKind, ToolInvocationIdentityV1, ToolObservationCompleteness, ToolPathChangeV1,
    ToolPreparationRequest, ToolPreparedV1, ToolRecoveryPolicy, ToolRunner, ToolSpec,
    MAX_TOOL_EFFECT_CANONICAL_BYTES, MAX_TOOL_EFFECT_PATHS, TOOL_PREPARED_SCHEMA,
    TOOL_RESULT_SCHEMA,
};
pub use compose::{
    budget, compact, compose_coding_actor, infer, mount, namespace_transitions, system, tools,
    with_key_namespace, HarnessGraph, HarnessPartId, HarnessView, MetaHarnessSpec,
};
pub use effect::{Effect, Schedule, TraceEvent};
pub use error::ActorError;
pub use exit::Exit;
pub use fact::{cut_log, parse_fact_json, Fact, LogCut, NewFact};
pub use log::{FileLog, FileLogRunOwnerFence, FileLogWriterFence, LogStore, MemoryLog};
