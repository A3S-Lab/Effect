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
pub mod effect;
pub mod error;
pub mod exit;
pub mod fact;
pub mod log;

pub use actor::{component, ingest, resume, Actor, ErasedComponent, Settlement, Transition};
pub use coding::{
    answer_fact, coding_actor, confirm_fact, ingest_coding, merge_coding_view, message_fact,
    resume_coding, CodingPhase, CodingServices, CodingView, Compactor, Completion,
    CompletionRequest, HarnessConfig, ModelDecision, PendingConfirmation, PendingQuestion,
    ToolCall, ToolRunner, ToolSpec,
};
pub use effect::{Effect, Schedule, TraceEvent};
pub use error::ActorError;
pub use exit::Exit;
pub use fact::{cut_log, parse_fact_json, Fact, LogCut, NewFact};
pub use log::{FileLog, LogStore, MemoryLog};
