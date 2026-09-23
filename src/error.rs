use thiserror::Error;

/// Expected failures of the actor runtime. Defects use [`crate::Exit::Die`].
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum ActorError {
    #[error("fact key {key} was reused with a different payload")]
    DuplicateFact { key: String },
    #[error("two components enabled transition {key}")]
    DuplicateTransition { key: String },
    #[error("settled after {limit} transitions without becoming quiescent")]
    StepLimit { limit: u32 },
    #[error("thread id {thread_id:?} is not a single path segment")]
    BadThreadId { thread_id: String },
    #[error("fact schema: {0}")]
    Schema(String),
    #[error("config: {0}")]
    Config(String),
    /// A defect. The coding fold does not retry it and does not append a fact.
    #[error("{0}")]
    Defect(String),
    #[error("transition {key} failed: {message}")]
    Handler { key: String, message: String },
}
