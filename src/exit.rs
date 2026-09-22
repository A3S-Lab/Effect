/// How a program stops.
///
/// `Fail` is an expected error and stays in the type. `Die` is a defect.
/// `Interrupt` is cancellation from a parent scope. Retries and `catch_fail`
/// see only `Fail`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Exit<E> {
    Fail(E),
    Die(String),
    Interrupt,
}

impl<E> Exit<E> {
    pub fn is_interrupt(&self) -> bool {
        matches!(self, Self::Interrupt)
    }
}

impl From<crate::error::ActorError> for Exit<crate::error::ActorError> {
    fn from(error: crate::error::ActorError) -> Self {
        Self::Fail(error)
    }
}
