//! Versioned, dependency-neutral facts for one tool invocation.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};

use crate::error::ActorError;

pub const TOOL_PREPARED_SCHEMA: &str = "a3s.code.tool-prepared.v1";
pub const TOOL_RESULT_SCHEMA: &str = "a3s.code.tool-result.v1";
pub const MAX_TOOL_EFFECT_PATHS: usize = 4096;
pub const MAX_TOOL_EFFECT_CANONICAL_BYTES: usize = 1024 * 1024;

macro_rules! closed_set {
    ($name:ident, $($variant:ident => $text:literal),+ $(,)?) => {
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
        #[serde(rename_all = "snake_case")]
        pub enum $name { $($variant),+ }
        impl<'de> Deserialize<'de> for $name {
            fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
                let value = String::deserialize(deserializer)?;
                Self::parse(&value).map_err(serde::de::Error::custom)
            }
        }
        impl $name {
            pub fn parse(value: &str) -> Result<Self, ActorError> {
                match value {
                    $($text => Ok(Self::$variant),)+
                    _ => Err(ActorError::Schema(format!(
                        "{} refuses unknown value {value:?}", stringify!($name)
                    ))),
                }
            }
            pub const fn as_str(self) -> &'static str {
                match self { $(Self::$variant => $text,)+ }
            }
        }
    };
}

closed_set!(ToolExecutionOutcome,
    Succeeded => "succeeded", Failed => "failed", Cancelled => "cancelled",
    Denied => "denied", Unknown => "unknown",
);
closed_set!(ToolRecoveryPolicy,
    NeverReplay => "never_replay", QueryStatus => "query_status",
    IdempotencyKey => "idempotency_key",
);
closed_set!(ToolObservationCompleteness, Complete => "complete", Incomplete => "incomplete");
closed_set!(ToolFileKind,
    File => "file", Directory => "directory", Symlink => "symlink",
    Missing => "missing", Unknown => "unknown",
);

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ToolAdmissionIdentityV1 {
    pub run_id: String,
    pub thread_id: String,
    pub capability_binding_digest: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ToolInvocationIdentityV1 {
    pub turn: u64,
    pub cycle: u64,
    pub tool_call_id: String,
    pub batch_index: u32,
    /// Absent only for the dependency-neutral/legacy runner adapter.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub admission: Option<ToolAdmissionIdentityV1>,
}

impl ToolInvocationIdentityV1 {
    pub fn validate(&self) -> Result<(), ActorError> {
        if self.turn == 0 || self.tool_call_id.is_empty() || self.tool_call_id.len() > 1024 {
            return Err(identity_error("invalid scheduler identity"));
        }
        if let Some(admission) = &self.admission {
            if admission.run_id.is_empty()
                || admission.run_id.len() > 1024
                || admission.thread_id.is_empty()
                || admission.thread_id.len() > 128
            {
                return Err(identity_error("invalid admission identity"));
            }
            validate_digest(&admission.capability_binding_digest)?;
        }
        Ok(())
    }

    pub fn matches_scheduler(&self, expected: &Self) -> bool {
        self.turn == expected.turn
            && self.cycle == expected.cycle
            && self.tool_call_id == expected.tool_call_id
            && self.batch_index == expected.batch_index
    }

    pub fn key(&self) -> Result<String, ActorError> {
        self.validate()?;
        digest_json("a3s.code.tool-invocation.v1", self)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ToolFileIdentityV1 {
    pub kind: ToolFileKind,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content_digest: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mode: Option<u32>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ToolPathChangeV1 {
    pub path: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub before: Option<ToolFileIdentityV1>,
    pub after: ToolFileIdentityV1,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ToolEffectObservationV1 {
    pub changes: Vec<ToolPathChangeV1>,
    pub completeness: ToolObservationCompleteness,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub unresolved_external_operations: Vec<String>,
}

impl ToolEffectObservationV1 {
    pub fn complete(changes: Vec<ToolPathChangeV1>) -> Self {
        Self {
            changes,
            completeness: ToolObservationCompleteness::Complete,
            reason: None,
            unresolved_external_operations: Vec::new(),
        }
    }

    pub fn incomplete(reason: impl Into<String>) -> Self {
        Self {
            changes: Vec::new(),
            completeness: ToolObservationCompleteness::Incomplete,
            reason: Some(reason.into()),
            unresolved_external_operations: Vec::new(),
        }
    }

    pub fn validate(&self) -> Result<(), ActorError> {
        if self.changes.len() > MAX_TOOL_EFFECT_PATHS
            || self.unresolved_external_operations.len() > MAX_TOOL_EFFECT_PATHS
        {
            return Err(observation_error(
                "observation exceeds path/operation bound",
            ));
        }
        if self.completeness == ToolObservationCompleteness::Incomplete
            && self
                .reason
                .as_deref()
                .is_none_or(|reason| reason.trim().is_empty())
        {
            return Err(observation_error("incomplete observation has no reason"));
        }
        let mut paths = std::collections::BTreeSet::new();
        for change in &self.changes {
            let escapes_workspace = std::path::Path::new(&change.path).components().any(|part| {
                matches!(
                    part,
                    std::path::Component::RootDir
                        | std::path::Component::ParentDir
                        | std::path::Component::Prefix(_)
                )
            });
            if change.path.is_empty()
                || change.path.len() > 4096
                || change.path.contains('\0')
                || escapes_workspace
                || !paths.insert(&change.path)
            {
                return Err(observation_error("invalid or duplicate mutation path"));
            }
            for identity in change.before.iter().chain(std::iter::once(&change.after)) {
                if let Some(digest) = &identity.content_digest {
                    validate_digest(digest)?;
                }
                if self.completeness == ToolObservationCompleteness::Complete
                    && (identity.kind == ToolFileKind::Unknown
                        || (matches!(identity.kind, ToolFileKind::File | ToolFileKind::Symlink)
                            && identity.content_digest.is_none()))
                {
                    return Err(observation_error(
                        "complete observation lacks byte identity",
                    ));
                }
            }
        }
        bound_json(self)?;
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ToolPreparedV1 {
    pub schema: String,
    pub identity: ToolInvocationIdentityV1,
    pub tool_name: String,
    pub original_args_digest: String,
    pub effective_args_digest: String,
    pub recovery: ToolRecoveryPolicy,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub baseline: Option<Value>,
}

impl ToolPreparedV1 {
    pub fn validate(&self) -> Result<(), ActorError> {
        if self.schema != TOOL_PREPARED_SCHEMA {
            return Err(schema_error(&self.schema));
        }
        self.identity.validate()?;
        if self.tool_name.is_empty() || self.tool_name.len() > 1024 {
            return Err(identity_error("invalid tool name"));
        }
        validate_digest(&self.original_args_digest)?;
        validate_digest(&self.effective_args_digest)?;
        bound_json(self)?;
        Ok(())
    }

    pub fn digest(&self) -> Result<String, ActorError> {
        self.validate()?;
        digest_json("a3s.code.tool-prepared-digest.v1", self)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ToolExecutionResultV1 {
    pub schema: String,
    pub identity: ToolInvocationIdentityV1,
    pub prepared_digest: String,
    pub outcome: ToolExecutionOutcome,
    pub exit_code: Option<i32>,
    pub output: Value,
    pub effects: ToolEffectObservationV1,
}

impl ToolExecutionResultV1 {
    pub fn validate(&self) -> Result<(), ActorError> {
        if self.schema != TOOL_RESULT_SCHEMA {
            return Err(schema_error(&self.schema));
        }
        self.identity.validate()?;
        validate_digest(&self.prepared_digest)?;
        self.effects.validate()?;
        Ok(())
    }

    pub fn ensure_prepared(&self, prepared: &ToolPreparedV1) -> Result<(), ActorError> {
        self.validate()?;
        if self.identity != prepared.identity || self.prepared_digest != prepared.digest()? {
            return Err(identity_error("terminal does not bind its preparation"));
        }
        Ok(())
    }

    pub fn unknown(
        prepared: &ToolPreparedV1,
        reason: impl Into<String>,
    ) -> Result<Self, ActorError> {
        let reason = reason.into();
        Ok(Self {
            schema: TOOL_RESULT_SCHEMA.into(),
            identity: prepared.identity.clone(),
            prepared_digest: prepared.digest()?,
            outcome: ToolExecutionOutcome::Unknown,
            exit_code: None,
            output: Value::String(format!("tool_effect_unresolved: {reason}")),
            effects: ToolEffectObservationV1::incomplete(reason),
        })
    }
}

pub fn tool_args_digest(args: &Value) -> Result<String, ActorError> {
    digest_json("a3s.code.tool-arguments.v1", args)
}

fn digest_json<T: Serialize>(domain: &str, value: &T) -> Result<String, ActorError> {
    let bytes = serde_json::to_vec(value).map_err(|error| ActorError::Schema(error.to_string()))?;
    let mut digest = Sha256::new();
    digest.update(domain.as_bytes());
    digest.update([0]);
    digest.update(&bytes);
    Ok(format!("sha256:{:x}", digest.finalize()))
}

fn bound_json<T: Serialize>(value: &T) -> Result<(), ActorError> {
    let bytes = serde_json::to_vec(value).map_err(|error| ActorError::Schema(error.to_string()))?;
    if bytes.len() > MAX_TOOL_EFFECT_CANONICAL_BYTES {
        return Err(observation_error("tool effect evidence exceeds byte bound"));
    }
    Ok(())
}

fn validate_digest(value: &str) -> Result<(), ActorError> {
    let Some(hex) = value.strip_prefix("sha256:") else {
        return Err(identity_error("digest must use sha256 prefix"));
    };
    if hex.len() != 64
        || !hex
            .bytes()
            .all(|byte| byte.is_ascii_digit() || matches!(byte, b'a'..=b'f'))
    {
        return Err(identity_error("digest is not canonical SHA-256"));
    }
    Ok(())
}

fn identity_error(message: &str) -> ActorError {
    ActorError::Schema(format!("tool_effect_identity_mismatch: {message}"))
}
fn schema_error(schema: &str) -> ActorError {
    ActorError::Schema(format!("tool_effect_schema_unsupported: {schema}"))
}
fn observation_error(message: &str) -> ActorError {
    ActorError::Schema(format!("tool_effect_observation_incomplete: {message}"))
}

#[cfg(test)]
mod tests;
