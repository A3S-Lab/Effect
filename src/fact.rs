use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};

use crate::error::ActorError;

/// One immutable fact. `cause` is the transition key that wrote it, or `None`
/// when a caller method appended it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Fact {
    pub seq: u64,
    #[serde(rename = "type")]
    pub kind: String,
    pub key: String,
    pub cause: Option<String>,
    pub payload: Value,
}

#[derive(Debug, Clone, PartialEq)]
pub struct NewFact {
    pub kind: String,
    pub key: String,
    pub payload: Value,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LogCut {
    pub seq: u64,
    pub digest: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct FactDto {
    seq: u64,
    #[serde(rename = "type")]
    kind: String,
    key: String,
    cause: Option<String>,
    payload: Value,
}

pub fn parse_fact_json(value: &str) -> Result<Fact, ActorError> {
    let dto: FactDto =
        serde_json::from_str(value).map_err(|error| ActorError::Schema(error.to_string()))?;
    if dto.kind.is_empty() || dto.key.is_empty() {
        return Err(ActorError::Schema("type and key are required".into()));
    }
    Ok(Fact {
        seq: dto.seq,
        kind: dto.kind,
        key: dto.key,
        cause: dto.cause,
        payload: dto.payload,
    })
}

pub fn digest_facts(facts: &[Fact]) -> String {
    let mut hasher = Sha256::new();
    for fact in facts {
        hasher.update(fact.seq.to_string());
        hasher.update(fact.kind.as_bytes());
        hasher.update(fact.key.as_bytes());
        hasher.update(fact.cause.as_deref().unwrap_or(""));
        hasher.update(fact.payload.to_string());
        hasher.update([0xff]);
    }
    format!("{:x}", hasher.finalize())
}

pub fn cut_log(facts: &[Fact]) -> LogCut {
    LogCut {
        seq: facts.last().map(|fact| fact.seq).unwrap_or(0),
        digest: digest_facts(facts),
    }
}
