use std::collections::HashMap;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use crate::error::ActorError;
use crate::fact::{parse_fact_json, Fact, NewFact};

fn valid_thread(thread_id: &str) -> Result<(), ActorError> {
    let ok = !thread_id.is_empty()
        && thread_id.len() <= 128
        && thread_id
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '_' | '.' | ':' | '-'));
    if ok {
        Ok(())
    } else {
        Err(ActorError::BadThreadId {
            thread_id: thread_id.to_string(),
        })
    }
}

fn same_payload(stored: &Fact, incoming: &NewFact) -> bool {
    stored.kind == incoming.kind && stored.payload == incoming.payload
}

fn commit(
    current: &[Fact],
    incoming: &[NewFact],
    cause: Option<&str>,
) -> Result<Vec<Fact>, ActorError> {
    if incoming.is_empty() {
        return Ok(current.to_vec());
    }
    let already = incoming.iter().all(|item| {
        current
            .iter()
            .any(|stored| stored.key == item.key && same_payload(stored, item))
    });
    if already {
        return Ok(current.to_vec());
    }
    if let Some(conflict) = incoming
        .iter()
        .find(|item| current.iter().any(|stored| stored.key == item.key))
    {
        return Err(ActorError::DuplicateFact {
            key: conflict.key.clone(),
        });
    }
    let mut seq = current.last().map(|fact| fact.seq).unwrap_or(0);
    let mut next = current.to_vec();
    for item in incoming {
        seq += 1;
        next.push(Fact {
            seq,
            kind: item.kind.clone(),
            key: item.key.clone(),
            cause: cause.map(str::to_string),
            payload: item.payload.clone(),
        });
    }
    Ok(next)
}

pub trait LogStore: Send + Sync {
    fn read(&self, thread_id: &str) -> Result<Vec<Fact>, ActorError>;
    fn append(
        &self,
        thread_id: &str,
        facts: &[NewFact],
        cause: Option<&str>,
    ) -> Result<Vec<Fact>, ActorError>;
}

#[derive(Default)]
pub struct MemoryLog {
    threads: Mutex<HashMap<String, Vec<Fact>>>,
}

impl MemoryLog {
    pub fn new() -> Self {
        Self::default()
    }
}

impl LogStore for MemoryLog {
    fn read(&self, thread_id: &str) -> Result<Vec<Fact>, ActorError> {
        valid_thread(thread_id)?;
        let threads = self
            .threads
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        Ok(threads.get(thread_id).cloned().unwrap_or_default())
    }

    fn append(
        &self,
        thread_id: &str,
        facts: &[NewFact],
        cause: Option<&str>,
    ) -> Result<Vec<Fact>, ActorError> {
        valid_thread(thread_id)?;
        let mut threads = self
            .threads
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let current = threads.get(thread_id).cloned().unwrap_or_default();
        let next = commit(&current, facts, cause)?;
        threads.insert(thread_id.to_string(), next.clone());
        Ok(next)
    }
}

pub struct FileLog {
    dir: PathBuf,
    lock: Mutex<()>,
}

impl FileLog {
    pub fn open(dir: impl AsRef<Path>) -> Result<Self, ActorError> {
        let dir = dir.as_ref().to_path_buf();
        fs::create_dir_all(&dir).map_err(|error| ActorError::Schema(error.to_string()))?;
        Ok(Self {
            dir,
            lock: Mutex::new(()),
        })
    }

    fn path(&self, thread_id: &str) -> PathBuf {
        self.dir.join(format!("{thread_id}.jsonl"))
    }
}

impl LogStore for FileLog {
    fn read(&self, thread_id: &str) -> Result<Vec<Fact>, ActorError> {
        valid_thread(thread_id)?;
        let _guard = self
            .lock
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let path = self.path(thread_id);
        if !path.exists() {
            return Ok(Vec::new());
        }
        let raw =
            fs::read_to_string(&path).map_err(|error| ActorError::Schema(error.to_string()))?;
        let mut facts = Vec::new();
        for line in raw.lines() {
            if line.is_empty() {
                continue;
            }
            facts.push(parse_fact_json(line)?);
        }
        Ok(facts)
    }

    fn append(
        &self,
        thread_id: &str,
        facts: &[NewFact],
        cause: Option<&str>,
    ) -> Result<Vec<Fact>, ActorError> {
        valid_thread(thread_id)?;
        let _guard = self
            .lock
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let path = self.path(thread_id);
        let current = if path.exists() {
            let raw =
                fs::read_to_string(&path).map_err(|error| ActorError::Schema(error.to_string()))?;
            let mut parsed = Vec::new();
            for line in raw.lines() {
                if !line.is_empty() {
                    parsed.push(parse_fact_json(line)?);
                }
            }
            parsed
        } else {
            Vec::new()
        };
        let next = commit(&current, facts, cause)?;
        if next.len() == current.len() {
            return Ok(next);
        }
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .map_err(|error| ActorError::Schema(error.to_string()))?;
        for fact in next.iter().skip(current.len()) {
            let line = serde_json::to_string(fact)
                .map_err(|error| ActorError::Schema(error.to_string()))?;
            writeln!(file, "{line}").map_err(|error| ActorError::Schema(error.to_string()))?;
        }
        file.sync_all()
            .map_err(|error| ActorError::Schema(error.to_string()))?;
        Ok(next)
    }
}
