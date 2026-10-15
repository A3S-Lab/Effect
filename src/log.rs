use std::collections::{HashMap, HashSet};
use std::fs;
use std::io::{self, Read, Write};
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, Weak};

use cap_fs_ext::{DirExt, FollowSymlinks, MetadataExt, OpenOptionsFollowExt, OpenOptionsSyncExt};
use cap_std::fs::{Dir, File, Metadata, OpenOptions};

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
    if incoming
        .iter()
        .any(|item| item.kind.is_empty() || item.key.is_empty())
    {
        return Err(ActorError::Schema("type and key are required".into()));
    }
    let mut keys = HashSet::new();
    if let Some(duplicate) = incoming.iter().find(|item| !keys.insert(&item.key)) {
        return Err(ActorError::DuplicateFact {
            key: duplicate.key.clone(),
        });
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
    let base_seq = current.last().map(|fact| fact.seq).unwrap_or(0);
    let mut next = current.to_vec();
    for (index, item) in incoming.iter().enumerate() {
        let seq = u64::try_from(index)
            .ok()
            .and_then(|index| index.checked_add(1))
            .and_then(|offset| base_seq.checked_add(offset))
            .ok_or_else(|| ActorError::Schema("fact sequence overflow".into()))?;
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
    dir: Dir,
    lock: Mutex<()>,
    writer: Arc<Mutex<Weak<fencing::LockedLogFile>>>,
}

impl std::fmt::Debug for FileLog {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FileLog").finish_non_exhaustive()
    }
}

mod fencing;
pub use fencing::{FileLogRunOwnerFence, FileLogWriterFence};

const MAX_LOG_BYTES: u64 = 16 * 1024 * 1024;
const MAX_LOG_FACTS: usize = 100_000;

struct OpenedLog {
    file: File,
    metadata: Metadata,
    facts: Vec<Fact>,
    needs_separator: bool,
}

impl FileLog {
    /// Ambient constructor for a caller-owned location. The caller owns the
    /// parent path's authority. Hosts resolving untrusted workspace components
    /// must instead pass their already validated handle to `from_directory`.
    pub fn open(dir: impl AsRef<Path>) -> Result<Self, ActorError> {
        let path = std::path::absolute(dir).map_err(storage_error)?;
        match fs::symlink_metadata(&path) {
            Ok(metadata) if !metadata.is_dir() || metadata.is_symlink() => {
                return Err(storage_error("fact log root must be a directory"));
            }
            Ok(_) => {}
            Err(error) if error.kind() == io::ErrorKind::NotFound => {
                fs::create_dir_all(&path).map_err(storage_error)?;
            }
            Err(error) => return Err(storage_error(error)),
        }
        let name = path
            .file_name()
            .ok_or_else(|| storage_error("fact log root needs a directory name"))?;
        let parent = path
            .parent()
            .ok_or_else(|| storage_error("fact log root needs a parent"))?;
        let parent =
            Dir::open_ambient_dir(parent, cap_std::ambient_authority()).map_err(storage_error)?;
        Self::from_directory(parent.open_dir_nofollow(name).map_err(storage_error)?)
    }

    /// Bind all operations to an already authorized directory handle. This
    /// activation constructor does not create a directory or resolve an
    /// ambient path. It tightens permissions through the handle on Unix.
    pub fn from_directory(dir: Dir) -> Result<Self, ActorError> {
        let handle = dir.try_clone().map_err(storage_error)?.into_std_file();
        if !handle.metadata().map_err(storage_error)?.is_dir() {
            return Err(storage_error("fact log root must be a directory"));
        }
        restrict_handle_permissions(&handle, true)?;
        Ok(Self {
            dir,
            lock: Mutex::new(()),
            writer: Arc::new(Mutex::new(Weak::new())),
        })
    }

    fn name(thread_id: &str) -> String {
        format!("{thread_id}.jsonl")
    }

    fn tmp_name(thread_id: &str) -> String {
        static REWRITES: AtomicU64 = AtomicU64::new(0);
        let n = REWRITES.fetch_add(1, Ordering::Relaxed);
        format!("{thread_id}.jsonl.tmp-{}-{n}", std::process::id())
    }

    fn read_opened(
        &self,
        thread_id: &str,
        writable: bool,
    ) -> Result<Option<OpenedLog>, ActorError> {
        let name = Self::name(thread_id);
        let before = match self.dir.symlink_metadata(&name) {
            Ok(metadata) => metadata,
            Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(None),
            Err(error) => return Err(storage_error(error)),
        };
        // A writer-less FIFO blocks the open itself. Refuse before opening;
        // nonblocking/no-follow options also cover replacement after this gate.
        validate_regular(&before)?;
        let mut opened = self.open_regular(&name, writable)?;
        same_generation(&before, &opened.metadata)?;
        let mut raw = Vec::new();
        (&opened.file)
            .take(MAX_LOG_BYTES + 1)
            .read_to_end(&mut raw)
            .map_err(storage_error)?;
        if raw.len() as u64 > MAX_LOG_BYTES || raw.len() as u64 != opened.metadata.len() {
            return Err(storage_error("fact log changed or exceeded its byte limit"));
        }
        same_generation(
            &opened.metadata,
            &opened.file.metadata().map_err(storage_error)?,
        )?;
        let text = std::str::from_utf8(&raw).map_err(storage_error)?;
        let mut keys = HashSet::new();
        for line in text.lines().filter(|line| !line.is_empty()) {
            if opened.facts.len() == MAX_LOG_FACTS {
                return Err(storage_error("fact log fact-count limit exceeded"));
            }
            let fact = parse_fact_json(line)?;
            if fact.seq != opened.facts.len() as u64 + 1 || !keys.insert(fact.key.clone()) {
                return Err(storage_error("invalid fact log sequence or duplicate key"));
            }
            opened.facts.push(fact);
        }
        opened.needs_separator = !raw.is_empty() && !raw.ends_with(b"\n");
        Ok(Some(opened))
    }

    fn open_regular(&self, name: &str, writable: bool) -> Result<OpenedLog, ActorError> {
        let mut options = OpenOptions::new();
        options
            .read(true)
            .append(writable)
            .follow(FollowSymlinks::No)
            .nonblock(true);
        let file = self.dir.open_with(name, &options).map_err(storage_error)?;
        let metadata = file.metadata().map_err(storage_error)?;
        validate_regular(&metadata)?;
        Ok(OpenedLog {
            file,
            metadata,
            facts: Vec::new(),
            needs_separator: false,
        })
    }

    fn check_current(&self, thread_id: &str, opened: &OpenedLog) -> Result<(), ActorError> {
        same_generation(
            &opened.metadata,
            &self
                .dir
                .symlink_metadata(Self::name(thread_id))
                .map_err(storage_error)?,
        )
    }

    fn create_new(&self, name: &str) -> Result<File, ActorError> {
        let mut options = OpenOptions::new();
        options
            .read(true)
            .append(true)
            .create_new(true)
            .follow(FollowSymlinks::No)
            .nonblock(true);
        #[cfg(unix)]
        {
            use cap_std::fs::OpenOptionsExt;
            options.mode(0o600);
        }
        self.dir.open_with(name, &options).map_err(storage_error)
    }

    /// Discard every fact after `seq`, keeping the retained prefix.
    ///
    /// The retained prefix is written to a sibling scratch file, flushed, and
    /// atomically renamed over the thread's log, so a crash leaves either the
    /// complete old log or the complete truncated one — never a partial line.
    /// A parse failure or an absent thread removes nothing. Returns how many
    /// facts were removed; a cut at or beyond the last seq removes nothing and
    /// leaves the file untouched, so repeating a cut is free.
    pub fn truncate_after(&self, thread_id: &str, seq: u64) -> Result<usize, ActorError> {
        valid_thread(thread_id)?;
        let _guard = self
            .lock
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        match self.dir.symlink_metadata(Self::name(thread_id)) {
            Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(0),
            Err(error) => return Err(storage_error(error)),
            Ok(_) => {}
        }
        let _writer = self.prepare_writer_fence(thread_id)?;
        let Some(opened) = self.read_opened(thread_id, false)? else {
            return Ok(0);
        };
        let retained = opened.facts.partition_point(|fact| fact.seq <= seq);
        let removed = opened.facts.len() - retained;
        if removed == 0 {
            return Ok(0);
        }
        let bytes = encode_facts(&opened.facts[..retained], MAX_LOG_BYTES, false)?;
        let tmp = Self::tmp_name(thread_id);
        // Exclusive creation never truncates an attacker-planted scratch file.
        let mut file = self.create_new(&tmp)?;
        let rewrite = (|| {
            file.write_all(&bytes).map_err(storage_error)?;
            file.sync_all().map_err(storage_error)?;
            self.check_current(thread_id, &opened)?;
            self.dir
                .rename(&tmp, &self.dir, Self::name(thread_id))
                .map_err(storage_error)
        })();
        if rewrite.is_err() {
            let _ = self.dir.remove_file(&tmp);
        }
        rewrite?;
        Ok(removed)
    }
}

impl LogStore for FileLog {
    fn read(&self, thread_id: &str) -> Result<Vec<Fact>, ActorError> {
        valid_thread(thread_id)?;
        let _guard = self
            .lock
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let _writer = self.existing_writer_fence()?;
        Ok(self
            .read_opened(thread_id, false)?
            .map(|opened| opened.facts)
            .unwrap_or_default())
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
        let _writer = self.prepare_writer_fence(thread_id)?;
        let opened = self.read_opened(thread_id, true)?;
        let current = opened
            .as_ref()
            .map(|opened| opened.facts.as_slice())
            .unwrap_or_default();
        let next = commit(current, facts, cause)?;
        if next.len() == current.len() {
            return Ok(next);
        }
        if next.len() > MAX_LOG_FACTS {
            return Err(storage_error("fact log fact-count limit exceeded"));
        }
        let length = opened
            .as_ref()
            .map(|opened| opened.metadata.len())
            .unwrap_or(0);
        let needs_separator = opened.as_ref().is_some_and(|opened| opened.needs_separator);
        let bytes = encode_facts(
            &next[current.len()..],
            MAX_LOG_BYTES - length,
            needs_separator,
        )?;
        let mut file = match opened {
            Some(opened) => {
                self.check_current(thread_id, &opened)?;
                opened.file
            }
            None => self.create_new(&Self::name(thread_id))?,
        };
        // The exact handle used to read/validate is also the append handle.
        // A later path replacement can never redirect this write elsewhere.
        restrict_handle_permissions(&file.try_clone().map_err(storage_error)?.into_std(), false)?;
        file.write_all(&bytes).map_err(storage_error)?;
        file.sync_all().map_err(storage_error)?;
        Ok(next)
    }
}

fn validate_regular(metadata: &Metadata) -> Result<(), ActorError> {
    if !metadata.is_file() || metadata.nlink() != 1 {
        return Err(storage_error("fact log must be a single-link regular file"));
    }
    if metadata.len() > MAX_LOG_BYTES {
        return Err(storage_error("fact log byte limit exceeded"));
    }
    Ok(())
}

fn same_generation(before: &Metadata, after: &Metadata) -> Result<(), ActorError> {
    validate_regular(after)?;
    if before.dev() != after.dev()
        || before.ino() != after.ino()
        || before.len() != after.len()
        || before.modified().map_err(storage_error)? != after.modified().map_err(storage_error)?
    {
        return Err(storage_error("fact log changed during this operation"));
    }
    Ok(())
}

struct BoundedBuffer {
    bytes: Vec<u8>,
    remaining: u64,
}

impl Write for BoundedBuffer {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        if bytes.len() as u64 > self.remaining {
            return Err(io::Error::other("fact log byte limit exceeded"));
        }
        self.bytes.extend_from_slice(bytes);
        self.remaining -= bytes.len() as u64;
        Ok(bytes.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

fn encode_facts(facts: &[Fact], limit: u64, separator: bool) -> Result<Vec<u8>, ActorError> {
    let mut buffer = BoundedBuffer {
        bytes: Vec::new(),
        remaining: limit,
    };
    if separator {
        buffer.write_all(b"\n").map_err(storage_error)?;
    }
    for fact in facts {
        serde_json::to_writer(&mut buffer, fact).map_err(storage_error)?;
        buffer.write_all(b"\n").map_err(storage_error)?;
    }
    Ok(buffer.bytes)
}

fn storage_error(error: impl std::fmt::Display) -> ActorError {
    ActorError::Schema(error.to_string())
}

fn restrict_handle_permissions(file: &fs::File, directory: bool) -> Result<(), ActorError> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mode = if directory { 0o700 } else { 0o600 };
        if file.metadata().map_err(storage_error)?.permissions().mode() & 0o777 != mode {
            file.set_permissions(fs::Permissions::from_mode(mode))
                .map_err(storage_error)?;
        }
    }
    #[cfg(not(unix))]
    {
        let _ = (file, directory);
    }
    Ok(())
}

#[cfg(all(test, unix))]
mod permission_tests {
    use super::*;

    #[test]
    fn fact_log_state_is_owner_only() {
        use std::os::unix::fs::PermissionsExt;
        let root = tempfile::tempdir().unwrap();
        std::fs::set_permissions(root.path(), std::fs::Permissions::from_mode(0o755)).unwrap();
        let log_dir = root.path().join("effect-log");
        let log = FileLog::open(&log_dir).unwrap();
        log.append(
            "thread",
            &[NewFact {
                kind: "user.message".into(),
                key: "m-1".into(),
                payload: serde_json::json!({ "text": "hello" }),
            }],
            None,
        )
        .unwrap();
        assert_eq!(
            std::fs::metadata(&log_dir).unwrap().permissions().mode() & 0o777,
            0o700,
            "fact log directory must be owner-only"
        );
        let file = log_dir.join("thread.jsonl");
        assert!(file.is_file());
        assert_eq!(
            std::fs::metadata(&file).unwrap().permissions().mode() & 0o777,
            0o600,
            "fact log file must be owner-only"
        );
    }
}

#[cfg(test)]
#[path = "log_tests.rs"]
mod tests;

#[cfg(test)]
mod security_tests;
