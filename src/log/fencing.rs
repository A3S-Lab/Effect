//! Original log resource fences. They carry no Goal, execution or permission state.

use super::{
    restrict_handle_permissions, same_generation, storage_error, valid_thread, validate_regular,
    FileLog,
};
use crate::ActorError;
use cap_fs_ext::{FollowSymlinks, OpenOptionsFollowExt, OpenOptionsSyncExt};
use cap_std::fs::{Metadata, OpenOptions};
use std::fs::File;
use std::io;
use std::sync::{Arc, Mutex, Weak};

const WRITER_LOCK: &str = "writer.lock";

#[derive(Debug)]
pub(super) struct LockedLogFile {
    file: File,
    metadata: Metadata,
}

impl Drop for LockedLogFile {
    fn drop(&mut self) {
        // A child between fork and exec can retain the fd after our close.
        // Original Run retirement still checks its explicit fallible release.
        let _ = fs2::FileExt::unlock(&self.file);
    }
}

/// Holds original read/validate/append exclusion for this exact FileLog handle.
/// Nested operations on that handle reuse the same kernel lock.
#[derive(Debug)]
pub struct FileLogWriterFence {
    held: Option<Arc<LockedLogFile>>,
    cache: Arc<Mutex<Weak<LockedLogFile>>>,
}

impl FileLogWriterFence {
    pub fn is_for(&self, log: &FileLog) -> Result<bool, ActorError> {
        let current = log
            .writer
            .lock()
            .map_err(|_| storage_error("poisoned writer fence"))?;
        Ok(self
            .held
            .as_ref()
            .is_some_and(|held| Weak::ptr_eq(&current, &Arc::downgrade(held))))
    }
}

impl Drop for FileLogWriterFence {
    fn drop(&mut self) {
        // Close the final fd before another borrower can observe an expired Weak.
        let _cache = self
            .cache
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        drop(self.held.take());
    }
}

/// A resource retained by the existing Run owner; this is not an execution lease.
#[derive(Debug)]
pub struct FileLogRunOwnerFence {
    held: LockedLogFile,
}

impl FileLogRunOwnerFence {
    /// Original retirement must observe successful unlock before reporting readiness.
    pub fn release(&mut self) -> Result<(), ActorError> {
        fs2::FileExt::unlock(&self.held.file).map_err(storage_error)
    }
}

impl FileLog {
    /// Fail fast before a second instance can enter an original synchronous write.
    pub fn try_writer_fence(&self) -> Result<FileLogWriterFence, ActorError> {
        self.writer_fence(true)?
            .ok_or_else(|| storage_error("writer fence was not created"))
    }

    pub(super) fn existing_writer_fence(&self) -> Result<Option<FileLogWriterFence>, ActorError> {
        self.writer_fence(false)
    }

    pub(super) fn prepare_writer_fence(
        &self,
        thread_id: &str,
    ) -> Result<FileLogWriterFence, ActorError> {
        match self.dir.symlink_metadata(WRITER_LOCK) {
            Err(error) if error.kind() == io::ErrorKind::NotFound => {
                // Preserve the original corrupt-log refusal without creating state.
                self.read_opened(thread_id, false)?;
            }
            Err(error) => return Err(storage_error(error)),
            Ok(_) => {}
        }
        self.try_writer_fence()
    }

    fn writer_fence(&self, create: bool) -> Result<Option<FileLogWriterFence>, ActorError> {
        let mut current = self
            .writer
            .lock()
            .map_err(|_| storage_error("poisoned writer fence"))?;
        let held = match current.upgrade() {
            Some(held) => held,
            None => {
                if !create {
                    match self.dir.symlink_metadata(WRITER_LOCK) {
                        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(None),
                        Err(error) => return Err(storage_error(error)),
                        Ok(_) => {}
                    }
                }
                let held = Arc::new(self.lock_file(WRITER_LOCK, create)?);
                *current = Arc::downgrade(&held);
                held
            }
        };
        same_generation(
            &held.metadata,
            &self
                .dir
                .symlink_metadata(WRITER_LOCK)
                .map_err(storage_error)?,
        )?;
        Ok(Some(FileLogWriterFence {
            held: Some(held),
            cache: Arc::clone(&self.writer),
        }))
    }

    /// Pin one original logical thread through its existing RunAdmissionLease.
    pub fn try_run_owner(&self, thread_id: &str) -> Result<FileLogRunOwnerFence, ActorError> {
        valid_thread(thread_id)?;
        Ok(FileLogRunOwnerFence {
            held: self.lock_file(&format!("run-owner-{thread_id}.lock"), true)?,
        })
    }

    fn lock_file(&self, name: &str, create: bool) -> Result<LockedLogFile, ActorError> {
        let before = match self.dir.symlink_metadata(name) {
            Ok(metadata) => {
                validate_lock_file(&metadata)?;
                Some(metadata)
            }
            Err(error) if error.kind() == io::ErrorKind::NotFound => None,
            Err(error) => return Err(storage_error(error)),
        };
        let mut options = OpenOptions::new();
        options
            .read(true)
            .append(true)
            .create(create)
            .follow(FollowSymlinks::No)
            .nonblock(true);
        #[cfg(unix)]
        {
            use cap_std::fs::OpenOptionsExt;
            options.mode(0o600);
        }
        let file = self.dir.open_with(name, &options).map_err(storage_error)?;
        let metadata = file.metadata().map_err(storage_error)?;
        validate_lock_file(&metadata)?;
        if let Some(before) = before {
            same_generation(&before, &metadata)?;
        }
        let file = file.into_std();
        restrict_handle_permissions(&file, false)?;
        fs2::FileExt::try_lock_exclusive(&file).map_err(|error| {
            if error.raw_os_error() == fs2::lock_contended_error().raw_os_error() {
                ActorError::LogBusy
            } else {
                storage_error(error)
            }
        })?;
        same_generation(
            &metadata,
            &self.dir.symlink_metadata(name).map_err(storage_error)?,
        )?;
        Ok(LockedLogFile { file, metadata })
    }
}

fn validate_lock_file(metadata: &Metadata) -> Result<(), ActorError> {
    validate_regular(metadata)?;
    if metadata.len() != 0 {
        return Err(storage_error("fact log lock file must be empty"));
    }
    Ok(())
}

#[cfg(test)]
mod tests;
