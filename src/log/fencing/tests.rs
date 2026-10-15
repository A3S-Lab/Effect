#![cfg(test)]

use super::*;
use crate::{LogStore, NewFact};
use std::process::Command;

fn fact(key: &str) -> NewFact {
    NewFact {
        kind: "test.fact".into(),
        key: key.into(),
        payload: serde_json::json!({"key":key}),
    }
}

#[test]
fn writer_fence_is_exact_reentrant_and_survives_until_the_last_borrow() {
    let directory = tempfile::tempdir().unwrap();
    let first = FileLog::open(directory.path()).unwrap();
    let second = FileLog::open(directory.path()).unwrap();
    let fence = first.try_writer_fence().unwrap();
    assert!(fence.is_for(&first).unwrap());
    assert!(!fence.is_for(&second).unwrap());
    first.append("thread", &[fact("one")], None).unwrap();
    let nested = first.try_writer_fence().unwrap();
    assert!(matches!(second.read("thread"), Err(ActorError::LogBusy)));
    assert!(matches!(
        second.append("thread", &[fact("bad")], None),
        Err(ActorError::LogBusy)
    ));
    assert!(matches!(
        second.truncate_after("thread", 0),
        Err(ActorError::LogBusy)
    ));
    drop(fence);
    assert!(matches!(
        second.try_writer_fence(),
        Err(ActorError::LogBusy)
    ));
    drop(nested);
    assert_eq!(second.read("thread").unwrap().len(), 1);
    assert_eq!(
        second.append("thread", &[fact("two")], None).unwrap().len(),
        2
    );
}

#[test]
fn same_handle_concurrent_fence_turnover_never_contends_with_itself() {
    let directory = tempfile::tempdir().unwrap();
    let log = Arc::new(FileLog::open(directory.path()).unwrap());
    let start = Arc::new(std::sync::Barrier::new(4));
    let workers = (0..4)
        .map(|_| {
            let log = Arc::clone(&log);
            let start = Arc::clone(&start);
            std::thread::spawn(move || {
                start.wait();
                for _ in 0..10_000 {
                    drop(log.try_writer_fence().unwrap());
                }
            })
        })
        .collect::<Vec<_>>();
    for worker in workers {
        worker.join().unwrap();
    }
}

#[cfg(unix)]
#[test]
fn releasing_fences_does_not_wait_for_an_inherited_file_description() {
    let directory = tempfile::tempdir().unwrap();
    let first = FileLog::open(directory.path()).unwrap();
    let second = FileLog::open(directory.path()).unwrap();
    let writer = first.try_writer_fence().unwrap();
    // dup shares the open file description, as fork does before close-on-exec.
    let inherited = writer.held.as_ref().unwrap().file.try_clone().unwrap();
    assert!(matches!(
        second.try_writer_fence(),
        Err(ActorError::LogBusy)
    ));
    drop(writer);
    let replacement = second.try_writer_fence();
    drop(inherited);
    drop(replacement.expect("the parent writer released despite the inherited fd"));

    let owner = first.try_run_owner("inherited-owner").unwrap();
    let inherited = owner.held.file.try_clone().unwrap();
    assert!(matches!(
        second.try_run_owner("inherited-owner"),
        Err(ActorError::LogBusy)
    ));
    drop(owner);
    let replacement = second.try_run_owner("inherited-owner");
    drop(inherited);
    drop(replacement.expect("a nonexecuting owner released despite the inherited fd"));
}

#[cfg(unix)]
#[test]
fn a_child_parked_before_exec_does_not_extend_parent_fences() {
    use std::io::{Read, Write};
    use std::os::unix::{net::UnixStream, process::CommandExt};
    let directory = tempfile::tempdir().unwrap();
    let first = FileLog::open(directory.path()).unwrap();
    let second = FileLog::open(directory.path()).unwrap();
    let writer = first.try_writer_fence().unwrap();
    let owner = first.try_run_owner("forked-owner").unwrap();
    let (mut parent, mut child) = UnixStream::pair().unwrap();
    for socket in [&parent, &child] {
        socket
            .set_read_timeout(Some(std::time::Duration::from_secs(5)))
            .unwrap();
    }
    let mut command = Command::new(std::env::current_exe().unwrap());
    command.arg("--list").stdout(std::process::Stdio::null());
    // SAFETY: after fork, only bounded socket read/write syscalls; no allocation or locks.
    unsafe {
        command.pre_exec(move || {
            child.write_all(&[1])?;
            child.read_exact(&mut [0])?;
            Ok(())
        });
    }
    let spawning = std::thread::spawn(move || command.spawn());
    parent.read_exact(&mut [0]).unwrap();
    assert!(matches!(
        second.try_writer_fence(),
        Err(ActorError::LogBusy)
    ));
    assert!(matches!(
        second.try_run_owner("forked-owner"),
        Err(ActorError::LogBusy)
    ));
    drop((writer, owner));
    let next_writer = second.try_writer_fence();
    let next_owner = second.try_run_owner("forked-owner");
    parent.write_all(&[1]).unwrap();
    assert!(spawning.join().unwrap().unwrap().wait().unwrap().success());
    drop(next_writer.expect("the parked child must not keep the parent writer locked"));
    drop(next_owner.expect("the parked child must not keep the parent owner locked"));
}

#[test]
fn owner_fence_uses_original_thread_and_keeps_status_reads_available() {
    let directory = tempfile::tempdir().unwrap();
    let first = FileLog::open(directory.path()).unwrap();
    let second = FileLog::open(directory.path()).unwrap();
    let mut owner = first.try_run_owner("session-a").unwrap();
    assert!(matches!(
        second.try_run_owner("session-a"),
        Err(ActorError::LogBusy)
    ));
    let other = second.try_run_owner("session-b").unwrap();
    assert!(second.read("session-a").unwrap().is_empty());
    owner.release().unwrap();
    let replacement = second.try_run_owner("session-a").unwrap();
    drop((owner, other, replacement));
}

fn probe(directory: &std::path::Path, kind: &str, busy: bool) {
    let output = Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "log::fencing::tests::child_process_probe",
            "--ignored",
            "--nocapture",
        ])
        .env("A3S_EFFECT_FENCE_TEST_DIR", directory)
        .env("A3S_EFFECT_FENCE_TEST_KIND", kind)
        .env("A3S_EFFECT_FENCE_TEST_BUSY", busy.to_string())
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "child probe: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn real_child_process_observes_writer_owner_and_release_boundaries() {
    let directory = tempfile::tempdir().unwrap();
    let log = FileLog::open(directory.path()).unwrap();
    let writer = log.try_writer_fence().unwrap();
    probe(directory.path(), "writer", true);
    drop(writer);
    probe(directory.path(), "writer", false);
    let mut owner = log.try_run_owner("session-a").unwrap();
    probe(directory.path(), "owner", true);
    owner.release().unwrap();
    probe(directory.path(), "owner", false);
}

#[test]
#[ignore = "invoked by the bounded parent process test with an explicit temporary log"]
fn child_process_probe() {
    let directory = std::env::var_os("A3S_EFFECT_FENCE_TEST_DIR").unwrap();
    let kind = std::env::var("A3S_EFFECT_FENCE_TEST_KIND").unwrap();
    let busy: bool = std::env::var("A3S_EFFECT_FENCE_TEST_BUSY")
        .unwrap()
        .parse()
        .unwrap();
    let log = FileLog::open(directory).unwrap();
    let result = match kind.as_str() {
        "writer" => log.try_writer_fence().map(|_| ()),
        "owner" => {
            assert!(log.read("session-a").unwrap().is_empty());
            log.try_run_owner("session-a").map(|_| ())
        }
        _ => panic!("unknown test probe kind"),
    };
    if busy {
        assert!(matches!(result, Err(ActorError::LogBusy)));
    } else {
        result.unwrap();
    }
}

#[test]
fn nonempty_or_directory_lock_paths_refuse_without_touching_facts() {
    for directory_lock in [false, true] {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join(WRITER_LOCK);
        if directory_lock {
            std::fs::create_dir(&path).unwrap();
        } else {
            std::fs::write(&path, b"protected bytes").unwrap();
        }
        let log = FileLog::open(directory.path()).unwrap();
        assert!(log.append("thread", &[fact("bad")], None).is_err());
        assert!(!directory.path().join("thread.jsonl").exists());
        if !directory_lock {
            assert_eq!(std::fs::read(path).unwrap(), b"protected bytes");
        }
    }
}

#[cfg(unix)]
#[test]
fn linked_lock_paths_refuse_before_changing_the_external_file() {
    use std::os::unix::fs::{symlink, PermissionsExt};
    for hard_link in [false, true] {
        let directory = tempfile::tempdir().unwrap();
        let outside = tempfile::NamedTempFile::new().unwrap();
        std::fs::set_permissions(outside.path(), std::fs::Permissions::from_mode(0o644)).unwrap();
        let path = directory.path().join(WRITER_LOCK);
        if hard_link {
            std::fs::hard_link(outside.path(), path).unwrap();
        } else {
            symlink(outside.path(), path).unwrap();
        }
        let log = FileLog::open(directory.path()).unwrap();
        assert!(log.try_writer_fence().is_err());
        assert_eq!(
            std::fs::metadata(outside.path())
                .unwrap()
                .permissions()
                .mode()
                & 0o777,
            0o644
        );
    }
}

#[cfg(unix)]
#[test]
fn writerless_fifo_lock_is_refused_without_waiting_for_a_writer() {
    let directory = tempfile::tempdir().unwrap();
    assert!(Command::new("mkfifo")
        .arg(directory.path().join(WRITER_LOCK))
        .status()
        .unwrap()
        .success());
    let log = FileLog::open(directory.path()).unwrap();
    assert!(log.try_writer_fence().is_err());
}

#[cfg(unix)]
#[test]
fn a_replaced_writer_lock_refuses_the_old_handle_before_append() {
    let directory = tempfile::tempdir().unwrap();
    let log = FileLog::open(directory.path()).unwrap();
    let fence = log.try_writer_fence().unwrap();
    std::fs::rename(
        directory.path().join(WRITER_LOCK),
        directory.path().join("old-lock"),
    )
    .unwrap();
    std::fs::write(directory.path().join(WRITER_LOCK), b"").unwrap();
    assert!(log.append("thread", &[fact("bad")], None).is_err());
    assert!(!directory.path().join("thread.jsonl").exists());
    drop(fence);
}

#[test]
fn invalid_owner_thread_does_not_create_an_escaped_lock() {
    let directory = tempfile::tempdir().unwrap();
    let log = FileLog::open(directory.path()).unwrap();
    for bad in ["", "../escape", "a/b", "bad\0thread"] {
        assert!(matches!(
            log.try_run_owner(bad),
            Err(ActorError::BadThreadId { .. })
        ));
    }
}
