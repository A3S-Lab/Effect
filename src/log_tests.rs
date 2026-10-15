#![cfg(test)]

use super::*;

fn user_fact(key: &str, text: &str) -> NewFact {
    NewFact {
        kind: "user.message".into(),
        key: key.into(),
        payload: serde_json::json!({ "text": text }),
    }
}

fn log_with(dir: &tempfile::TempDir, count: u64) -> FileLog {
    let log = FileLog::open(dir.path()).unwrap();
    for seq in 1..=count {
        log.append(
            "thread-1",
            &[user_fact(&format!("m-{seq}"), &format!("note {seq}"))],
            None,
        )
        .unwrap();
    }
    log
}

#[test]
fn truncate_after_keeps_the_prefix_and_reports_the_removed_count() {
    let dir = tempfile::tempdir().unwrap();
    let log = log_with(&dir, 5);
    let removed = log.truncate_after("thread-1", 3).unwrap();
    assert_eq!(removed, 2);
    let facts = log.read("thread-1").unwrap();
    assert_eq!(facts.len(), 3);
    assert_eq!(facts.last().unwrap().seq, 3);
    assert_eq!(facts.last().unwrap().key, "m-3");
    // The rewrite must survive a reopen and stay owner-only.
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let file = dir.path().join("thread-1.jsonl");
        assert_eq!(
            std::fs::metadata(&file).unwrap().permissions().mode() & 0o777,
            0o600,
            "the rewritten fact log must stay owner-only"
        );
    }
    let reopened = FileLog::open(dir.path()).unwrap();
    assert_eq!(reopened.read("thread-1").unwrap().len(), 3);
    // The next append continues the retained timeline: seq 4 is free again.
    reopened
        .append("thread-1", &[user_fact("m-after-fork", "next")], None)
        .unwrap();
    let facts = reopened.read("thread-1").unwrap();
    assert_eq!(facts.last().unwrap().seq, 4);
    assert_eq!(facts.last().unwrap().key, "m-after-fork");
    // The scratch file never survives a successful rename.
    let scratch = std::fs::read_dir(dir.path())
        .unwrap()
        .filter_map(|entry| entry.ok())
        .filter(|entry| entry.file_name().to_string_lossy().contains("tmp-"))
        .count();
    assert_eq!(
        scratch, 0,
        "a successful rewrite must leave no scratch file"
    );
}

#[test]
fn truncate_after_beyond_the_end_removes_nothing() {
    let dir = tempfile::tempdir().unwrap();
    let log = log_with(&dir, 3);
    let file = dir.path().join("thread-1.jsonl");
    let before = std::fs::read(&file).unwrap();
    assert_eq!(log.truncate_after("thread-1", 99).unwrap(), 0);
    assert_eq!(
        std::fs::read(&file).unwrap(),
        before,
        "a no-op cut must not rewrite the log"
    );
    assert_eq!(log.read("thread-1").unwrap().len(), 3);
    // A thread with no log file at all is a no-op, not an error.
    assert_eq!(log.truncate_after("absent-thread", 4).unwrap(), 0);
}

#[test]
fn truncate_after_at_the_same_cut_is_idempotent() {
    let dir = tempfile::tempdir().unwrap();
    let log = log_with(&dir, 5);
    assert_eq!(log.truncate_after("thread-1", 3).unwrap(), 2);
    let file = dir.path().join("thread-1.jsonl");
    let settled = std::fs::read(&file).unwrap();
    assert_eq!(log.truncate_after("thread-1", 3).unwrap(), 0);
    assert_eq!(std::fs::read(&file).unwrap(), settled);
    assert_eq!(log.read("thread-1").unwrap().len(), 3);
}

#[test]
fn truncate_after_rejects_an_invalid_thread_id() {
    let dir = tempfile::tempdir().unwrap();
    let log = log_with(&dir, 2);
    assert!(matches!(
        log.truncate_after("bad/thread", 1),
        Err(ActorError::BadThreadId { .. })
    ));
    assert_eq!(log.read("thread-1").unwrap().len(), 2);
}

#[test]
fn truncate_after_on_a_corrupt_line_leaves_the_log_untouched() {
    let dir = tempfile::tempdir().unwrap();
    let log = log_with(&dir, 2);
    drop(log);
    let file = dir.path().join("thread-1.jsonl");
    let mut handle = std::fs::OpenOptions::new()
        .append(true)
        .open(&file)
        .unwrap();
    writeln!(handle, "{{").unwrap();
    drop(handle);
    let before = std::fs::read(&file).unwrap();
    let log = FileLog::open(dir.path()).unwrap();
    assert!(matches!(
        log.truncate_after("thread-1", 1),
        Err(ActorError::Schema(_))
    ));
    assert_eq!(std::fs::read(&file).unwrap(), before);
}
