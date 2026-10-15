#![cfg(test)]

use super::*;

const BYTE_LIMIT: usize = 16 * 1024 * 1024;

fn input(key: &str, text: &str) -> NewFact {
    NewFact {
        kind: "user.message".into(),
        key: key.into(),
        payload: serde_json::json!({"text": text}),
    }
}

fn encoded(seq: u64, key: &str, text: &str) -> String {
    let fact = Fact {
        seq,
        kind: "user.message".into(),
        key: key.into(),
        cause: None,
        payload: serde_json::json!({"text":text}),
    };
    format!("{}\n", serde_json::to_string(&fact).expect("fact"))
}

#[cfg(unix)]
#[test]
fn read_refuses_a_thread_symlink() {
    let root = tempfile::tempdir().expect("workspace");
    let outside = root.path().join("outside.jsonl");
    std::fs::write(&outside, encoded(1, "outside", "outside fixture")).expect("fixture");
    let dir = root.path().join("log");
    let log = FileLog::open(&dir).expect("log");
    std::os::unix::fs::symlink(&outside, dir.join("thread.jsonl")).expect("link");
    assert!(
        log.read("thread").is_err(),
        "symlink bytes must not be read"
    );
}

#[cfg(unix)]
#[test]
fn append_refuses_a_thread_symlink_without_changing_its_target() {
    let root = tempfile::tempdir().expect("workspace");
    let outside = root.path().join("outside.jsonl");
    let before = encoded(1, "outside", "outside fixture");
    std::fs::write(&outside, &before).expect("fixture");
    let dir = root.path().join("log");
    let log = FileLog::open(&dir).expect("log");
    std::os::unix::fs::symlink(&outside, dir.join("thread.jsonl")).expect("link");
    let result = log.append("thread", &[input("next", "must stay local")], None);
    println!(
        "outsideChanged={}",
        std::fs::read(&outside).expect("fixture") != before.as_bytes()
    );
    assert!(result.is_err(), "append must refuse a symlink");
    assert_eq!(std::fs::read(&outside).expect("fixture"), before.as_bytes());
}

#[cfg(unix)]
#[test]
fn truncate_refuses_a_thread_symlink_without_replacing_it() {
    let root = tempfile::tempdir().expect("workspace");
    let outside = root.path().join("outside.jsonl");
    let before = encoded(1, "one", "one") + &encoded(2, "two", "two");
    std::fs::write(&outside, &before).expect("fixture");
    let dir = root.path().join("log");
    let log = FileLog::open(&dir).expect("log");
    let target = dir.join("thread.jsonl");
    std::os::unix::fs::symlink(&outside, &target).expect("link");
    assert!(log.truncate_after("thread", 1).is_err());
    assert!(std::fs::symlink_metadata(&target)
        .expect("link")
        .is_symlink());
    assert_eq!(std::fs::read(&outside).expect("fixture"), before.as_bytes());
}

#[cfg(unix)]
#[test]
fn open_refuses_a_symlinked_log_directory() {
    let root = tempfile::tempdir().expect("workspace");
    let outside = tempfile::tempdir().expect("outside");
    let link = root.path().join("log");
    std::os::unix::fs::symlink(outside.path(), &link).expect("link");
    assert!(FileLog::open(&link).is_err());
    assert_eq!(
        std::fs::read_dir(outside.path()).expect("outside").count(),
        0
    );
}

#[cfg(unix)]
#[test]
fn an_open_log_stays_bound_when_its_directory_path_is_replaced() {
    let root = tempfile::tempdir().expect("workspace");
    let dir = root.path().join("log");
    let log = FileLog::open(&dir).expect("log");
    let first = log
        .append("thread", &[input("original", "original")], None)
        .expect("seed");
    let moved = root.path().join("original-directory");
    std::fs::rename(&dir, &moved).expect("move");
    let outside = tempfile::tempdir().expect("outside");
    let external = outside.path().join("thread.jsonl");
    let before = encoded(1, "foreign", "foreign");
    std::fs::write(&external, &before).expect("external fixture");
    std::os::unix::fs::symlink(outside.path(), &dir).expect("replacement");
    let read = log.read("thread").expect("read");
    let appended = log
        .append("thread", &[input("next", "next")], None)
        .expect("append");
    assert_eq!(read, first, "the original directory owns this log");
    assert_eq!(appended[0].key, "original");
    assert_eq!(
        std::fs::read(&external).expect("external"),
        before.as_bytes()
    );
    assert_eq!(
        FileLog::open(&moved)
            .expect("original")
            .read("thread")
            .expect("facts")
            .len(),
        2
    );
}

#[test]
fn oversized_read_refuses_instead_of_loading_a_valid_large_record() {
    let root = tempfile::tempdir().expect("workspace");
    let log = FileLog::open(root.path()).expect("log");
    std::fs::write(
        root.path().join("thread.jsonl"),
        encoded(1, "large", &"x".repeat(BYTE_LIMIT)),
    )
    .expect("large fixture");
    assert!(log.read("thread").is_err());
}

#[test]
fn oversized_append_refuses_before_creating_a_log_file() {
    let root = tempfile::tempdir().expect("workspace");
    let log = FileLog::open(root.path()).expect("log");
    assert!(log
        .append("thread", &[input("large", &"x".repeat(BYTE_LIMIT))], None)
        .is_err());
    assert!(!root.path().join("thread.jsonl").exists());
}

#[test]
fn duplicate_keys_in_one_batch_refuse_atomically_on_both_stores() {
    let root = tempfile::tempdir().expect("workspace");
    let memory = MemoryLog::new();
    let file = FileLog::open(root.path()).expect("log");
    for log in [&memory as &dyn LogStore, &file as &dyn LogStore] {
        let result = log.append(
            "thread",
            &[input("same", "first"), input("same", "second")],
            None,
        );
        assert!(matches!(result, Err(ActorError::DuplicateFact { .. })));
        assert!(log.read("thread").expect("facts").is_empty());
    }
    assert!(!root.path().join("thread.jsonl").exists());
}

#[test]
fn missing_fact_fields_refuse_a_whole_batch_on_both_stores() {
    let root = tempfile::tempdir().expect("workspace");
    let memory = MemoryLog::new();
    let file = FileLog::open(root.path()).expect("log");
    let mut missing_kind = input("missing-kind", "invalid");
    missing_kind.kind.clear();
    let mut missing_key = input("missing-key", "invalid");
    missing_key.key.clear();
    for log in [&memory as &dyn LogStore, &file as &dyn LogStore] {
        for invalid in [&missing_kind, &missing_key] {
            assert!(matches!(
                log.append("thread", &[input("valid", "valid"), invalid.clone()], None),
                Err(ActorError::Schema(_))
            ));
            assert!(log.read("thread").expect("unchanged facts").is_empty());
        }
    }
    assert!(!root.path().join("thread.jsonl").exists());
}

#[test]
fn non_contiguous_sequence_refuses_before_append_or_truncation() {
    let root = tempfile::tempdir().expect("workspace");
    let log = FileLog::open(root.path()).expect("log");
    let path = root.path().join("thread.jsonl");
    let before = encoded(2, "wrong-seq", "wrong sequence");
    std::fs::write(&path, &before).expect("fixture");
    assert!(log.read("thread").is_err());
    assert!(log.append("thread", &[input("new", "new")], None).is_err());
    assert!(log.truncate_after("thread", 1).is_err());
    assert_eq!(std::fs::read(path).expect("fixture"), before.as_bytes());
}

#[test]
fn duplicate_stored_keys_refuse_before_append_or_truncation() {
    let root = tempfile::tempdir().expect("workspace");
    let log = FileLog::open(root.path()).expect("log");
    let path = root.path().join("thread.jsonl");
    let before = encoded(1, "same", "one") + &encoded(2, "same", "two");
    std::fs::write(&path, &before).expect("fixture");
    assert!(log.read("thread").is_err());
    assert!(log.append("thread", &[input("new", "new")], None).is_err());
    assert!(log.truncate_after("thread", 1).is_err());
    assert_eq!(std::fs::read(path).expect("fixture"), before.as_bytes());
}

#[test]
fn from_directory_uses_the_supplied_capability_without_creating_state() {
    let root = tempfile::tempdir().expect("workspace");
    let dir = Dir::open_ambient_dir(root.path(), cap_std::ambient_authority()).expect("handle");
    let log = FileLog::from_directory(dir).expect("bound log");
    assert!(log.read("missing").expect("absent").is_empty());
    assert_eq!(log.truncate_after("missing", 0).expect("absent"), 0);
    assert_eq!(
        std::fs::read_dir(root.path()).expect("workspace").count(),
        0
    );
    let first = log
        .append("thread", &[input("one", "one")], None)
        .expect("append");
    assert_eq!(log.read("thread").expect("read"), first);
}

#[cfg(unix)]
#[test]
fn fifo_is_refused_before_open_for_all_operations() {
    let root = tempfile::tempdir().expect("workspace");
    let log = FileLog::open(root.path()).expect("log");
    let path = root.path().join("thread.jsonl");
    assert!(std::process::Command::new("mkfifo")
        .arg(&path)
        .status()
        .expect("mkfifo")
        .success());
    assert!(log.read("thread").is_err());
    assert!(log.append("thread", &[input("new", "new")], None).is_err());
    assert!(log.truncate_after("thread", 0).is_err());
}

#[cfg(unix)]
#[test]
fn opened_handle_gate_refuses_a_fifo_swapped_in_after_precheck() {
    let root = tempfile::tempdir().expect("workspace");
    let log = FileLog::open(root.path()).expect("log");
    let path = root.path().join("thread.jsonl");
    std::fs::write(&path, encoded(1, "one", "one")).expect("fixture");
    let before = log.dir.symlink_metadata("thread.jsonl").expect("precheck");
    validate_regular(&before).expect("regular at precheck");
    std::fs::remove_file(&path).expect("replace");
    assert!(std::process::Command::new("mkfifo")
        .arg(&path)
        .status()
        .expect("mkfifo")
        .success());
    assert!(log.open_regular("thread.jsonl", false).is_err());
    assert!(log.open_regular("thread.jsonl", true).is_err());
}

#[cfg(unix)]
#[test]
fn nofollow_gate_refuses_a_symlink_swapped_in_after_precheck() {
    let root = tempfile::tempdir().expect("workspace");
    let log = FileLog::open(root.path()).expect("log");
    let path = root.path().join("thread.jsonl");
    std::fs::write(&path, encoded(1, "one", "one")).expect("fixture");
    let before = log.dir.symlink_metadata("thread.jsonl").expect("precheck");
    validate_regular(&before).expect("regular at precheck");
    std::fs::remove_file(&path).expect("replace");
    let external = tempfile::NamedTempFile::new().expect("outside");
    std::fs::write(external.path(), encoded(1, "outside", "outside")).expect("fixture");
    std::os::unix::fs::symlink(external.path(), &path).expect("link");
    assert!(log.open_regular("thread.jsonl", false).is_err());
    assert!(log.open_regular("thread.jsonl", true).is_err());
}

#[test]
fn replacing_a_validated_regular_file_refuses_the_pending_mutation() {
    let root = tempfile::tempdir().expect("workspace");
    let log = FileLog::open(root.path()).expect("log");
    let path = root.path().join("thread.jsonl");
    std::fs::write(&path, encoded(1, "one", "one")).expect("fixture");
    let opened = log
        .read_opened("thread", true)
        .expect("read")
        .expect("exists");
    std::fs::rename(&path, root.path().join("original.jsonl")).expect("move");
    std::fs::write(&path, encoded(1, "two", "two")).expect("replacement");
    assert!(log.check_current("thread", &opened).is_err());
}

#[test]
fn hard_linked_thread_is_refused_without_changing_either_name() {
    let root = tempfile::tempdir().expect("workspace");
    let log = FileLog::open(root.path()).expect("log");
    let external = root.path().join("outside.jsonl");
    let before = encoded(1, "outside", "outside");
    std::fs::write(&external, &before).expect("fixture");
    let path = root.path().join("thread.jsonl");
    std::fs::hard_link(&external, &path).expect("hard link");
    assert!(log.read("thread").is_err());
    assert!(log.append("thread", &[input("new", "new")], None).is_err());
    assert!(log.truncate_after("thread", 0).is_err());
    assert_eq!(std::fs::read(path).expect("thread"), before.as_bytes());
    assert_eq!(std::fs::read(external).expect("outside"), before.as_bytes());
}

#[test]
fn exclusive_creation_refuses_an_existing_scratch_file() {
    let root = tempfile::tempdir().expect("workspace");
    let log = FileLog::open(root.path()).expect("log");
    let path = root.path().join("planted.tmp");
    std::fs::write(&path, "must survive").expect("fixture");
    assert!(log.create_new("planted.tmp").is_err());
    assert_eq!(
        std::fs::read_to_string(path).expect("fixture"),
        "must survive"
    );
}

#[cfg(unix)]
#[test]
fn exclusive_creation_refuses_a_planted_scratch_symlink() {
    let root = tempfile::tempdir().expect("workspace");
    let log = FileLog::open(root.path()).expect("log");
    let external = tempfile::NamedTempFile::new().expect("outside");
    std::fs::write(external.path(), "must survive").expect("fixture");
    std::os::unix::fs::symlink(external.path(), root.path().join("planted.tmp")).expect("link");
    assert!(log.create_new("planted.tmp").is_err());
    assert_eq!(
        std::fs::read_to_string(external.path()).expect("fixture"),
        "must survive"
    );
}

#[test]
fn a_valid_final_line_without_a_newline_can_be_extended_and_reopened() {
    let root = tempfile::tempdir().expect("workspace");
    let log = FileLog::open(root.path()).expect("log");
    std::fs::write(
        root.path().join("thread.jsonl"),
        encoded(1, "one", "one").trim_end(),
    )
    .expect("fixture");
    let next = log
        .append("thread", &[input("two", "two")], Some("one"))
        .expect("append");
    assert_eq!(next.len(), 2);
    assert_eq!(next[1].cause.as_deref(), Some("one"));
    assert_eq!(
        FileLog::open(root.path())
            .expect("reopen")
            .read("thread")
            .expect("facts"),
        next
    );
}

#[test]
fn exact_byte_limit_is_readable_but_growth_is_refused_without_rewriting() {
    let root = tempfile::tempdir().expect("workspace");
    let log = FileLog::open(root.path()).expect("log");
    let overhead = encoded(1, "edge", "").len();
    let text = "x".repeat(BYTE_LIMIT - overhead);
    let path = root.path().join("thread.jsonl");
    let next = log
        .append("thread", &[input("edge", &text)], None)
        .expect("exact bound");
    assert_eq!(
        std::fs::metadata(&path).expect("size").len(),
        BYTE_LIMIT as u64
    );
    assert_eq!(log.read("thread").expect("read"), next);
    let before = std::fs::read(&path).expect("bytes");
    assert_eq!(
        log.append("thread", &[input("edge", &text)], None)
            .expect("idempotent"),
        next
    );
    assert!(log.append("thread", &[input("new", "new")], None).is_err());
    assert_eq!(std::fs::read(&path).expect("bytes"), before);
    assert_eq!(log.truncate_after("thread", 0).expect("truncate"), 1);
    assert!(log.read("thread").expect("empty").is_empty());
}

#[test]
fn fact_count_limit_refuses_growth_and_an_over_limit_stored_log() {
    let root = tempfile::tempdir().expect("workspace");
    let log = FileLog::open(root.path()).expect("log");
    let path = root.path().join("thread.jsonl");
    {
        let mut fixture = std::io::BufWriter::new(std::fs::File::create(&path).expect("fixture"));
        for seq in 1..=MAX_LOG_FACTS {
            fixture
                .write_all(encoded(seq as u64, &format!("key-{seq}"), "x").as_bytes())
                .expect("fact");
        }
        fixture.flush().expect("flush");
    }
    assert!(std::fs::metadata(&path).expect("size").len() < MAX_LOG_BYTES);
    assert_eq!(log.read("thread").expect("at bound").len(), MAX_LOG_FACTS);
    let before = std::fs::read(&path).expect("bytes");
    assert!(log.append("thread", &[input("new", "new")], None).is_err());
    assert_eq!(std::fs::read(&path).expect("bytes"), before);
    let mut fixture = std::fs::OpenOptions::new()
        .append(true)
        .open(&path)
        .expect("fixture");
    fixture
        .write_all(encoded(MAX_LOG_FACTS as u64 + 1, "overflow", "x").as_bytes())
        .expect("overflow");
    assert!(log.read("thread").is_err());
    assert!(log.truncate_after("thread", 0).is_err());
}

#[test]
fn incomplete_tail_refuses_without_quarantine_or_rewrite() {
    let root = tempfile::tempdir().expect("workspace");
    let log = FileLog::open(root.path()).expect("log");
    let path = root.path().join("thread.jsonl");
    let before = encoded(1, "one", "one") + "{\"seq\":2";
    std::fs::write(&path, &before).expect("fixture");
    assert!(log.read("thread").is_err());
    assert!(log.append("thread", &[input("new", "new")], None).is_err());
    assert!(log.truncate_after("thread", 0).is_err());
    assert_eq!(std::fs::read(path).expect("bytes"), before.as_bytes());
    assert_eq!(
        std::fs::read_dir(root.path()).expect("workspace").count(),
        1
    );
}
