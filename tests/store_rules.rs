use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use a3s_effect::{
    component, cut_log, parse_fact_json, resume, Actor, ActorError, Effect, Exit, Fact,
    HarnessConfig, LogStore, MemoryLog, NewFact, Transition,
};

struct RemovedDir(PathBuf);

impl RemovedDir {
    fn new(label: &str) -> Self {
        static N: AtomicUsize = AtomicUsize::new(0);
        let dir = std::env::temp_dir().join(format!(
            "a3s-effect-{label}-{}-{}",
            std::process::id(),
            N.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = std::fs::remove_dir_all(&dir);
        Self(dir)
    }
}

impl Drop for RemovedDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn fact(seq: u64, kind: &str, key: &str, cause: Option<&str>, payload: serde_json::Value) -> Fact {
    Fact {
        seq,
        kind: kind.to_string(),
        key: key.to_string(),
        cause: cause.map(str::to_string),
        payload,
    }
}

fn message(text: &str) -> NewFact {
    NewFact {
        kind: "user.message".into(),
        key: "m1".into(),
        payload: serde_json::json!({ "text": text }),
    }
}

#[test]
fn parse_fact_json_round_trips_and_rejects_bad_shapes() {
    let original = fact(
        1,
        "user.message",
        "m1",
        None,
        serde_json::json!({ "text": "hi" }),
    );
    let raw = serde_json::to_string(&original).unwrap();
    assert!(raw.contains("\"type\":\"user.message\""));
    assert_eq!(parse_fact_json(&raw).unwrap(), original);

    let unknown = parse_fact_json(
        r#"{"seq":1,"type":"user.message","key":"m1","cause":null,"payload":{},"extra":1}"#,
    )
    .unwrap_err();
    assert!(matches!(unknown, ActorError::Schema(_)));

    let empty_type =
        parse_fact_json(r#"{"seq":1,"type":"","key":"m1","cause":null,"payload":{}}"#).unwrap_err();
    assert_eq!(
        empty_type,
        ActorError::Schema("type and key are required".into())
    );

    let empty_key =
        parse_fact_json(r#"{"seq":1,"type":"user.message","key":"","cause":null,"payload":{}}"#)
            .unwrap_err();
    assert_eq!(
        empty_key,
        ActorError::Schema("type and key are required".into())
    );
}

fn assert_append_rules(log: &dyn LogStore, jsonl: Option<&Path>) {
    let first = log
        .append("thread-1", &[message("hi")], None)
        .expect("append");
    let before = jsonl.map(|path| std::fs::read(path).unwrap());
    let second = log
        .append("thread-1", &[message("hi")], None)
        .expect("replay");
    assert_eq!(first, second);
    let stored = log.read("thread-1").unwrap();
    assert_eq!(stored.len(), 1);
    assert_eq!(stored[0].seq, 1);
    assert_eq!(stored[0].payload, serde_json::json!({ "text": "hi" }));
    if let (Some(path), Some(before)) = (jsonl, before.as_ref()) {
        assert_eq!(&std::fs::read(path).unwrap(), before);
    }

    let error = log.append("thread-1", &[message("no")], None).unwrap_err();
    assert_eq!(error, ActorError::DuplicateFact { key: "m1".into() });
    let stored = log.read("thread-1").unwrap();
    assert_eq!(stored.len(), 1);
    assert_eq!(stored[0].payload, serde_json::json!({ "text": "hi" }));
    if let (Some(path), Some(before)) = (jsonl, before.as_ref()) {
        assert_eq!(&std::fs::read(path).unwrap(), before);
    }
}

fn assert_bad_thread_ids(log: &dyn LogStore, dir: Option<&Path>) {
    let long = "a".repeat(129);
    let ids = ["", long.as_str(), "a/b"];
    for id in ids {
        let append_error = log.append(id, &[message("hi")], None).unwrap_err();
        let read_error = log.read(id).unwrap_err();
        assert!(
            matches!(append_error, ActorError::BadThreadId { ref thread_id } if thread_id == id)
        );
        assert!(matches!(read_error, ActorError::BadThreadId { ref thread_id } if thread_id == id));
    }
    if let Some(dir) = dir {
        assert!(!dir.join(".jsonl").exists());
        assert!(!dir.join(format!("{long}.jsonl")).exists());
        assert!(!dir.join("a").exists());
        let jsonl = std::fs::read_dir(dir).unwrap().any(|entry| {
            entry
                .ok()
                .and_then(|entry| entry.path().extension().map(|ext| ext == "jsonl"))
                .unwrap_or(false)
        });
        assert!(!jsonl);
    }
}

#[test]
fn identical_append_is_a_noop_and_changed_payload_is_duplicate() {
    let memory = MemoryLog::new();
    assert_append_rules(&memory, None);
    assert_bad_thread_ids(&MemoryLog::new(), None);

    let dir = RemovedDir::new("append");
    let file = a3s_effect::FileLog::open(&dir.0).unwrap();
    assert_append_rules(&file, Some(&dir.0.join("thread-1.jsonl")));

    let bad_dir = RemovedDir::new("bad-thread");
    let bad = a3s_effect::FileLog::open(&bad_dir.0).unwrap();
    assert_bad_thread_ids(&bad, Some(&bad_dir.0));
}

#[test]
fn cut_log_digest_changes_with_seq_type_key_cause_and_payload() {
    let empty = cut_log(&[]);
    let empty_again = cut_log(&[]);
    assert_eq!(empty.seq, 0);
    assert_eq!(empty_again.seq, 0);
    assert_eq!(empty.digest, empty_again.digest);

    let base = fact(
        1,
        "user.message",
        "m1",
        None,
        serde_json::json!({ "text": "hi" }),
    );
    let first = cut_log(&[base.clone()]);
    let second = cut_log(&[base.clone()]);
    assert_eq!(first.digest, second.digest);
    assert_eq!(first.seq, base.seq);

    let variants = [
        fact(
            2,
            "user.message",
            "m1",
            None,
            serde_json::json!({ "text": "hi" }),
        ),
        fact(1, "note", "m1", None, serde_json::json!({ "text": "hi" })),
        fact(
            1,
            "user.message",
            "m2",
            None,
            serde_json::json!({ "text": "hi" }),
        ),
        fact(
            1,
            "user.message",
            "m1",
            Some("infer:1:0"),
            serde_json::json!({ "text": "hi" }),
        ),
        fact(
            1,
            "user.message",
            "m1",
            None,
            serde_json::json!({ "text": "no" }),
        ),
    ];
    for variant in variants {
        assert_ne!(cut_log(&[variant]).digest, first.digest);
    }
}

#[test]
fn harness_config_rejects_zero_model_attempts() {
    let error = HarnessConfig::new(1, 10, 8, 0, Vec::new(), Vec::new()).unwrap_err();
    assert_eq!(
        error,
        ActorError::Config("model_attempts must be at least 1".into())
    );
}

fn spin_actor(calls: Arc<AtomicUsize>) -> Actor<(), ()> {
    let part = component(
        || 0usize,
        |count, _fact: &Fact| count + 1,
        move |count: &usize| {
            let calls = Arc::clone(&calls);
            let n = *count;
            (
                (),
                vec![Transition {
                    key: format!("spin:{n}"),
                    run: Effect::from_async(move |_services: Arc<()>, _cancel| {
                        let calls = Arc::clone(&calls);
                        async move {
                            calls.fetch_add(1, Ordering::SeqCst);
                            Ok(vec![NewFact {
                                kind: "tick".into(),
                                key: format!("tick:{n}"),
                                payload: serde_json::json!({}),
                            }])
                        }
                    }),
                }],
            )
        },
    );
    Actor::new("spin", vec![part], |_views| ())
}

#[tokio::test]
async fn resume_step_limit_stops_before_the_next_fresh_key() {
    let calls = Arc::new(AtomicUsize::new(0));
    let actor = spin_actor(Arc::clone(&calls));
    let log = MemoryLog::new();
    let error = resume(&actor, &log, Arc::new(()), "thread-1", 1).await;
    assert!(matches!(
        error,
        Err(Exit::Fail(ActorError::StepLimit { limit: 1 }))
    ));
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    let stored = log.read("thread-1").unwrap();
    assert_eq!(stored.len(), 1);
    assert_eq!(stored[0].cause.as_deref(), Some("spin:0"));

    let calls = Arc::new(AtomicUsize::new(0));
    let actor = spin_actor(Arc::clone(&calls));
    let log = MemoryLog::new();
    let error = resume(&actor, &log, Arc::new(()), "thread-1", 0).await;
    assert!(matches!(
        error,
        Err(Exit::Fail(ActorError::StepLimit { limit: 0 }))
    ));
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    assert!(log.read("thread-1").unwrap().is_empty());
}

#[tokio::test]
async fn resume_of_a_log_with_no_transition_returns_zero_steps() {
    let part = component(
        || (),
        |state, _fact: &Fact| state,
        |_state| ((), Vec::<Transition<()>>::new()),
    );
    let actor = Actor::new("quiet", vec![part], |_views| ());
    let log = MemoryLog::new();
    let once = resume(&actor, &log, Arc::new(()), "thread-1", 1)
        .await
        .expect("limit 1");
    assert_eq!(once.steps, 0);
    let zero = resume(&actor, &log, Arc::new(()), "thread-1", 0)
        .await
        .expect("limit 0");
    assert_eq!(zero.steps, 0);
    assert!(log.read("thread-1").unwrap().is_empty());
}
