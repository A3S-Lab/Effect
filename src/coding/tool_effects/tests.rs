#![cfg(test)]

use super::*;

#[test]
fn closed_sets_refuse_unknown_values_and_round_trip() {
    assert_eq!(
        ToolExecutionOutcome::parse("failed").unwrap().as_str(),
        "failed"
    );
    assert!(ToolExecutionOutcome::parse("ok")
        .unwrap_err()
        .to_string()
        .contains("ToolExecutionOutcome"));
    assert!(ToolRecoveryPolicy::parse("retry").is_err());
    assert!(ToolObservationCompleteness::parse("partial").is_err());
    assert!(ToolFileKind::parse("link").is_err());
}

#[test]
fn complete_changes_require_byte_identity() {
    let effects = ToolEffectObservationV1::complete(vec![ToolPathChangeV1 {
        path: "hello".into(),
        before: None,
        after: ToolFileIdentityV1 {
            kind: ToolFileKind::File,
            content_digest: None,
            mode: None,
        },
    }]);
    assert!(effects.validate().is_err());
    assert!(ToolEffectObservationV1::incomplete("read failed")
        .validate()
        .is_ok());
}

fn prepared() -> ToolPreparedV1 {
    let args = serde_json::json!({"path": "target.txt"});
    ToolPreparedV1 {
        schema: TOOL_PREPARED_SCHEMA.into(),
        identity: ToolInvocationIdentityV1 {
            turn: 1,
            cycle: 0,
            tool_call_id: "call".into(),
            batch_index: 0,
            admission: Some(ToolAdmissionIdentityV1 {
                run_id: "run".into(),
                thread_id: "thread".into(),
                capability_binding_digest: format!("sha256:{}", "a".repeat(64)),
            }),
        },
        tool_name: "write".into(),
        original_args_digest: tool_args_digest(&args).unwrap(),
        effective_args_digest: tool_args_digest(&args).unwrap(),
        recovery: ToolRecoveryPolicy::NeverReplay,
        baseline: None,
    }
}

#[test]
fn version_identity_and_prepared_digest_are_required() {
    let prepared = prepared();
    let result = ToolExecutionResultV1::unknown(&prepared, "interrupted").unwrap();
    result.ensure_prepared(&prepared).unwrap();
    let mut foreign = result.clone();
    foreign.identity.admission.as_mut().unwrap().run_id = "another-run".into();
    assert!(foreign
        .ensure_prepared(&prepared)
        .unwrap_err()
        .to_string()
        .contains("tool_effect_identity_mismatch"));
    let mut altered = prepared.clone();
    altered.effective_args_digest =
        tool_args_digest(&serde_json::json!({"path":"different"})).unwrap();
    assert!(result.ensure_prepared(&altered).is_err());
    altered.schema = "future.v2".into();
    assert!(altered
        .validate()
        .unwrap_err()
        .to_string()
        .contains("tool_effect_schema_unsupported"));
    let mut invalid = prepared;
    invalid.identity.turn = 0;
    assert!(invalid.validate().is_err());
}

#[test]
fn wire_unknown_outcome_names_the_closed_set() {
    let result = ToolExecutionResultV1::unknown(&prepared(), "interrupted").unwrap();
    let mut wire = serde_json::to_value(result).unwrap();
    wire["outcome"] = serde_json::json!("future_state");
    let error = serde_json::from_value::<ToolExecutionResultV1>(wire).unwrap_err();
    assert!(error.to_string().contains("ToolExecutionOutcome"));
}

#[test]
fn observation_bound_and_missing_reason_refuse_complete_claims() {
    let missing_reason = ToolEffectObservationV1 {
        completeness: ToolObservationCompleteness::Incomplete,
        reason: None,
        changes: vec![],
        unresolved_external_operations: vec![],
    };
    assert!(missing_reason.validate().is_err());
    let mut oversized = prepared();
    oversized.baseline = Some(Value::String("x".repeat(MAX_TOOL_EFFECT_CANONICAL_BYTES)));
    assert!(oversized.validate().is_err());
    let duplicate = ToolPathChangeV1 {
        path: "removed".into(),
        before: None,
        after: ToolFileIdentityV1 {
            kind: ToolFileKind::Missing,
            content_digest: None,
            mode: None,
        },
    };
    assert!(ToolEffectObservationV1::complete(vec![duplicate.clone()])
        .validate()
        .is_ok());
    assert!(
        ToolEffectObservationV1::complete(vec![duplicate.clone(), duplicate])
            .validate()
            .is_err()
    );
}

#[test]
fn typed_paths_refuse_workspace_escape_and_preserve_legal_spelling() {
    let observation = |path: &str| {
        ToolEffectObservationV1::complete(vec![ToolPathChangeV1 {
            path: path.into(),
            before: None,
            after: ToolFileIdentityV1 {
                kind: ToolFileKind::Missing,
                content_digest: None,
                mode: None,
            },
        }])
    };
    for path in ["/outside", "../outside", "safe/../../outside", "nul\0path"] {
        assert!(
            observation(path).validate().is_err(),
            "unsafe path accepted: {path:?}"
        );
        let mut unknown = observation(path);
        unknown.completeness = ToolObservationCompleteness::Incomplete;
        unknown.reason = Some("interrupted".into());
        assert!(
            unknown.validate().is_err(),
            "Unknown must not bypass workspace scope: {path:?}"
        );
    }
    for path in [" padded ", "数据.txt", "line\nbreak", "tab\tname"] {
        let original = observation(path);
        original.validate().unwrap();
        let decoded: ToolEffectObservationV1 =
            serde_json::from_value(serde_json::to_value(original).unwrap()).unwrap();
        assert_eq!(decoded.changes[0].path, path);
    }
    #[cfg(unix)]
    for path in [r"literal\name", r".a3s\legal-file", r"C:\legal-unix-file"] {
        observation(path).validate().unwrap();
    }
    #[cfg(windows)]
    for path in [r"C:\outside", r"\\server\share\outside", r"safe\..\outside"] {
        assert!(observation(path).validate().is_err());
    }
}
