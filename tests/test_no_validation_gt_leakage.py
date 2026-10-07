from pathlib import Path


REPO = Path(__file__).resolve().parents[1]


def test_canonical_evaluator_has_no_validation_label_memory():
    evaluator = (REPO / "experiments" / "evaluate_fixed_end_to_end.py").read_text(
        encoding="utf-8"
    )
    launcher = (REPO / "experiments" / "run_fixed_end_to_end.sh").read_text(
        encoding="utf-8"
    )

    banned = (
        "partial_validation_memory_weights",
        "selected_targets",
        "apply_released_memory",
        "IMAGE_MEMORY_ALPHA",
        "MULTIMODAL_MEMORY_ALPHA",
    )
    for token in banned:
        assert token not in evaluator
        assert token not in launcher

    # learner-night creates /workspace/continual during container startup.
    assert '-v "$repo_root:/workspace"' in launcher
    assert '-v "$repo_root:/workspace:ro"' not in launcher

    assert not (REPO / "experiments" / "evaluate_partial_validation_memory.py").exists()
    assert not (
        REPO
        / "experiments"
        / "image_only"
        / "release_20261006"
        / "partial_validation_memory_weights.npz"
    ).exists()


def test_targets_are_loaded_after_both_prediction_tracks_are_final():
    source = (REPO / "experiments" / "evaluate_fixed_end_to_end.py").read_text(
        encoding="utf-8"
    )
    targets_position = source.index("targets = data_utils.multihot(validation, vocab)")
    image_position = source.index("image_scores =")
    multimodal_position = source.index("multimodal_scores, multimodal_vocab =")

    assert image_position < targets_position
    assert multimodal_position < targets_position
    assert source.index("apply_overlap_knn(") < targets_position


def test_overlap_bank_is_training_artifact_not_evaluation_target_input():
    builder = (
        REPO / "experiments" / "image_only" / "build_overlap_knn_bank.py"
    ).read_text(encoding="utf-8")
    evaluator = (REPO / "experiments" / "evaluate_fixed_end_to_end.py").read_text(
        encoding="utf-8"
    )

    assert "original_train_records" in builder
    assert "selected_validation_records" in builder
    assert 'bank["labels"]' in evaluator
    assert "targets[overlap" not in evaluator
    assert "validation[overlap" not in evaluator


def test_retracted_scores_are_not_reproducibility_gates():
    source = (REPO / "experiments" / "evaluate_fixed_end_to_end.py").read_text(
        encoding="utf-8"
    )

    assert "0.8367713093757629" not in source
    assert "0.8511210680007935" not in source
    assert "validation_labels_used_for\": \"metric_only" in source


def test_archived_result_is_explicitly_retracted():
    archived = (
        REPO
        / "experiments"
        / "image_only"
        / "release_20261006"
        / "results.json"
    ).read_text(encoding="utf-8")

    assert '"status": "retracted_validation_gt_leakage"' in archived
    assert '"full_validation_f1_at_5"' not in archived
    assert '"invalid_target_corrected_full_validation_f1_at_5"' in archived
