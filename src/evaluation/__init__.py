"""Evaluation infrastructure: metrics and policy evaluation."""

from .metrics import (
    RepresentationMetrics,
    PolicyMetrics,
    compute_hyperbolic_distance,
    compute_hyperbolic_norm,
    compute_temporal_geometric_alignment,
    compute_norm_length_correlation,
    compute_angle_midpoint_correlation,
    compute_embedding_statistics,
    compute_all_representation_metrics,
    compute_action_entropy,
    compute_policy_metrics,
)
from .evaluator import (
    EpisodeResult,
    EvaluationResult,
    evaluate_policy,
    run_episode,
    evaluate_generalization,
    compute_generalization_summary,
    run_full_evaluation,
    compare_policies,
    save_evaluation_results,
    load_evaluation_results,
    print_comparison_table,
)

__all__ = [
    # Metrics dataclasses
    "RepresentationMetrics",
    "PolicyMetrics",
    # Metric computations
    "compute_hyperbolic_distance",
    "compute_hyperbolic_norm",
    "compute_temporal_geometric_alignment",
    "compute_norm_length_correlation",
    "compute_angle_midpoint_correlation",
    "compute_containment_auroc",
    "compute_embedding_statistics",
    "compute_all_representation_metrics",
    "compute_action_entropy",
    "compute_policy_metrics",
    # Evaluation
    "EpisodeResult",
    "EvaluationResult",
    "evaluate_policy",
    "run_episode",
    "evaluate_generalization",
    "compute_generalization_summary",
    "run_full_evaluation",
    "compare_policies",
    "save_evaluation_results",
    "load_evaluation_results",
    "print_comparison_table",
]
