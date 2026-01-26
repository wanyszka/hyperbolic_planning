"""Visualization utilities for hyperbolic embeddings and experiments."""

from .hyperbolic_viz import (
    compute_geodesic_points,
    hyperbolic_norm,
    hyperbolic_norms,
    embed_intervals,
    plot_poincare_disk,
    test_specificity_gradient,
    test_geodesic,
    plot_all_intervals_with_geodesics,
)
from .experiment_viz import (
    setup_plotting_style,
    plot_embedding_space_2d,
    plot_norm_vs_length,
    plot_angle_vs_midpoint,
    plot_learning_curves,
    plot_success_rate_comparison,
    plot_sample_trajectories,
    plot_action_distribution_heatmap,
    create_representation_table,
    create_policy_table,
    save_all_figures,
)

__all__ = [
    # Hyperbolic visualization
    "compute_geodesic_points",
    "hyperbolic_norm",
    "hyperbolic_norms",
    "embed_intervals",
    "plot_poincare_disk",
    "test_specificity_gradient",
    "test_geodesic",
    "plot_all_intervals_with_geodesics",
    "interval_intersection",
    "interval_relationship",
    "analyze_geodesic_apex_conjecture",
    # Experiment visualization
    "setup_plotting_style",
    "plot_embedding_space_2d",
    "plot_norm_vs_length",
    "plot_angle_vs_midpoint",
    "plot_learning_curves",
    "plot_success_rate_comparison",
    "plot_sample_trajectories",
    "plot_action_distribution_heatmap",
    "create_representation_table",
    "create_policy_table",
    "save_all_figures",
]
