"""Phase: exploratory data analysis. Saves statistics to outputs/metrics and figures to outputs/figures/eda.

Usage (from the project root):  python scripts/run_eda.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.paths import use_project_matplotlib_cache  # noqa: E402

use_project_matplotlib_cache(ROOT)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.data import eda  # noqa: E402
from src.data.load import load_raw  # noqa: E402
from src.evaluation import plots  # noqa: E402
from src.utils.io import load_config, save_json  # noqa: E402

SELECTED_FEATURES = ["V17", "V14", "V12", "V10", "V16", "V3"]  # largest |SMD|; for illustration only


def main() -> None:
    cfg = load_config(ROOT)
    df = load_raw(ROOT, cfg)
    metrics = ROOT / cfg["paths"]["metrics_dir"]
    figs = ROOT / cfg["paths"]["figures_dir"] / "eda"

    stats = eda.compute_eda_stats(df)
    save_json(stats, metrics / "eda_stats.json")
    eda.feature_summary(df).to_csv(metrics / "eda_feature_summary.csv")
    comparison = eda.class_comparison(df)
    comparison.to_csv(metrics / "eda_class_comparison.csv")

    plots.apply_style()
    plots.plot_class_counts(df, figs / "01_class_counts.png")
    plots.plot_class_share(df, figs / "02_class_distribution.png")
    plots.plot_amount_distribution(df, figs / "03_amount_distribution.png")
    plots.plot_time_distribution(df, figs / "04_time_distribution.png")
    plots.plot_feature_scales(df, figs / "05_feature_scales.png")
    plots.plot_feature_grid(df, figs / "06_feature_distributions_all.png")
    plots.plot_selected_features(df, SELECTED_FEATURES, figs / "07_selected_features.png")
    plots.plot_correlation_heatmaps(df, figs / "08_correlation_heatmaps.png")
    plots.plot_class_comparison(comparison, figs / "09_normal_vs_fraud_comparison.png")
    plt.close("all")

    cb = stats["class_balance"]
    print(f"rows={stats['overview']['rows']:,} normal={cb['normal']:,} fraud={cb['fraud']:,} fraud%={cb['fraud_pct']:.4f}")
    print(f"saved stats to {metrics.relative_to(ROOT)} and figures to {figs.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
