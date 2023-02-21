from pathlib import Path

import pandas as pd
import seaborn as sns

sns.set_style("whitegrid")

results = pd.read_csv("results/eval2.csv")
results = results.query("Environment in ['4_1', '9_4', '28_3']")

T_sensitivity_analysis = pd.read_csv("results/eval3.csv")

sdc_stc = results.query("Agent in ['SDC', 'STC']")
sdc_spc = results.query("Agent in ['SDC', 'Shortest Path']")
sdc_stc_spc = results.query("Agent in ['SDC', 'STC', 'Shortest Path']")


def _plot_metric(metric, unit, comparison, comparison_name):
    g = sns.catplot(
        data=comparison,
        kind="bar",
        x="Environment",
        y=metric,
        hue="Agent",
        palette="Set1",
        alpha=0.8,
        height=5,
    )
    g.despine(left=True)
    g.set_axis_labels("", f"{metric} ({unit})")
    g.legend.set_title("")
    fig_path = Path(f"results/figures/{comparison_name}")
    if not fig_path.exists():
        fig_path.mkdir(parents=True)
    g.figure.savefig(str(fig_path / f"{metric.lower()}.png"))


_plot_metric("Winrate", "%", T_sensitivity_analysis, "T_sensitivity_analysis")
_plot_metric("Number of Interactions", "n", T_sensitivity_analysis, "T_sensitivity_analysis")
_plot_metric("Number of Time Steps", "n", T_sensitivity_analysis, "T_sensitivity_analysis")
