"""Copy the main result figures from outputs/ into figures/ under descriptive names (used by README.md)."""
import shutil
from pathlib import Path

FIGURES = [
    ("outputs/tess_vetted_period/fig1_tess_examples.png", "01_what_the_cnn_looks_at.png"),
    ("outputs/tess_vetted_period/fig3_tess_confusion.png", "02_confusion_matrix.png"),
    ("outputs/tess_vetted_period/fig7_tess_significance.png", "03_explanations_vs_chance.png"),
    ("outputs/tess_vetted_period/fig2_tess_deletion.png", "04_deletion_test.png"),
    ("outputs/tess_vetted_period/fig17_tess_period_split.png", "05_shape_vs_period.png"),
    ("outputs/fixes_vetted/fig13_fix_shortcut.png", "06_hidden_shortcut_and_fix.png"),
    ("outputs/benchmark/fig19_benchmark.png", "07_benchmark.png"),
    ("outputs/tess_vetted_period/fig10_tess_never_seen.png", "08_never_seen_star_types.png"),
    ("outputs/tess_vetted_period/fig11_tess_detection_limit.png", "09_detection_limit.png"),
    ("outputs/tess_vetted_period/fig6_tess_misclassified.png", "10_misclassified_stars.png"),
    ("outputs/benchmark/fig20_gao_check.png", "11_catalogue_check_gao2025.png"),
    ("outputs/fig3_shortcut.png", "12_synthetic_cheating_test.png"),
]


def main():
    out = Path("figures")
    out.mkdir(exist_ok=True)
    for src, dst in FIGURES:
        if Path(src).exists():
            shutil.copyfile(src, out / dst)
            print(f"{src} -> figures/{dst}")
        else:
            print(f"missing (run the script that makes it first): {src}")


if __name__ == "__main__":
    main()
