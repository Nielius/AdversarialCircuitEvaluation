"""Generate LaTeX tables of circuit outputs for use in the workshop paper."""

from itertools import groupby
from pathlib import Path

import pandas as pd
import torch

from acdc.nudb.adv_opt.data_fetchers import AdvOptTaskName


def load_all_distribution_data() -> dict[str, torch.Tensor]:
    suffix = Path("artifacts/metrics_canonical.pt")
    metric_paths = {
        AdvOptTaskName.IOI: Path(
            "/home/niels/data/advopt/raw/tidy/2024-03-02-bruteforce-v1/2024-03-02-011541_bruteforce_ioi_1" / suffix
        ),
        AdvOptTaskName.GREATERTHAN: Path(
            "/home/niels/data/advopt/raw/tidy/2024-03-02-bruteforce-v1/2024-03-02-081117_bruteforce_greaterthan_1"
            / suffix
        ),
        AdvOptTaskName.DOCSTRING: Path(
            "/home/niels/data/advopt/raw/tidy/2024-03-02-bruteforce-v1/2024-03-02-130221_bruteforce_docstring_1"
            / suffix
        ),
    }

    return {task.value: torch.load(path, map_location="cpu") * 50_257 for task, path in metric_paths.items()}


if __name__ == "__main__":

    def add_columns_with_z_scores(percentiles: pd.DataFrame) -> pd.DataFrame:
        def column_as_std_deviations_from_mean(column: pd.Series) -> pd.Series:
            return (column.loc["min":"max"] - column.loc["mean"]) / column.loc["std"]

        def sort_columns_alphabetically(df: pd.DataFrame) -> pd.DataFrame:
            return df.reindex(sorted(df.columns), axis=1)

        def group_columns_inplace(df: pd.DataFrame) -> None:
            column_groups = [key for key, _ in groupby(df.columns, key=lambda x: x.split("_")[0])]
            multi_index = pd.MultiIndex.from_product([column_groups, ["abs", "z-score"]])
            df.columns = multi_index

        percentiles_with_std = percentiles.assign(
            **{
                f"{column_name}_std": column_as_std_deviations_from_mean(percentiles[column_name])
                for column_name in ["ioi", "greaterthan", "docstring"]
            }
        ).pipe(sort_columns_alphabetically)
        group_columns_inplace(percentiles_with_std)

        return percentiles_with_std

    percentiles = pd.DataFrame(load_all_distribution_data()).describe(
        percentiles=[0.25, 0.5, 0.75, 0.95, 0.99, 0.999, 0.9999]
    )
    percentiles_with_std = add_columns_with_z_scores(percentiles)

    # Format as LaTeX

    latex_str = (
        percentiles_with_std.style.format("{:.2f}", na_rep="")
        # .format_index(axis=1, formatter=format_column_name_for_latex)
        .format_index(axis=1, formatter=lambda x: f"\\centering {x}")  # center column labels
        .format_index(axis=0, escape="latex")
        .to_latex(multicol_align="c", hrules=True)
        .replace("\\begin{tabular}", "\\begin{tblr}")
        .replace("\\end{tabular}", "\\end{tblr}")
    )

    # Replace the second line, because pandas doesn't support tabularray, and tabularray doesn't support multicolumn (https://github.com/lvjr/tabularray/issues/28)
    lines = latex_str.split("\n")
    for i, line in enumerate(lines):
        if "\\multicolumn" in line:
            lines[
                i
            ] = "   & \SetCell[c=2]{c} docstring  & & \SetCell[c=2]{c} greaterthan & & \SetCell[c=2]{c}{ioi} \\\\"
    latex_str = "\n".join(lines)

    # Store results

    Path("/home/niels/proj/acdc/nudb/paper/test-latex/testtable.tex").write_text(latex_str)
    print(latex_str)
