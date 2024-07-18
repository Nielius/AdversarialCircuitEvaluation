"""Generate LaTeX tables of circuit outputs for use in the workshop paper.

The input is basically a BruteForceExperimentOutputAnalysisV1."""

import sys
from pathlib import Path

import jsonpickle
import pandas as pd

from acdc.nudb.adv_opt.analysis.analyzer_brute_force_v1 import BruteForceExperimentOutputAnalysisV1
from acdc.nudb.adv_opt.data_fetchers import AdvOptTaskName


def load_df(path: Path, version: int) -> pd.DataFrame:
    def load_brute_force_analysis_v1(path: Path) -> BruteForceExperimentOutputAnalysisV1:
        decoded_obj = jsonpickle.loads(path.read_text())
        match decoded_obj:
            case [BruteForceExperimentOutputAnalysisV1() as analysis]:
                return analysis
            case BruteForceExperimentOutputAnalysisV1() as analysis:
                return analysis
            case _:
                raise ValueError("Could not decode the object")

    def assign_combined_output_and_logit_columns(df: pd.DataFrame):
        return df.assign(
            **df[["most_likely_output_model", "logits_model"]]
            .apply(lambda x: list(zip(*x)), axis=1)
            .apply(pd.Series)
            .rename(columns=lambda idx: f"model {idx}")
        ).assign(
            **df[["most_likely_output_circuit", "logits_circuit"]]
            .apply(lambda x: list(zip(*x)), axis=1)
            .apply(pd.Series)
            .rename(columns=lambda idx: f"circuit {idx}")
        )

    analysis = load_brute_force_analysis_v1(path)
    return (
        pd.DataFrame(analysis.top_k_worst_inputs)
        .assign(
            loss=lambda df: df.loss * (50_257 if version == 1 else 1)
        )  # * 50_257 is because of an error in the original code
        .pipe(assign_combined_output_and_logit_columns)
    )


def format_with_nested_tabular(t: tuple[str, float]) -> str:
    # See https://tex.stackexchange.com/questions/40561/table-with-multiple-lines-in-some-cells for alternatives
    # (want multi-line within one cell)
    return "\\begin{tblr}{@{}c@{}}" + repr(t[0]) + "\\\\ " + f"({t[1]:.2f})" + "\\end{tblr}"


def generate_latex_table(df: pd.DataFrame, task_name: AdvOptTaskName) -> str:
    if task_name == AdvOptTaskName.DOCSTRING:

        def input_formatter(s: str) -> str:
            return "\\texttt{" + s.replace("\n", " \\newlinesymbol\n") + "}"
    else:

        def input_formatter(s: str) -> str:
            return s

    return (
        (
            df.rename({"patch_input": "patch input"}, axis="columns")
            .style.format(
                formatter=format_with_nested_tabular,
                # lambda x: f"{x[0].__repr__()}\\newline({x[1]:.2f})",
                subset=["circuit 0", "circuit 1", "circuit 2", "model 0", "model 1", "model 2"],
            )
            .format(subset=["loss"], formatter="{:.2f}".format)
            .format(subset=["input", "patch input"], formatter=input_formatter)
            .hide(
                subset=["most_likely_output_model", "logits_model", "most_likely_output_circuit", "logits_circuit"],
                axis=1,
            )
            .hide(axis="index")
            .to_latex(column_format="X[l] X[l] c | c c c | c c c ", hrules=True)
        )
        .replace("\\begin{tabular}", "\\begin{tblr}")
        .replace("\\end{tabular}", "\\end{tblr}")
        .replace("<|BOS|>", "\\bossymbol")
    )


if __name__ == "__main__":
    version = 1 if len(sys.argv) < 2 else int(sys.argv[1])

    if version == 1:
        base_dir = Path("/home/niels/data/advopt/processed/2024-05-20-brute-force-loss-and-output-analysis")
        output_base_path = Path("/home/niels")

        for task_name in [AdvOptTaskName.IOI, AdvOptTaskName.DOCSTRING, AdvOptTaskName.GREATERTHAN]:
            analysis_path = base_dir / str(task_name) / "analyses.json"
            output_path = output_base_path / f"{task_name.value}-outputtable.tex"

            latex_str = generate_latex_table(load_df(analysis_path, version=version), task_name)
            output_path.write_text(latex_str)

    elif version == 2:
        print("Running version 2, with the normal input matched to the patched input.")

        base_dir = Path(
            "/home/niels/data/advopt/processed/2024-07-11-brute-force-loss-and-output-analysis-matched-corruptions"
        )
        output_base_path = Path("/home/niels")

        for task_name in [AdvOptTaskName.IOI, AdvOptTaskName.DOCSTRING, AdvOptTaskName.GREATERTHAN]:
            analysis_path = base_dir / str(task_name.value) / "analysis.json"
            output_path = output_base_path / f"{task_name.value}-outputtable-matched-corruptions.tex"

            latex_str = generate_latex_table(load_df(analysis_path, version=version), task_name)
            output_path.write_text(latex_str)
            print("Wrote to ", output_path)
