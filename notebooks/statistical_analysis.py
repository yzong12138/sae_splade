from scipy import stats
from pathlib import Path

P_THRESHOLD = 0.05

ROOT = Path("/lustre/fswork/projects/rech/nku/upf44ba/statistical_test_jean_zay")

baseline_paths_list = [
    ROOT / "splade_baseline" / "ms_lotte.txt",
    ROOT / "splade_baseline" / "ms_beir.txt",
    ROOT / "splade_baseline" / "ms_lotte.txt",
    ROOT / "splade_baseline" / "ms_beir.txt",
    ROOT / "splade_baseline" / "ms_lotte.txt",
    ROOT / "splade_baseline" / "ms_beir.txt",
    ROOT / "multilingual" / "splade" / "mm_mcl.txt",
]
our_paths_list = [
    ROOT / "sae_splade" / "k_ab" / "k=4" / "ms_lotte.txt",
    ROOT / "sae_splade" / "k_ab" / "k=4" / "ms_beir.txt",
    ROOT / "sae_splade" / "k_ab" / "k=8" / "ms_lotte.txt",
    ROOT / "sae_splade" / "k_ab" / "k=8" / "ms_beir.txt",
    ROOT / "sae_splade" / "k_ab" / "k=16" / "ms_lotte.txt",
    ROOT / "sae_splade" / "k_ab" / "k=16" / "ms_beir.txt",
    ROOT / "multilingual" / "sae_splade" / "mm_mcl.txt",
]
ms_lotte_metric = ["RR@10"] + ["nDCG@10"] * 2 + ["Success@5"] * 5
ms_beir_metric = ["RR@10"] + ["nDCG@10"] * 15
mm_mcl_metric = ["RR@10"] * 7 + ["nDCG@10"] * 8

ms_lotte_names = [
    "dev-small",
    "DL19",
    "DL20",
    "LoT_Wrt",
    "LoT_Rcr",
    "LoT_Sci",
    "LoT_Tch",
    "LoT_LS",
]
ms_beir_names = [
    "dev-small",
    "DL19",
    "DL20",
    "ArguAna",
    "Climate_FEVER",
    "DBPedia",
    "FEVER",
    "FiQA",
    "HotpotQA",
    "NFCorpus",
    "NQ",
    "Quora",
    "SCIDOCS",
    "SciFact",
    "TREC_COVID",
    "Touche2020_v2",
]
mm_mcl_names = [
    "mm_ar",
    "mm_es",
    "mm_fr",
    "mm_ja",
    "mm_ru",
    "mm_zh",
    "msmarco_dev",
    "DL19",
    "DL20",
    "mcl_ar",
    "mcl_es",
    "mcl_fr",
    "mcl_ja",
    "mcl_ru",
    "mcl_zh",
]

metrics_list = [
    ms_lotte_metric,
    ms_beir_metric,
    ms_lotte_metric,
    ms_beir_metric,
    ms_lotte_metric,
    ms_beir_metric,
    mm_mcl_metric,
]
ds_names = [
    ms_lotte_names,
    ms_beir_names,
    ms_lotte_names,
    ms_beir_names,
    ms_lotte_names,
    ms_beir_names,
    mm_mcl_names,
]

for baseline_paths, our_paths, metrics, ds_name in zip(
    baseline_paths_list, our_paths_list, metrics_list, ds_names
):
    with baseline_paths.open() as f1, our_paths.open() as f2:
        for baseline_path, our_path, metric, ds in zip(f1, f2, metrics, ds_name):

            baseline_path = baseline_path.strip()
            our_path = our_path.strip()

            baseline_results = {}
            try:
                with open(baseline_path, "r") as base:
                    for line in base:
                        detailed_res = line.split()
                        if detailed_res[0] == metric:
                            baseline_results[detailed_res[1]] = detailed_res[2]
            except (OSError, FileNotFoundError):
                print(f"Fail to open file {baseline_path}, continue.")  # noqa: T201
                continue

            our_results = {}
            with open(our_path) as our:
                for line in our:
                    detailed_res = line.split()
                    if detailed_res[0] == metric:
                        our_results[detailed_res[1]] = detailed_res[2]

            assert set(baseline_results.keys()) == set(
                our_results.keys()
            ), "baseline and ours result should have the same set of ids."

            baseline_scores = []
            our_scores = []
            for qid, score in baseline_results.items():
                baseline_scores.append(float(score))
                our_scores.append(float(our_results[qid]))

            assert len(baseline_scores) == len(our_scores)
            size = len(baseline_scores)
            t_stat, p_value = stats.ttest_rel(our_scores, baseline_scores)
            print(  # noqa: T201
                f"p_value for the statistical test on {ds} is {p_value}"
            )
            if p_value < P_THRESHOLD and sum(baseline_scores) < sum(our_scores):
                print("\t We are significant better.")  # noqa: T201
            elif p_value < P_THRESHOLD and sum(baseline_scores) > sum(our_scores):
                print("\t We are significant worse.")  # noqa: T201
        print("\n\n\n")  # noqa: T201
