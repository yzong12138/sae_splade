from typing import List, Dict, Tuple
from pathlib import Path
from xpmir.evaluation import Evaluations, EvaluationsCollection
from datamaestro_text.data.ir import (
    DocumentStore,
    TopicsStore,
    Adhoc,
    FileAccess,
)
from xpmir.papers.helpers.samplers import (
    prepare_collection,
    msmarco_v1_tests,
    MEASURES,
)
from xpmir.datasets.adapters import RandomFold, MemoryTopicStore
from dataset.adapters import MMSubTopicAdhoc, MMSubCollectionAdhoc
from dataset.samplers import DistillationListwiseSampler
from dataset.distillation import (
    JSONBasedBatchDistillationSamples,
    MultiLingualListwiseHydrator,
    ListwiseHydrator,
)
import logging


def msmarco_colbert_distillation_samples(path, nway) -> DistillationListwiseSampler:
    """Distillation samples from ColBERTv2 training."""

    # Access to topic text
    train_topics = prepare_collection("irds.msmarco-passage.train.queries")

    # Combine the training triplets with the document and queries texts
    distillation_samples = ListwiseHydrator.C(
        samples=JSONBasedBatchDistillationSamples.C(
            id="colbertv2.distillation",
            path=Path(path),
            nway_num=nway,
        ),
        documentstore=prepare_collection("irds.msmarco-passage.documents"),
        querystore=MemoryTopicStore.C(topics=train_topics),
    )

    # Generate a sampler from the samples
    return DistillationListwiseSampler.C(samples=distillation_samples)


def msmarco_lotte_evaluation_sets() -> List[EvaluationsCollection]:
    return [
        msmarco_v1_tests(only_judged=True),
        EvaluationsCollection(
            LoT_Wrt=Evaluations(
                prepare_collection("irds.lotte.writing.test.search"), MEASURES
            ),
        ),
        EvaluationsCollection(
            LoT_Rcr=Evaluations(
                prepare_collection("irds.lotte.recreation.test.search"), MEASURES
            ),
        ),
        EvaluationsCollection(
            LoT_Sci=Evaluations(
                prepare_collection("irds.lotte.science.test.search"), MEASURES
            ),
        ),
        EvaluationsCollection(
            LoT_Tch=Evaluations(
                prepare_collection("irds.lotte.technology.test.search"), MEASURES
            ),
        ),
        EvaluationsCollection(
            LoT_LS=Evaluations(
                prepare_collection("irds.lotte.lifestyle.test.search"), MEASURES
            ),
        ),
    ]


def msmarco_lotte_evalutation_documents() -> List[DocumentStore]:
    doc_list: List[DocumentStore] = [
        prepare_collection("irds.msmarco-passage.documents"),
        prepare_collection("irds.lotte.writing.test.documents").tag("ds", "LoT_Wrt"),
        prepare_collection("irds.lotte.recreation.test.documents").tag("ds", "LoT_Rcr"),
        prepare_collection("irds.lotte.science.test.documents").tag("ds", "LoT_Sci"),
        prepare_collection("irds.lotte.technology.test.documents").tag("ds", "LoT_Tch"),
        prepare_collection("irds.lotte.lifestyle.test.documents").tag("ds", "LoT_LS"),
    ]
    for d in doc_list:
        d.file_access = FileAccess.MEMORY
    return doc_list


def msmarco_beir_evaluation_sets() -> List[EvaluationsCollection]:
    return [
        msmarco_v1_tests(only_judged=True),
        EvaluationsCollection(
            ArguAna=Evaluations(prepare_collection("irds.beir.arguana"), MEASURES)
        ),
        EvaluationsCollection(
            Climate_FEVER=Evaluations(
                prepare_collection("irds.beir.climate-fever"), MEASURES
            )
        ),
        EvaluationsCollection(
            DBPedia=Evaluations(
                prepare_collection("irds.beir.dbpedia-entity.test"), MEASURES
            )
        ),
        EvaluationsCollection(
            FEVER=Evaluations(prepare_collection("irds.beir.fever.test"), MEASURES)
        ),
        EvaluationsCollection(
            FiQA=Evaluations(prepare_collection("irds.beir.fiqa.test"), MEASURES)
        ),
        EvaluationsCollection(
            HotpotQA=Evaluations(
                prepare_collection("irds.beir.hotpotqa.test"), MEASURES
            )
        ),
        EvaluationsCollection(
            NFCorpus=Evaluations(
                prepare_collection("irds.beir.nfcorpus.test"), MEASURES
            )
        ),
        EvaluationsCollection(
            NQ=Evaluations(prepare_collection("irds.beir.nq"), MEASURES)
        ),
        EvaluationsCollection(
            Quora=Evaluations(prepare_collection("irds.beir.quora.test"), MEASURES)
        ),
        EvaluationsCollection(
            SCIDOCS=Evaluations(prepare_collection("irds.beir.scidocs"), MEASURES)
        ),
        EvaluationsCollection(
            SciFact=Evaluations(prepare_collection("irds.beir.scifact.test"), MEASURES)
        ),
        EvaluationsCollection(
            TREC_COVID=Evaluations(prepare_collection("irds.beir.trec-covid"), MEASURES)
        ),
        EvaluationsCollection(
            Touche2020_v2=Evaluations(
                prepare_collection("irds.beir.webis-touche2020.v2"), MEASURES
            )
        ),
    ]


def msmarco_beir_evaluation_documents() -> List[DocumentStore]:
    doc_list: List[DocumentStore] = [
        prepare_collection("irds.msmarco-passage.documents"),
        prepare_collection("irds.beir.arguana.documents").tag("ds", "ArguAna"),
        prepare_collection("irds.beir.climate-fever.documents").tag(
            "ds", "Climate_FEVER"
        ),
        prepare_collection("irds.beir.dbpedia-entity.documents").tag("ds", "DBPedia"),
        prepare_collection("irds.beir.fever.documents").tag("ds", "FEVER"),
        prepare_collection("irds.beir.fiqa.documents").tag("ds", "FiQA"),
        prepare_collection("irds.beir.hotpotqa.documents").tag("ds", "HotpotQA"),
        prepare_collection("irds.beir.nfcorpus.documents").tag("ds", "NFCorpus"),
        prepare_collection("irds.beir.nq.documents").tag("ds", "NQ"),
        prepare_collection("irds.beir.quora.documents").tag("ds", "Quora"),
        prepare_collection("irds.beir.scidocs.documents").tag("ds", "SCIDOCS"),
        prepare_collection("irds.beir.scifact.documents").tag("ds", "SciFact"),
        prepare_collection("irds.beir.trec-covid.documents").tag("ds", "TREC_COVID"),
        prepare_collection("irds.beir.webis-touche2020.v2.documents").tag(
            "ds", "Touche2020_v2"
        ),
    ]
    for d in doc_list:
        d.file_access = FileAccess.MEMORY
    return doc_list


# --- Multilingual
# Language order: other language (order by alphabetic) + english

# ------ MMARCO
def mmarco_documents(langs: List[str]) -> List[DocumentStore]:
    doc_list: List[DocumentStore] = [
        prepare_collection(f"irds.mmarco.v2.{lang}.documents") for lang in langs
    ]
    doc_list.append(prepare_collection("irds.msmarco-passage.documents"))
    for d in doc_list:
        d.file_access = FileAccess.MEMORY
    return doc_list


def mmarco_train_queries(langs: List[str]) -> List[TopicsStore]:
    topic_list: List[TopicsStore] = [
        prepare_collection(f"irds.mmarco.v2.{lang}.train.queries") for lang in langs
    ]
    topic_list.append(prepare_collection("irds.msmarco-passage.train.queries"))
    return topic_list


def mmarco_dev(langs: List[str]) -> List[Adhoc]:
    adhoc: List[TopicsStore] = [
        prepare_collection(f"irds.mmarco.v2.{lang}.dev") for lang in langs
    ]
    adhoc.append(prepare_collection("irds.msmarco-passage.dev.judged"))
    return adhoc


def mmarco_dev_small(langs: List[str]) -> List[Adhoc]:
    adhoc: List[TopicsStore] = [
        prepare_collection(f"irds.mmarco.v2.{lang}.dev.small") for lang in langs
    ]
    adhoc.append(prepare_collection("irds.msmarco-passage.dev.small"))
    return adhoc


def mmarco_eval(langs: List[str]) -> List[EvaluationsCollection]:
    eval_list = [
        {
            f"mm_{lang}": Evaluations(
                prepare_collection(f"irds.mmarco.v2.{lang}.dev.small").tag(
                    "ds", f"mm_{lang}"
                ),
                MEASURES,
            )
        }
        for lang in langs
    ]
    eval_collections = [EvaluationsCollection(**eval) for eval in eval_list]
    eval_collections.append(msmarco_v1_tests(only_judged=True))
    return eval_collections


# ---------- MMARCO training samples
def mmarco_colbert_distillation_samples(
    path, nway, lang, docstores: List[DocumentStore]
) -> DistillationListwiseSampler:
    """Distillation samples from ColBERTv2 training."""

    # Access to topic text
    train_topics_list: List[TopicsStore] = mmarco_train_queries(lang)
    # Combine the training triplets with the document and queries texts
    distillation_samples = MultiLingualListwiseHydrator.C(
        samples=JSONBasedBatchDistillationSamples.C(
            id="colbertv2.distillation",
            path=Path(path),
            nway_num=nway,
        ),
        documentstore=docstores,
        querystore=[
            MemoryTopicStore.C(topics=train_topics)
            for train_topics in train_topics_list
        ],
    )

    # Generate a sampler from the samples
    return DistillationListwiseSampler.C(samples=distillation_samples)


# ---------- MMARCO for validation
def mmarco_validation_fulldoc(
    cfg,
    en_dev: Adhoc,
    en_dev_small: Adhoc,
    mm_devs: List[Adhoc],  # no english here.
    launcher=None,
):
    randfold_english = RandomFold.C(
        dataset=en_dev,
        seed=cfg.seed,
        fold=0,
        sizes=[cfg.size],
        exclude=en_dev_small.topics,
    ).submit(launcher=launcher)

    ds = [
        MMSubTopicAdhoc.C(
            eng_adhoc=randfold_english,
            mm_adhoc=mm_dev,
        ).submit(launcher=launcher)
        for mm_dev in mm_devs
    ]
    ds.append(randfold_english)
    return ds


def mmarco_validation_subdoc(
    eng_adhoc: Adhoc,  # output of the retrieverbasedcollection
    mm_devs: List[Adhoc],  # no english here.
    launcher=None,
):
    ds = [
        MMSubCollectionAdhoc.C(
            eng_sub_docs=eng_adhoc.documents,
            mm_adhoc=mm_dev,
        ).submit(launcher=launcher)
        for mm_dev in mm_devs
    ]
    for adhoc in ds:
        adhoc.documents.in_memory = True
    ds.append(eng_adhoc)
    return ds


# ------ MIRACL
def miracl_documents(langs: List[str]) -> List[DocumentStore]:
    # langs.append("en")
    doc_list: List[DocumentStore] = [
        prepare_collection(f"irds.miracl.{lang}.documents") for lang in langs
    ]
    for d in doc_list:
        d.file_access = FileAccess.MEMORY
    return doc_list


def miracl_eval(langs: List[str]) -> List[EvaluationsCollection]:
    # langs.append("en")
    eval_list = [
        {
            f"mcl_{lang}": Evaluations(
                prepare_collection(f"irds.miracl.{lang}.dev").tag("ds", f"mcl_{lang}"),
                MEASURES,
            )
        }
        for lang in langs
    ]
    eval_collections = [EvaluationsCollection(**eval) for eval in eval_list]
    return eval_collections


# evaluation result post process
def calculate_dataset_mean(
    eval_sets,
    mean_set_dict: Dict[
        str, Tuple[List, str]
    ],  # dict: {set_name: ([eval_set_name], metric_name), ...}
    tag_names: List[str],  # list [k, splade_k, ...]
):
    """Calculate and print the mean of the datasets in eval_sets
    each mean_set only report one metric

    All the dataset should be ready, or it will be skipped.

    The sampled tags must be distiguishable for the trained models,
    or it will be buggy
    """
    # store the data
    set_data_dict = {set_name: [] for set_name in mean_set_dict.keys()}
    set_subset_name_dict = {set_name: [] for set_name in mean_set_dict.keys()}
    tag_names_list = [("tag", t) for t in tag_names]
    for eval in eval_sets:
        for sub_ds_name, evaluations in eval.collection.items():
            for set_name, (subset_name_list, metric) in mean_set_dict.items():
                if sub_ds_name in subset_name_list:
                    data = evaluations.to_dataframe()  # the raw data
                    try:
                        # get the tags, and metric we want
                        data = data[tag_names_list + [("metric", metric)]]
                    except KeyError:
                        # means the data is empty
                        # or the part of key doesn't exist
                        continue
                    # rename the metric key to the sub-ds name
                    data = data.rename(columns={metric: sub_ds_name})
                    # for later process and logging
                    set_data_dict[set_name].append(data)
                    set_subset_name_dict[set_name].append(sub_ds_name)

    # get the mean for each
    for set_name, set_data_list in set_data_dict.items():
        if len(set_data_dict[set_name]) == 0:
            logging.warning(f"No eval of Set {set_name} are ready, skip")
            continue

        # outer merge
        df_merged = set_data_list[0]
        for i in range(1, len(set_data_list)):
            df_merged = df_merged.merge(
                set_data_list[i],
                how="outer",
                on=tag_names_list,
            )

        # calculate the mean
        mean_index = [("metric", sub_ds) for sub_ds in set_subset_name_dict[set_name]]
        assert len(mean_index) == len(set_data_list)
        df_merged[mean_index] = df_merged[mean_index].astype(float)
        mean_values = df_merged[mean_index].mean(axis=1)
        df_merged[("metric", f"{set_name} mean")] = mean_values

        # print it out!
        metric_name = mean_set_dict[set_name][1]  # get the original metric name
        print(f"## Mean {metric_name} for Dataset {set_name}: ")  # noqa: T201
        print(f"Subset consist of {set_subset_name_dict[set_name]}")  # noqa: T201
        print(df_merged.to_markdown())  # noqa: T201
        print("\n\n")  # noqa: T201
