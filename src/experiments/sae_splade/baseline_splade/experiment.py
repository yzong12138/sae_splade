import logging
from itertools import product
from experimaestro import setmeta
from experimaestro.launcherfinder import find_launcher
from datamaestro_text.data.ir import DocumentStore, FileAccess
from xpmir.learning.optim import GradientLogHook, GradientClippingHook

from xpmir.datasets.adapters import RetrieverBasedCollection
from xpmir.papers.helpers.samplers import (
    prepare_collection,
    msmarco_v1_validation_dataset,
)
from xpmir.models import AutoModel
from xpmir.text.adapters import TopicTextConverter
from xpmir.index.sparse import SparseRetriever, SparseRetrieverIndexBuilder
from xpmir.letor.trainers.batchwise import SoftmaxCrossEntropy
from xpmir.learning.learner import Learner
from xpmir.text.huggingface import (
    HFMaskedLanguageModel,
)
from xpmir.learning.batchers import PowerAdaptativeBatcher
from xpmir.experiments.ir import ir_experiment, IRExperimentHelper
from sae import (
    DualSAELatentAppearanceLogger,
    QDFlopsRegularizer,
    SPLADEEncoder,
    SAESPLADE,
)
from dataset.samplers import (
    DistillationInBatchNegativesSampler,
)
from text.tokenizer import HFStringTokenizerColBERT
from letor import (
    FullRetrieverRescorerWithFLOPs,
    ParetoFLOPIRValidationListener,
    DistillationInBatchNegativeTrainer,
    DistillationBatchwiseMSELoss,
    DistillationBatchwiseKLLoss,
)
from utils.datasets import (
    msmarco_lotte_evaluation_sets,
    msmarco_lotte_evalutation_documents,
    msmarco_beir_evaluation_sets,
    msmarco_beir_evaluation_documents,
    calculate_dataset_mean,
    msmarco_colbert_distillation_samples,
)
from experiments.configuration import SAESPLADEConfiguration

logging.basicConfig(level=logging.INFO)


@ir_experiment()
def run(
    helper: IRExperimentHelper,
    cfg: SAESPLADEConfiguration,
):
    # some additional params to test
    scaling = [True]

    # launchers
    launcher_learner = find_launcher(cfg.learning.requirements)
    launcher_preprocessing = find_launcher(cfg.preprocessing.requirements)
    launcher_evaluate = find_launcher(cfg.retrieval.requirements)
    launcher_gpu_indexing = find_launcher(cfg.indexation.requirements)

    # misc
    device = cfg.device
    random = cfg.random

    # for training
    documents: DocumentStore = prepare_collection("irds.msmarco-passage.documents")
    documents.file_access = FileAccess.MEMORY

    spladev2_base, splade_init_tasks = AutoModel.load_from_hf_hub(
        "xpmir/SPLADE_DistilMSE"
    )

    # for evaluations
    if cfg.retrieval.eval_on_beir:
        eval_sets = msmarco_beir_evaluation_sets()
        eval_docs = msmarco_beir_evaluation_documents()
    else:
        eval_sets = msmarco_lotte_evaluation_sets()
        eval_docs = msmarco_lotte_evalutation_documents()

    # build the splade index and retriever for validation
    baseline_sparse_index = SparseRetrieverIndexBuilder.C(
        batch_size=512,
        batcher=PowerAdaptativeBatcher.C(),
        encoder=spladev2_base.encoder,
        device=device,
        documents=documents,
        ordered_index=False,
        max_docs=cfg.indexation.max_docs,
    ).submit(launcher=launcher_learner, init_tasks=splade_init_tasks)

    validation_splade_retriever = SparseRetriever.C(
        index=baseline_sparse_index,
        topk=cfg.learning.validation_top_k,
        batchsize=1,
        encoder=spladev2_base._query_encoder,
        in_memory=True,
    )

    # the validation dataset for training splade
    ds_val_full = RetrieverBasedCollection.C(
        dataset=msmarco_v1_validation_dataset(
            cfg.validation, launcher=launcher_preprocessing, only_judged=True
        ),
        retrievers=[validation_splade_retriever],
    ).submit(launcher=launcher_learner, init_tasks=splade_init_tasks)
    ds_val_full.documents.in_memory = True

    # build the tokenizers
    # Use the same type of tokenizers with the other type of experiments
    def tokenizer_model_builder(hf_id):
        converter = TopicTextConverter.C()
        doc_tokenizer_col = HFStringTokenizerColBERT.from_pretrained_id(
            hf_id,
            query=False,
            num_add_tokens=1,
            converter=converter,
        )
        qry_tokenizer_col = HFStringTokenizerColBERT.from_pretrained_id(
            hf_id,
            query=True,
            converter=converter,
        )

        # the base encoder from huggingface
        base_encoder = HFMaskedLanguageModel.from_pretrained_id(hf_id).tag(
            "model", "distil_with_head"
        )
        return doc_tokenizer_col, qry_tokenizer_col, base_encoder

    def build_splade_model(
        k,
        scale,
        doc_tokenizer,
        qry_tokenizer,
        base_encoder,
        query_length=32,
    ):
        # We truncate the query also to the size of 200
        splade_document_encoder = SPLADEEncoder.C(
            tokenizer=doc_tokenizer,
            encoder=base_encoder,
            length=256,
            mask_punctuation=False,
            override_k=k,
        )
        splade_query_encoder = SPLADEEncoder.C(
            tokenizer=qry_tokenizer,
            encoder=base_encoder,
            length=query_length,
            mask_punctuation=False,
            override_k=k,
        )
        # we don't scale the tradtional splade model
        splade = (
            SAESPLADE.C(
                encoder=splade_document_encoder,
                query_encoder=splade_query_encoder,
                scale=scale,
            )
            .tag("splade_k", k)
            .tag("scale", scale)
        )
        return splade

    def train_splade(splade, ir_optimizer, sp_hook):
        # hooks
        train_hook = [
            sp_hook,
            setmeta(DualSAELatentAppearanceLogger.C(), True),
            setmeta(QDFlopsRegularizer.C(), True),
        ]
        learn_hooks = [
            GradientLogHook.C(name="gradient_norm"),
            GradientClippingHook.C(max_norm=1),
        ]

        splade_validation_full = ParetoFLOPIRValidationListener.C(
            id="splade_full_validation_splade",
            dataset=ds_val_full,
            retriever=FullRetrieverRescorerWithFLOPs.C(
                documents=ds_val_full.documents,
                scorer=splade,
                batchsize=cfg.retrieval.batch_size,
                batcher=PowerAdaptativeBatcher.C(),
            ),
            validation_interval=cfg.learning.validation_interval,
            metrics={"RR@10": True},
        )

        # trainer
        splade_trainer = DistillationInBatchNegativeTrainer.C(
            sampler=DistillationInBatchNegativesSampler.C(
                sampler=msmarco_colbert_distillation_samples(
                    path=cfg.train_config.sae_splade.distil_data_path,
                    nway=cfg.train_config.sae_splade.nway,
                )
            ),
            batcher=PowerAdaptativeBatcher.C(),
            batch_size=cfg.learning.ir_optimization.batch_size,
            lossfn=SoftmaxCrossEntropy.C(),
            lossfn_distillation=[
                DistillationBatchwiseKLLoss.C(weight=1),
                DistillationBatchwiseMSELoss.C(weight=0.05),
            ],
            hooks=train_hook,
            need_ibn=False,
            need_distil=True,
        )

        # define the learner
        splade_learner = Learner.C(
            # Misc settings
            device=device,
            random=random,
            # How to train the model
            trainer=splade_trainer,
            # The model to train (splade contains all the parameters)
            model=splade,
            use_fp16=True,
            # Optimization settings
            steps_per_epoch=cfg.learning.ir_optimization.steps_per_epoch,
            optimizers=ir_optimizer,
            max_epochs=cfg.learning.ir_optimization.max_epochs,
            # The listeners (here, for validation)
            # listeners=[validation_rerank_splade, validation_first_stage],
            listeners=[splade_validation_full],
            # The hook used for evaluation
            hooks=learn_hooks,
        )
        learner_init_tasks = []
        outputs = splade_learner.submit(
            launcher=launcher_learner,
            init_tasks=learner_init_tasks,
        )
        helper.tensorboard_service.add(splade_learner, splade_learner.logpath)
        return outputs

    def evaluation_per_model(outputs, model_config, long_model_config=None):
        loaded_models = {
            # "best_RR@10": outputs.listeners["splade_full_validation_splade"][
            #     "RR@10"
            # ].tag("cp", "RR@10"),
            "last": outputs.learned_model.tag("cp", "last"),
        }

        for _, loaded_model in loaded_models.items():
            for doc, eval in zip(eval_docs, eval_sets):
                sparse_index = SparseRetrieverIndexBuilder.C(
                    batch_size=1024,
                    batcher=PowerAdaptativeBatcher.C(),
                    encoder=model_config.encoder,
                    device=device,
                    documents=doc,
                    ordered_index=False,
                    max_docs=cfg.indexation.max_docs,
                ).submit(launcher=launcher_gpu_indexing, init_tasks=[loaded_model])
                if list(eval.collection.keys())[0] in ["ArguAna", "Climate_FEVER"]:
                    splade_retriever = SparseRetriever.C(
                        index=sparse_index,
                        topk=1000,
                        batchsize=64,
                        encoder=long_model_config._query_encoder,
                    )
                else:
                    splade_retriever = SparseRetriever.C(
                        index=sparse_index,
                        topk=1000,
                        batchsize=64,
                        encoder=model_config._query_encoder,
                    )
                # evaluate model
                eval.evaluate_retriever(
                    splade_retriever,
                    launcher_evaluate,
                    model_id=None,
                    init_tasks=[loaded_model],
                )

    for hf_id, sp_hook, splade_k, ir_optimizer, scale in product(
        cfg.train_config.sae.hf_id,
        cfg.train_config.sae_splade.flop_hooks,
        cfg.train_config.sae_splade.override_k_option,
        cfg.learning.ir_optimization.optimizer_list,
        scaling,
    ):
        assert (
            splade_k > 0 or splade_k == -1
        ), "splade = 0 is invalid in one stage training"
        doc_tokenizer, qry_tokenizer, base_encoder = tokenizer_model_builder(hf_id)
        splade_model = build_splade_model(
            splade_k, scale, doc_tokenizer, qry_tokenizer, base_encoder
        )
        splade_output = train_splade(splade_model, ir_optimizer, sp_hook)
        # for the arguana evaluation, use a larger query length
        # only the query length is different, others are the same
        if cfg.retrieval.eval_on_beir:
            sae_splade_arguana = build_splade_model(
                splade_k,
                scale,
                doc_tokenizer,
                qry_tokenizer,
                base_encoder,
                query_length=256,
            )
        else:
            sae_splade_arguana = None

        evaluation_per_model(splade_output, splade_model, sae_splade_arguana)

    # output the evaluation result
    for eval in eval_sets:
        eval.output_results_per_tag()
        print("\n\n")  # noqa: T201

    # output the mean
    if cfg.retrieval.eval_on_beir:
        calculate_dataset_mean(
            eval_sets=eval_sets,
            mean_set_dict={
                "dev_small": (["msmarco_dev"], "RR@10"),
                "TREC-DL": (["trec2019", "trec2020"], "nDCG@10"),
                "BEIR": (
                    [
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
                    ],
                    "nDCG@10",
                ),
            },
            tag_names=cfg.logging,
        )
    else:
        calculate_dataset_mean(
            eval_sets=eval_sets,
            mean_set_dict={
                "dev_small": (["msmarco_dev"], "RR@10"),
                "TREC-DL": (["trec2019", "trec2020"], "nDCG@10"),
                "LoTTE": (
                    ["LoT_Wrt", "LoT_Rcr", "LoT_Sci", "LoT_Tch", "LoT_LS"],
                    "Success@5",
                ),
            },
            tag_names=cfg.logging,
        )
