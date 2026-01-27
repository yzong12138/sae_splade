import logging
from typing import List
from itertools import product
from experimaestro import setmeta
from experimaestro.launcherfinder import find_launcher
from datamaestro_text.data.ir import DocumentStore, FileAccess
from xpmir.learning.optim import GradientLogHook, GradientClippingHook

from xpmir.datasets.adapters import RetrieverBasedCollection
from xpmir.papers.helpers.samplers import (
    prepare_collection,
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
    calculate_dataset_mean,
    mmarco_documents,
    mmarco_eval,
    mmarco_dev_small,
    mmarco_dev,
    mmarco_validation_fulldoc,
    mmarco_validation_subdoc,
    mmarco_colbert_distillation_samples,
    miracl_eval,
    miracl_documents,
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
    ml_langs = cfg.train_config.ml_langs

    # doc_store
    documents_list: List[DocumentStore] = mmarco_documents(ml_langs)

    spladev2, splade_init_tasks = AutoModel.load_from_hf_hub("xpmir/SPLADE_DistilMSE")

    # for evaluations
    if cfg.retrieval.eval_on_mcl:
        eval_sets = mmarco_eval(ml_langs) + miracl_eval(ml_langs)
        eval_docs = documents_list + miracl_documents(ml_langs)
    else:
        eval_sets = mmarco_eval(ml_langs)
        eval_docs = documents_list

    # eng documents
    eng_documents = prepare_collection("irds.msmarco-passage.documents")
    eng_documents.file_access = FileAccess.MEMORY

    # build the splade index and retriever for validation
    baseline_sparse_index = SparseRetrieverIndexBuilder.C(
        batch_size=512,
        batcher=PowerAdaptativeBatcher.C(),
        encoder=spladev2.encoder,
        device=device,
        documents=eng_documents,
        ordered_index=False,
        max_docs=cfg.indexation.max_docs,
    ).submit(launcher=launcher_learner, init_tasks=splade_init_tasks)

    validation_splade_retriever = SparseRetriever.C(
        index=baseline_sparse_index,
        topk=cfg.learning.validation_top_k,
        batchsize=1,
        encoder=spladev2._query_encoder,
        in_memory=True,
    )

    # for validation
    mm_dev_small = mmarco_dev_small(ml_langs)
    mm_dev = mmarco_dev(ml_langs)
    full_validation = mmarco_validation_fulldoc(
        cfg.validation,
        mm_dev[-1],
        mm_dev_small[-1],
        mm_dev[:-1],
        launcher=launcher_preprocessing,
    )
    eng_val_sub = RetrieverBasedCollection.C(
        dataset=full_validation[-1],
        retrievers=[validation_splade_retriever],
    ).submit(launcher=launcher_learner, init_tasks=splade_init_tasks)
    validation_ds_list = mmarco_validation_subdoc(
        eng_adhoc=eng_val_sub,
        mm_devs=full_validation[:-1],
        launcher=launcher_preprocessing,
    )

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

    def build_splade_model(k, scale, doc_tokenizer, qry_tokenizer, base_encoder):
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
            length=32,
            mask_punctuation=False,
            override_k=k,
        )
        # we don't scale the tradtional splade model
        if k == -1:
            # if using transform head with a large width, using a smaller alpha
            # to avoid fp16 overflow
            alpha = 0.8
        else:
            alpha = 1
        splade = (
            SAESPLADE.C(
                encoder=splade_document_encoder,
                query_encoder=splade_query_encoder,
                scale=scale,
                scale_init=alpha,
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

        splade_trainer = DistillationInBatchNegativeTrainer.C(
            sampler=DistillationInBatchNegativesSampler.C(
                sampler=mmarco_colbert_distillation_samples(
                    path=cfg.train_config.sae_splade.distil_data_path,
                    nway=cfg.train_config.sae_splade.nway,
                    lang=ml_langs,
                    docstores=documents_list,
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

        splade_validation_list = [
            ParetoFLOPIRValidationListener.C(
                id=f"mm_validation_{lang}",
                dataset=validation_ds,
                retriever=FullRetrieverRescorerWithFLOPs.C(
                    documents=validation_ds.documents,
                    scorer=splade,
                    batchsize=cfg.retrieval.batch_size,
                    batcher=PowerAdaptativeBatcher.C(),
                ),
                validation_interval=cfg.learning.validation_interval,
                metrics={"RR@10": True},
            )
            for validation_ds, lang in zip(validation_ds_list, ml_langs + ["en"])
        ]

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
            listeners=splade_validation_list,
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

    def evaluation_per_model(outputs, model_config):
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
        evaluation_per_model(splade_output, splade_model)

    # output the evaluation result
    for eval in eval_sets:
        eval.output_results_per_tag()
        print("\n\n")  # noqa: T201

    # output the mean
    if cfg.retrieval.eval_on_mcl:
        calculate_dataset_mean(
            eval_sets=eval_sets,
            mean_set_dict={
                "TREC-DL": (["trec2019", "trec2020"], "nDCG@10"),
                "mm_lang": (
                    [f"mm_{lang}" for lang in ml_langs] + ["msmarco_dev"],
                    "RR@10",
                ),
                "mcl_lang": ([f"mcl_{lang}" for lang in ml_langs], "nDCG@10"),
            },
            tag_names=cfg.logging,
        )
    else:
        calculate_dataset_mean(
            eval_sets=eval_sets,
            mean_set_dict={
                "TREC-DL": (["trec2019", "trec2020"], "nDCG@10"),
                "mm_lang": (
                    [
                        "mm_ar",
                        "mm_es",
                        "mm_fr",
                        "mm_ja",
                        "mm_ru",
                        "mm_zh",
                        "msmarco_dev",
                    ],
                    "RR@10",
                ),
            },
            tag_names=cfg.logging,
        )
