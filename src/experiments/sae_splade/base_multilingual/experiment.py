import logging
from itertools import product
from experimaestro import setmeta
from experimaestro.launcherfinder import find_launcher
from datamaestro_text.data.ir import DocumentStore, FileAccess
from typing import List

from xpmir.documents.samplers import RandomDocumentSampler
from xpmir.learning.learner import Learner
from xpmir.learning.batchers import PowerAdaptativeBatcher
from xpmir.learning.optim import GradientLogHook

from xpmir.models import AutoModel
from xpmir.datasets.adapters import RetrieverBasedCollection
from xpmir.papers.helpers.samplers import (
    prepare_collection,
)
from xpmir.text.adapters import TopicTextConverter
from xpmir.text.huggingface import HFModel, HFMaskedLanguageModel

from xpmir.learning.hooks import LayerFreezer
from xpmir.learning.parameters import RegexParametersIterator
from xpmir.index.sparse import SparseRetriever, SparseRetrieverIndexBuilder

from xpmir.letor.trainers.batchwise import SoftmaxCrossEntropy
from xpmir.experiments.ir import ir_experiment, IRExperimentHelper
from letor import (
    DocumentOnlySAETrainer,
    InfiniteDocumentSampler,
    DistillationInBatchNegativeTrainer,
    DistillationBatchwiseMSELoss,
    DistillationBatchwiseKLLoss,
    FullRetrieverRescorerWithFLOPs,
    ParetoFLOPIRValidationListener,
)
from dataset.documents import MultipleDocumentSampler
from dataset.samplers import (
    DistillationInBatchNegativesSampler,
)
from text.tokenizer import HFStringTokenizerColBERT
from sae import (
    SAEAdapter,
    FastTopKSAEAdapter,
    SAETokenizedEncoder,
    SAESparsityRegu,
    SAESPLADETokenizedEncoder,
    SAESPLADE,
    SAEDeadNodeLogger,
    SAEGradientAdjustHook,
    QDFlopsRegularizer,
    ColBERTSAEBiasInitialization,
    SAEGradientClippingHook,
    SAELatentAppearanceLogger,
    DualSAELatentAppearanceLogger,
    HFTokensEncoderNoType,
    HFTokensEncoderNoTypeTransform,
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
    miracl_documents,
    miracl_eval,
)
from experiments.configuration import SAESPLADEConfiguration


logging.basicConfig(level=logging.INFO)


@ir_experiment()
def run(
    helper: IRExperimentHelper,
    cfg: SAESPLADEConfiguration,
):
    # some additional params to test: Remove Later
    scaling = [True]

    # hyperparameters that won't change during the xps
    DEAD_TOKENS = 10_000_000
    GC = True  # currently the gradient norm is ok so no need GC.
    AUX_K_MULTI = 2

    # Launchers
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
    # tokenizers for colbert during the training
    def tokenizer_builder(hf_id):
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
        return doc_tokenizer_col, qry_tokenizer_col

    # build the base model
    # This one is shared across the latter SPLADE training tasks
    def text_token_encoder_builder(layer, hf_id):
        if layer == "h":
            # using the transform head
            base_colbert_encoder = HFMaskedLanguageModel.from_pretrained_id(
                hf_id,
            ).tag("model", "distil_with_head")

            # the base encoder to generate the vectors
            text_token_encoder = HFTokensEncoderNoTypeTransform.C(
                model=base_colbert_encoder,
            )
        else:
            base_colbert_encoder = HFModel.from_pretrained_id(
                hf_id,
            ).tag("model", "distil")
            text_token_encoder = HFTokensEncoderNoType.C(
                layers=layer,
                model=base_colbert_encoder,
            ).tag("layers", layer)

        return text_token_encoder

    # build the TopK SAE adapter
    def topk_adapter_builder(sae_width, k, norm_sae, text_token_encoder):
        sae_token_encoder = (
            FastTopKSAEAdapter.C(
                model=text_token_encoder,
                sae_width=sae_width,
                k=k,
                aux_k=k * AUX_K_MULTI,
                dead_steps_threshold=DEAD_TOKENS,
                normalize=norm_sae,
            )
            .tag("sae_width", sae_width)
            .tag("k", k)
            .tag("norm_sae", norm_sae)
        )
        return sae_token_encoder

    def sae_train_model(
        token_encoder: SAEAdapter,
        doc_tokenizer,
    ):
        # If SAE is trained with the query, we don't mask the punctuation
        # build the model that is used to train the SAE.
        sae_encoder = SAETokenizedEncoder.C(
            tokenizer=doc_tokenizer,
            encoder=token_encoder,
            mask_punctuation=False,
            length=256,
            override_k=0,
        )
        return sae_encoder

    def splade_train_model(
        token_encoder: SAEAdapter,
        override_k: int,
        scale: bool,
        doc_tokenizer,
        qry_tokenizer,
    ):
        splade_document_encoder = SAESPLADETokenizedEncoder.C(
            tokenizer=doc_tokenizer,
            encoder=token_encoder,  # same instance as base training
            length=256,
            mask_punctuation=False,
            override_k=override_k,
            aggregation="amax",
        )
        splade_query_encoder = SAESPLADETokenizedEncoder.C(
            tokenizer=qry_tokenizer,
            encoder=token_encoder,
            length=32,
            mask_punctuation=False,
            override_k=override_k,
            aggregation="amax",
        )
        sae_splade: SAESPLADE = (
            # If have override_k = 65536 for model with projection head need to
            # make the alpha < 1
            SAESPLADE.C(
                encoder=splade_document_encoder,
                query_encoder=splade_query_encoder,
                scale=scale,
            )
            .tag("splade_k", override_k)
            .tag("scale", scale)
        )
        return sae_splade

    def build_sae_train_hooks(rcst_hook):
        trainer_hooks = [
            SAEDeadNodeLogger.C(num_tokens=DEAD_TOKENS),
            setmeta(SAELatentAppearanceLogger.C(), True),
            SAESparsityRegu.C(coeff=0),
        ]
        trainer_hooks.append(rcst_hook)
        return trainer_hooks

    def build_splade_train_hooks(sp_hook):
        trainer_hooks = [
            setmeta(QDFlopsRegularizer.C(), True),
            SAEDeadNodeLogger.C(num_tokens=DEAD_TOKENS),
            setmeta(DualSAELatentAppearanceLogger.C(), True),
        ]
        trainer_hooks.append(sp_hook)
        return trainer_hooks

    def learn_sae(
        sae_encoder: SAETokenizedEncoder,
        trainer_hooks,
        norm_sae: bool,
        layer,  # int or str
        doc_tokenizer,
        hf_id,
    ):
        learner_hooks = [
            GradientLogHook.C(name="gradient_norm"),
            SAEGradientAdjustHook.C(),
        ]
        if GC:
            learner_hooks.insert(
                0, SAEGradientClippingHook.C(max_norm=1, use_norm=True)
            )

        if hf_id == "distilbert/distilbert-base-multilingual-cased":
            if layer == "h":
                learner_hooks.append(
                    LayerFreezer.C(
                        selector=RegexParametersIterator.C(
                            regex=r"""distilbert|vocab_""",
                            model=sae_encoder.encoder,
                        ),
                    )  # .tag("freeze_b", False)
                )
            else:
                learner_hooks.append(
                    LayerFreezer.C(
                        selector=RegexParametersIterator.C(
                            regex=r"""embeddings|transformer""",
                            model=sae_encoder.encoder,
                        ),
                    )  # .tag("freeze_b", False)
                )
        else:
            raise NotImplementedError

        doc_sampler = MultipleDocumentSampler.C(
            documents=documents_list[0],  # useless
            doc_samplers=[
                RandomDocumentSampler.C(
                    documents=documents,
                    random=random,
                )
                for documents in documents_list
            ],
        )

        encoder_trainer = DocumentOnlySAETrainer.C(
            sampler=InfiniteDocumentSampler.C(
                doc_sampler=doc_sampler,
                restore_state=False,
            ),
            batcher=PowerAdaptativeBatcher.C(),
            batch_size=cfg.learning.sae_optimization.batch_size,
            hooks=trainer_hooks,
        )

        # define the learner
        sae_learner = Learner.C(
            # Misc settings
            device=device,
            random=random,
            # How to train the model
            trainer=encoder_trainer,
            # The model to train (splade contains all the parameters)
            model=sae_encoder,
            use_fp16=True,
            init_scale=2**16,
            # Optimization settings
            steps_per_epoch=cfg.learning.sae_optimization.steps_per_epoch,
            optimizers=cfg.learning.sae_optimization.optimizer,
            max_epochs=cfg.learning.sae_optimization.max_epochs,
            # The listeners (here, for validation)
            # listeners=[validation_rerank_splade, validation_first_stage],
            listeners=[],
            # The hook used for evaluation
            hooks=learner_hooks,
        )

        learner_init_tasks = []

        if norm_sae:
            learner_init_tasks.append(
                ColBERTSAEBiasInitialization.C(
                    encoder=sae_encoder.encoder,
                    tokenizer=doc_tokenizer,
                    d_sampler=RandomDocumentSampler.C(
                        # use english
                        documents=documents_list[-1],
                        max_count=8192,
                    ),
                    batch_size=1024,
                    device=device,
                )
            )

        outputs = sae_learner.submit(
            launcher=launcher_learner,
            init_tasks=learner_init_tasks,
        )
        # helper.tensorboard_service.add(sae_learner, sae_learner.logpath)
        # output the learned result and the model config
        return outputs

    def learn_splade(
        splade: SAESPLADE,
        train_hooks,
        ir_optimizer,
        init_task,
    ):
        # no gradient clip for the moment to see what it can give
        learner_hooks = [GradientLogHook.C(name="gradient_norm")]
        if GC:
            learner_hooks.insert(
                0, SAEGradientClippingHook.C(max_norm=1, use_norm=True)
            )

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
            hooks=train_hooks,
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
            hooks=learner_hooks,
        )
        learner_init_tasks = [
            init_task,  # load model from trained SAE.
        ]

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

    # ---- The experimental plan
    # the pipeline for the topk sae models
    for (
        hf_id,
        k,
        rcst_hook,
        sae_width,
        flop_hook,
        override_k,
        norm_sae,
        layer,
        ir_optimizer,
        scale,
    ) in product(
        cfg.train_config.sae.hf_id,
        cfg.train_config.sae.ks,
        cfg.train_config.sae.rcst_hooks,
        cfg.train_config.sae.sae_width_list,
        cfg.train_config.sae_splade.flop_hooks,
        cfg.train_config.sae_splade.override_k_option,
        cfg.train_config.sae.norm_sae_input_opt,
        cfg.train_config.sae.layers,
        cfg.learning.ir_optimization.optimizer_list,
        scaling,
    ):
        if override_k > k and override_k != sae_width:
            continue

        if k == override_k:
            override_k = 0

        # manually skip some [k, splade_k] options
        tuples_joint_rm = [tuple(k) for k in cfg.train_config.sae_splade.joint_remove]
        if (k, override_k) in tuples_joint_rm:
            continue

        # the base encoder model and shared instance between
        # sae and sae_splade training
        doc_tokenizer, qry_tokenizer = tokenizer_builder(hf_id)
        text_token_encoder = text_token_encoder_builder(layer, hf_id)
        # the shared instance between two training stage
        sae_token_encoder = topk_adapter_builder(
            sae_width, k, norm_sae, text_token_encoder
        )

        # train the sae
        sae = sae_train_model(sae_token_encoder, doc_tokenizer)
        sae_train_hooks = build_sae_train_hooks(rcst_hook)
        sae_learner_output = learn_sae(
            sae, sae_train_hooks, norm_sae, layer, doc_tokenizer, hf_id
        )
        sae_trained_init_task = sae_learner_output.learned_model

        # train splade model
        sae_splade = splade_train_model(
            sae_token_encoder, override_k, scale, doc_tokenizer, qry_tokenizer
        )
        splade_train_hooks = build_splade_train_hooks(flop_hook)
        splade_learner_output = learn_splade(
            sae_splade, splade_train_hooks, ir_optimizer, sae_trained_init_task
        )

        # evaluate the trained sae_splade
        evaluation_per_model(splade_learner_output, sae_splade)

    # output the evaluation result
    for eval in eval_sets:
        eval.output_results_per_tag()
        print("\n\n")  # noqa: T201

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
