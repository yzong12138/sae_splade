import logging
from itertools import product
from experimaestro import setmeta
from experimaestro.launcherfinder import find_launcher
from datamaestro_text.data.ir import DocumentStore, FileAccess

from xpmir.learning.learner import Learner
from xpmir.learning.batchers import PowerAdaptativeBatcher
from xpmir.learning.optim import GradientLogHook

from xpmir.models import AutoModel
from xpmir.datasets.adapters import RetrieverBasedCollection
from xpmir.papers.helpers.samplers import (
    prepare_collection,
    msmarco_v1_validation_dataset,
)
from xpmir.text.adapters import TopicTextConverter
from xpmir.text.huggingface import HFMaskedLanguageModel, HFModel
from xpmir.index.sparse import SparseRetriever, SparseRetrieverIndexBuilder

from xpmir.letor.trainers.batchwise import SoftmaxCrossEntropy
from xpmir.experiments.ir import ir_experiment, IRExperimentHelper
from dataset.samplers import (
    DistillationInBatchNegativesSampler,
)
from letor import (
    DistillationInBatchNegativeTrainer,
    DistillationBatchwiseMSELoss,
    DistillationBatchwiseKLLoss,
    FullRetrieverRescorerWithFLOPs,
    ParetoFLOPIRValidationListener,
)
from text.tokenizer import HFStringTokenizerColBERT
from sae import (
    SAEAdapter,
    FastTopKSAEAdapter,
    SAESPLADETokenizedEncoder,
    SAESPLADE,
    SAEDeadNodeLogger,
    QDFlopsRegularizer,
    SAEGradientClippingHook,
    DualSAELatentAppearanceLogger,
    HFTokensEncoderNoType,
    HFTokensEncoderNoTypeTransform,
)
from utils.datasets import (
    msmarco_lotte_evaluation_sets,
    msmarco_lotte_evalutation_documents,
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
    # Some additional parameters we want to test -- For the final run only keep one
    scaling = [True]

    # Hyperparameters won't change during the xp.
    DEAD_TOKENS = 10_000_000
    GC = True  # currently the gradient norm is ok so no need GC.

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

    spladev2, splade_init_tasks = AutoModel.load_from_hf_hub("xpmir/SPLADE_DistilMSE")

    # for evaluations
    eval_sets = msmarco_lotte_evaluation_sets()
    eval_docs = msmarco_lotte_evalutation_documents()

    # build the splade index and retriever for validation
    baseline_sparse_index = SparseRetrieverIndexBuilder.C(
        batch_size=512,
        batcher=PowerAdaptativeBatcher.C(),
        encoder=spladev2.encoder,
        device=device,
        documents=documents,
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

    # the validation dataset for training splade
    ds_val_full = RetrieverBasedCollection.C(
        dataset=msmarco_v1_validation_dataset(
            cfg.validation, launcher=launcher_preprocessing, only_judged=True
        ),
        retrievers=[validation_splade_retriever],
    ).submit(launcher=launcher_learner, init_tasks=splade_init_tasks)
    ds_val_full.documents.in_memory = True

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

    # A base model. k and aux_k is not important.
    # will be override later.
    def topk_adapter_builder(sae_width, text_token_encoder):
        sae_token_encoder = FastTopKSAEAdapter.C(
            model=text_token_encoder,
            sae_width=sae_width,
            k=8,  # no influence as no SAE training
            aux_k=16,  # no influence as no SAE training
            dead_steps_threshold=DEAD_TOKENS,
            normalize=False,
        ).tag("sae_width", sae_width)
        return sae_token_encoder

    def splade_train_model(
        token_encoder: SAEAdapter,
        override_k: int,
        scale: bool,
        layer,
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
        if layer == "h" and override_k > 30000:
            # if using transform head with a large width, using a smaller alpha
            # to avoid fp16 overflow
            alpha = 0.8
        else:
            alpha = 1
        sae_splade: SAESPLADE = (
            SAESPLADE.C(
                encoder=splade_document_encoder,
                query_encoder=splade_query_encoder,
                scale=scale,
                scale_init=alpha,
            )
            .tag("splade_k", override_k)
            .tag("scale", scale)
            .tag("s_init", alpha)
        )
        return sae_splade

    def build_splade_train_hooks(sp_hook):
        trainer_hooks = [
            setmeta(QDFlopsRegularizer.C(), True),
            SAEDeadNodeLogger.C(num_tokens=DEAD_TOKENS),
            setmeta(DualSAELatentAppearanceLogger.C(), True),
        ]
        trainer_hooks.append(sp_hook)
        return trainer_hooks

    def learn_splade(splade: SAESPLADE, ir_optimizer, train_hooks):
        # no gradient clip for the moment to see what it can give
        learner_hooks = [GradientLogHook.C(name="gradient_norm")]
        if GC:
            learner_hooks.insert(
                0, SAEGradientClippingHook.C(max_norm=1, use_norm=True)
            )

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
            hooks=train_hooks,
            need_ibn=False,
            need_distil=True,
        )

        # splade_trainer = DistillationPairwiseTrainer.C(
        #     batch_size=cfg.learning.ir_optimization.batch_size,
        #     sampler=msmarco_hofstaetter_ensemble_hard_negatives(),
        #     lossfn=MSEDifferenceLoss.C(),
        #     hooks=train_hooks,
        # )

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
            hooks=learner_hooks,
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

    # ---- The experimental plan
    # the pipeline for the topk sae models
    for (
        hf_id,
        sae_width,
        flop_hook,
        override_k,
        layer,
        ir_optimizer,
        scale,
    ) in product(
        cfg.train_config.sae.hf_id,
        cfg.train_config.sae.sae_width_list,
        cfg.train_config.sae_splade.flop_hooks,
        cfg.train_config.sae_splade.override_k_option,
        cfg.train_config.sae.layers,
        cfg.learning.ir_optimization.optimizer_list,
        scaling,
    ):
        assert override_k > 0, "override_k = 0 is invalid in one stage training"

        doc_tokenizer, qry_tokenizer = tokenizer_builder(hf_id)

        # the base encoder model
        text_token_encoder = text_token_encoder_builder(layer, hf_id)
        sae_token_encoder = topk_adapter_builder(sae_width, text_token_encoder)

        # train splade model
        sae_splade = splade_train_model(
            sae_token_encoder, override_k, scale, layer, doc_tokenizer, qry_tokenizer
        )
        splade_train_hooks = build_splade_train_hooks(flop_hook)
        splade_learner_output = learn_splade(
            sae_splade,
            ir_optimizer,
            splade_train_hooks,
        )

        # evaluate the trained sae_splade
        evaluation_per_model(splade_learner_output, sae_splade)

    # output the evaluation result
    for eval in eval_sets:
        eval.output_results_per_tag()
        print("\n\n")  # noqa: T201

    # output the mean
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
