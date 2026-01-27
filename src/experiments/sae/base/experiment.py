import logging
from itertools import product
from experimaestro import setmeta
from experimaestro.launcherfinder import find_launcher
from datamaestro_text.data.ir import DocumentStore, FileAccess

from xpmir.documents.samplers import RandomDocumentSampler
from xpmir.learning.learner import Learner
from xpmir.learning.batchers import PowerAdaptativeBatcher
from xpmir.learning.optim import GradientLogHook

from xpmir.papers.helpers.samplers import (
    prepare_collection,
)
from xpmir.text.adapters import TopicTextConverter
from xpmir.text.huggingface import HFModel, HFMaskedLanguageModel

from xpmir.learning.hooks import LayerFreezer
from xpmir.learning.parameters import RegexParametersIterator

from xpmir.experiments.ir import ir_experiment, IRExperimentHelper
from letor import (
    DocumentOnlySAETrainer,
    InfiniteDocumentSampler,
)
from dataset.documents import DocumentQuerySampler
from text.tokenizer import HFStringTokenizerColBERT
from sae import (
    SAEAdapter,
    FastTopKSAEAdapter,
    SAESparsityRegu,
    SAEDeadNodeLogger,
    SAEGradientAdjustHook,
    ColBERTSAEBiasInitialization,
    SAEGradientClippingHook,
    SAELatentAppearanceLogger,
    SAETokenizedEncoder,
    HFTokensEncoderNoType,
    HFTokensEncoderNoTypeTransform,
)
from experiments.configuration import SAESPLADEConfiguration


logging.basicConfig(level=logging.INFO)


@ir_experiment()
def run(
    helper: IRExperimentHelper,
    cfg: SAESPLADEConfiguration,
):
    # Hyperparameters won't change during the xp.
    DEAD_TOKENS = 10_000_000
    GC = True  # currently the gradient norm is ok so no need GC.
    AUX_K_MULTI = 2

    # Launchers
    launcher_learner = find_launcher(cfg.learning.requirements)

    # misc
    device = cfg.device
    random = cfg.random

    # for training
    documents: DocumentStore = prepare_collection("irds.msmarco-passage.documents")
    documents.file_access = FileAccess.MEMORY
    train_queries = prepare_collection("irds.msmarco-passage.train.queries")

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
        return doc_tokenizer_col

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

    def model_builder(token_encoder: SAEAdapter, doc_tokenizer):
        # If SAE is trained with the query, we don't mask the punctuation
        sae_encoder = SAETokenizedEncoder.C(
            tokenizer=doc_tokenizer,
            encoder=token_encoder,
            mask_punctuation=False,
            length=256,
            override_k=0,
        )
        return sae_encoder

    def building_trainer_hooks(rcst_hook):
        trainer_hooks = [
            SAEDeadNodeLogger.C(num_tokens=DEAD_TOKENS),
            setmeta(SAELatentAppearanceLogger.C(), True),
            SAESparsityRegu.C(coeff=0),
        ]
        trainer_hooks.append(rcst_hook)
        return trainer_hooks

    def learn(
        sae_encoder: SAETokenizedEncoder,
        trainer_hooks,
        norm_sae: bool,
        sae_train_query: bool,
        layer,
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

        if hf_id == "distilbert/distilbert-base-uncased":
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
        elif hf_id == "google-bert/bert-base-uncased":
            if layer == "h":
                learner_hooks.append(
                    LayerFreezer.C(
                        selector=RegexParametersIterator.C(
                            regex=r"""bert|cls""",
                            model=sae_encoder.encoder,
                        ),
                    )  # .tag("freeze_b", False)
                )
            else:
                learner_hooks.append(
                    LayerFreezer.C(
                        selector=RegexParametersIterator.C(
                            regex=r"""encoder|embeddings""",
                            model=sae_encoder.encoder,
                        ),
                    )  # .tag("freeze_b", False)
                )

        doc_sampler = RandomDocumentSampler.C(
            documents=documents,
            random=random,
        )
        if sae_train_query:
            doc_sampler = DocumentQuerySampler.C(
                doc_sampler=doc_sampler,
                topics=train_queries,
                random=random,
                documents=documents,
            )

        encoder_trainer = DocumentOnlySAETrainer.C(
            sampler=InfiniteDocumentSampler.C(
                doc_sampler=doc_sampler,
            ),
            batcher=PowerAdaptativeBatcher.C(),
            batch_size=cfg.learning.sae_optimization.batch_size,
            hooks=trainer_hooks,
        ).tag("train_qry", sae_train_query)

        # define the learner
        learner = Learner.C(
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
                        documents=documents,
                        max_count=8192,
                    ),
                    batch_size=1024,
                    device=device,
                )
            )

        outputs = learner.submit(
            launcher=launcher_learner,
            init_tasks=learner_init_tasks,
        )
        helper.tensorboard_service.add(learner, learner.logpath)
        # output the learned result and the model config
        return outputs

    # the pipeline for the jumprelu models
    for (hf_id, k, rcst_hook, sae_width, norm_sae, sae_train_query, layer) in product(
        cfg.train_config.sae.hf_id,
        cfg.train_config.sae.ks,
        cfg.train_config.sae.rcst_hooks,
        cfg.train_config.sae.sae_width_list,
        cfg.train_config.sae.norm_sae_input_opt,
        cfg.train_config.sae.sae_train_with_query_opt,
        cfg.train_config.sae.layers,
    ):
        doc_tokenizer = tokenizer_builder(hf_id)
        text_token_encoder = text_token_encoder_builder(layer, hf_id)
        sae_token_encoder = topk_adapter_builder(
            sae_width, k, norm_sae, text_token_encoder
        )
        sae_colbert = model_builder(sae_token_encoder, doc_tokenizer)
        trainer_hooks = building_trainer_hooks(rcst_hook)
        learn(
            sae_colbert,
            trainer_hooks,
            norm_sae,
            sae_train_query,
            layer,
            doc_tokenizer,
            hf_id,
        )
