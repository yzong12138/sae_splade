from dataclasses import InitVar

from experimaestro import Param

from transformers.models.bert import BertForMaskedLM, BertModel
from transformers.models.distilbert import DistilBertForMaskedLM, DistilBertModel

from xpmir.text.huggingface import HFModel, HFMaskedLanguageModel
from xpmir.text.huggingface.encoders import HFTokensEncoder
from xpmir.utils.utils import easylog
from xpmir.text.encoders import (
    TokenizedTexts,
    TokensRepresentationOutput,
)
from sae.huggingface import (
    ColBERTModel,
    ColBERTConfig,
    HFColBERTConfigFromId,
    DistilColBERTModel,
    DistilColBERTConfig,
    TransformersColBERTOutput,
)

logger = easylog()


# base model
class ColBERTEncoder(HFModel):
    model: InitVar[ColBERTModel]
    automodel = ColBERTModel
    autoconfig = ColBERTConfig

    @classmethod
    def from_pretrained_id(cls, model_id: str, out_dim: int):
        return cls.C(config=HFColBERTConfigFromId.C(model_id=model_id, out_dim=out_dim))


class ColBERTEncoderScratch(ColBERTEncoder):
    model: InitVar[DistilColBERTModel]
    automodel = DistilColBERTModel
    autoconfig = DistilColBERTConfig


# Used for the DistilBERT SAE with no transform of MLM head
class HFTokensEncoderNoType(HFTokensEncoder):
    """The huggingface Token Encoder for the models that doesn't support token
    type id, with additional layer selection mechanism
    """

    layers: Param[int] = -1
    """The number of layers for the Base model,
    -1 means no layers truncation
    """

    def __initialize__(self, options):
        super().__initialize__(options)
        assert isinstance(self.model.model, DistilBertModel) or isinstance(
            self.model.model, BertModel
        )
        if isinstance(self.model.model, DistilBertModel) and self.layers >= 0:
            assert self.layers <= self.model.model.transformer.n_layers
            self.model.model.transformer.layer = self.model.model.transformer.layer[
                : self.layers
            ]
        elif isinstance(self.model.model, BertModel) and self.layers >= 0:
            assert self.layers <= len(self.model.model.encoder.layer)
            self.model.model.encoder.layer = self.model.model.encoder.layer[
                : self.layers
            ]
        else:
            logger.info("Use all model layers for base encoder")

    def forward(self, tokenized: TokenizedTexts) -> TokensRepresentationOutput:
        tokenized = tokenized.to(self.model.contextual_model.device)
        y = self.model.contextual_model(
            tokenized.ids,
            attention_mask=tokenized.mask.to(self.device),
        )
        return TokensRepresentationOutput(
            tokenized=tokenized, value=y.last_hidden_state
        )


# Used for the DistilBERT SAE with no transform of MLM head
class HFTokensEncoderNoTypeTransform(HFTokensEncoder):
    """A model that does return the last embeddings of the model before applying
    the vocabulary projection"""

    model: Param[HFMaskedLanguageModel]
    """A Hugging-Face model that does the MLM"""

    def __initialize__(self, options):
        super().__initialize__(options)

    def forward(self, tokenized: TokenizedTexts) -> TokensRepresentationOutput:
        tokenized = tokenized.to(self.model.contextual_model.device)
        if isinstance(self.model.model, DistilBertForMaskedLM):
            hidden_state = self.model.model.distilbert(
                tokenized.ids,
                attention_mask=tokenized.mask.to(self.device),
            ).last_hidden_state  # shape [bs, length, hs]
            hidden_state = self.model.model.vocab_transform(hidden_state)
            hidden_state = self.model.model.activation(hidden_state)
            hidden_state = self.model.model.vocab_layer_norm(hidden_state)
        elif isinstance(self.model.model, BertForMaskedLM):
            hidden_state = self.model.model.bert(
                tokenized.ids,
                attention_mask=tokenized.mask.to(self.device),
            ).last_hidden_state  # shape [bs, length, hs]
            hidden_state = self.model.model.cls.predictions.transform(hidden_state)
        else:
            raise NotImplementedError

        return TokensRepresentationOutput(
            tokenized=tokenized,
            value=hidden_state,
        )


# cannot use the HFTokensEncoder directly as the distilbert has no token_type_ids
# the xpmir colbert encoder, without any additional function on top
class HFColBERTTokensEncoder(HFTokensEncoder):
    """The colbert base encoder
    the last hiddens state is normalized
    """

    model: Param[ColBERTEncoder]
    """The colbert model/encoder"""

    def __initialize__(self, options):
        super().__initialize__(options)

    def forward(self, tokenized: TokenizedTexts) -> TokensRepresentationOutput:
        tokenized = tokenized.to(self.model.contextual_model.device)
        if isinstance(self.model, ColBERTEncoderScratch):
            # distilbert doesn't have the token type ids
            y: TransformersColBERTOutput = self.model.contextual_model(
                tokenized.ids,
                attention_mask=tokenized.mask.to(self.device),
            )
        else:
            y: TransformersColBERTOutput = self.model.contextual_model(
                tokenized.ids,
                attention_mask=tokenized.mask.to(self.device),
                token_type_ids=tokenized.token_type_ids,
            )
        return TokensRepresentationOutput(
            tokenized=tokenized,
            value=y.last_hidden_state,
        )
