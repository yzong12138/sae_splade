from typing import Optional, Union, List
import torch
import string
import torch.nn as nn
from attrs import define
from abc import ABC, abstractmethod
from experimaestro import Param, Meta, LightweightTask

from xpmir.learning.optim import ModuleInitMode
from xpmir.letor import Device, DeviceInformation

from xpmir.text import TokenizerOptions
from xpmir.text.encoders import (
    TokenizedTexts,
    TextsRepresentationOutput,
    TokensRepresentationOutput,
)
from xpmir.text.huggingface.encoders import HFTokensEncoder
from xpmir.text.encoders import TokenizedTextEncoder
from xpmir.neural.dual import DotDense
from xpmir.utils.utils import easylog, batchiter
from xpmir.utils.misc import opt_slice
from xpmir.text.tokenizers import TokenizerBase
from xpmir.documents.samplers import DocumentSampler
from xpmir.text.huggingface.base import HFMaskedLanguageModel

from sae.encoder import (
    HFColBERTTokensEncoder,
)
from sae import SAESimilarityInput
from sae.fasttopk import FastTopKSAE
from utils.utils import repad, unpad

logger = easylog()

# deprecated for the moment.
@define
class SAETokensRepresentationOutput(TokensRepresentationOutput):

    sae_input: torch.Tensor
    """The tensor before doing the reconstruction of SAE
    [bs, length, hs],
    """

    inds: Optional[torch.Tensor] = None
    """The indice for the sparse vectors after the SAE encoding
    [bs, length, k]
    """

    vals: Optional[torch.Tensor] = None
    """The vals for the sparse vectors after the SAE encoding
    [bs, length, k]
    """

    aux_inds: Optional[torch.Tensor] = None
    """The aux indice for the sparse vectors after the SAE encoding
    [bs, length, aux_k]
    """

    aux_vals: Optional[torch.Tensor] = None
    """The aux vals for the sparse vectors after the SAE encoding
    [bs, length, aux_k]
    """

    # log only
    sparsity: Optional[torch.Tensor] = None
    """The sparsity of the current input encoded by SAE
    shape [tokens,], so N.A. when using the __getitem__
    """

    before_sparsity: Optional[torch.Tensor] = None
    """The sparsity before applying the topk
    shape [tokens, ] so N.A. when using the __getitem__
    """

    latent_app_doc: Optional[torch.Tensor] = None
    """the matrix represent the if a latent activates within
    documents, the of shape [bs, sae_width]
    """

    def __getitem__(self, ix: Union[slice, int]):
        return self.__class__(
            opt_slice(self.value, ix),
            opt_slice(self.tokenized, ix),
            opt_slice(self.sae_input, ix),
            opt_slice(self.inds, ix),
            opt_slice(self.vals, ix),
            opt_slice(self.aux_inds, ix),
            opt_slice(self.aux_vals, ix),
            None,
            None,
            opt_slice(self.latent_app_doc, ix),
        )


class SAEAdapter(HFTokensEncoder, ABC):

    model: Param[HFTokensEncoder]
    """The colbert model/encoder
    Usually in type of HFColBERTTokensEncoder for vanilla version
    Or in type ColBERTProjectorAdapter if want to have a projector on top
    """

    sae_width: Param[int] = 65536
    """The number of neurons for the SAE"""

    normalize: Param[bool] = True
    """If true, we make the input of the SAE's mean norm is around
    1 and the bias is 0

    Else we no normalizing and the input of the model is normalized
    """

    def __initialize__(self, options):
        super().__initialize__(options)
        if isinstance(self.model, HFColBERTTokensEncoder):
            # If the starting point is a pretrained IR model
            hidden_size = self.model.model.hf_config.colbert.dim
        else:
            hidden_size = self.model.model.hf_config.hidden_size
        self.sae = self.initialize_sae(hidden_size)

    @abstractmethod
    def initialize_sae(self, hidden_size):
        raise NotImplementedError

    def forward(self, tokenized: TokenizedTexts) -> TokensRepresentationOutput:
        raise NotImplementedError


class FastTopKSAEAdapter(SAEAdapter):
    """Currently the input vectors for the SAE is pseudo normalize.
    which means the mean norm of the vectors is in average 1.
    """

    k: Param[int]
    """The number of top_k for the model"""

    aux_k: Param[int] = 512
    """The auxillary top_k value"""

    dead_steps_threshold: Param[int] = 10_000_000
    """The dead steps threshold"""

    def initialize_sae(self, hidden_size):
        if self.normalize:
            self.register_buffer("mean_norm", torch.tensor(1.0))
            self.register_buffer("mean_bias", torch.zeros(hidden_size))
        return FastTopKSAE(
            sae_width=self.sae_width,
            hidden_size=hidden_size,
            k=self.k,
            aux_k=self.aux_k,
            dead_steps_threshold=self.dead_steps_threshold,
        )

    def forward(
        self,
        tokenized: TokenizedTexts,
        # all use default values when training SAE
        override_k: int = 0,  # the override k used instead of the one in the param
        mask_attend: bool = False,  # if the mask tokens applies the SAE
    ):
        y: TokensRepresentationOutput = self.model(tokenized)
        # the value is (psuedo-)normalized here,
        # if use the projector adaptor, norm <= 1, else == 1
        sae_input = y.value
        bs, length, _ = sae_input.shape

        if self.normalize:
            # rescale the input vectors to in average norm 1
            sae_input = (sae_input - self.mean_bias) / self.mean_norm

        if mask_attend:
            # all the tokens is processed by the SAE.
            unpad_values, indices = unpad(sae_input, torch.ones_like(y.tokenized.mask))
        else:
            # only the unmasked tokens is processed by the SAE.
            # the masked tokens becomes all 0 after repad
            unpad_values, indices = unpad(sae_input, y.tokenized.mask)

        # only the sae_encoder, no sae_decoder
        inds, vals, aux_inds, aux_vals, sp, b_sp = self.sae(unpad_values, override_k)
        with torch.no_grad():  # for logging
            if inds is not None:
                doc_id = indices // length  # the assignment of the doc
                doc_overlap = torch.zeros(bs, self.sae_width, dtype=torch.int8).to(
                    vals.device
                )
                token_id = (
                    torch.arange(inds.shape[0])
                    .to(inds.device)
                    .unsqueeze(-1)
                    .expand_as(inds)[vals > 0]
                )
                latent_id = inds[vals > 0]
                doc_overlap[doc_id[token_id], latent_id] = 1
            else:
                doc_overlap = None

        # repad the rcst vector and aux reconstruction to the doc level
        if inds is not None:
            inds = repad(inds, indices, bs, length)  # [bs, length, k]
            aux_inds = repad(aux_inds, indices, bs, length)  # [bs, length, aux_k]
            aux_vals = repad(aux_vals, indices, bs, length)  # [bs, length, aux_k]

        vals = repad(vals, indices, bs, length)  # shape [bs, length, k]

        # if self.normalize:
        #     # if the input is normalized
        #     vals = vals * self.mean_norm  # reconstruct the original magnitude
        #     if aux_vals is not None:
        #         aux_vals = aux_vals * self.mean_norm

        return SAETokensRepresentationOutput(
            value=None,
            tokenized=y.tokenized,
            sae_input=sae_input,  # not masked
            inds=inds,  # masked tokens are 0
            vals=vals,  # masked tokens are 0
            aux_inds=aux_inds,  # masked tokens are 0
            aux_vals=aux_vals,  # masked tokens are 0
            # only for logging, no gradients
            sparsity=sp,
            before_sparsity=b_sp,
            latent_app_doc=doc_overlap,
        )


class SAETokenizedEncoder(TokenizedTextEncoder):
    """
    The class that tokenize and encoder the vectors.

    The abstract class that tokenize and encoder the SAE into the
    representations for IR.

    It is used for SAE training also.
    """

    encoder: Param[SAEAdapter]
    """The basic ColBERT SAE tokenized Encoder from that we obtain the SAE
    output, contains the sparse index and corresponding activations at the
    token level
    """

    length: Param[int]
    """The length of the tokenized text to encode"""

    mask_punctuation: Param[bool] = True
    """Whether we mask the punctuation of the documents
    """

    override_k: Param[int] = 1024
    """In the splade training, we apply a larger k in order to cover more
    If this value = 0 means no override
    Also support full encoding, where k = sae_width
    """

    def __initialize__(self, options):
        super().__initialize__(options)
        self.skiplist = [
            self.tokenizer.tokenizer.tokenizer.encode(symbol, add_special_tokens=False)[
                0
            ]
            for symbol in string.punctuation
        ]
        self.sae_width = self.encoder.sae_width
        self.pad_token_id = self.tokenizer.tokenizer.tokenizer.pad_token_id

    def apply_punctuation_mask(self, tokenized: TokenizedTexts):
        mask = [
            [(x not in self.skiplist) and (x != self.pad_token_id) for x in d]
            for d in tokenized.ids.cpu().tolist()
        ]
        mask = (
            torch.tensor(mask).to(dtype=tokenized.mask.dtype).to(tokenized.mask.device)
        )  # shape [bs, length]
        return TokenizedTexts(
            tokens=tokenized.tokens,
            ids=tokenized.ids,
            lens=tokenized.lens,
            mask=mask,
            token_type_ids=tokenized.token_type_ids,
        )

    @property
    def dimension(self):
        return self.sae_width

    def tokenize(self, inputs):
        """Using the base encoder's tokenizer to tokenize"""
        return self.tokenizer.tokenize(inputs, TokenizerOptions(self.length))

    def forward(self, inputs):
        tokenized = self.tokenize(inputs)
        return self.forward_tokenized(tokenized)

    def forward_tokenized(self, tokenized):
        """Used for SAE training, so do recon"""
        sae_output: SAETokensRepresentationOutput = self.encoder(tokenized)
        vals = sae_output.vals  # shape [bs, length, k]
        if self.mask_punctuation:
            tokenized = self.apply_punctuation_mask(sae_output.tokenized)
            vals = vals * tokenized.mask.unsqueeze(-1)
        else:
            tokenized = sae_output.tokenized

        # FIXME: better return TokenRepresentationOutput,
        # currently using this cause the hooks use SimilarityInput.
        return SAESimilarityInput(
            # the reconstruction value, not normalized
            value=sae_output.value,
            mask=tokenized.mask,
            # the input before the sae, normalized depend on opt
            sae_input=sae_output.sae_input,
            sparsity=sae_output.sparsity,
            inds=sae_output.inds,
            vals=sae_output.vals,
            # the aux_rcst, not normalized
            aux_inds=sae_output.aux_inds,  # masked tokens are 0
            aux_vals=sae_output.aux_vals,  # masked tokens are 0
            # for loggging
            before_sparsity=sae_output.before_sparsity,  # only for topk sae
            latent_app_doc=sae_output.latent_app_doc,
        )


# For SAE-SPLADE
class SAESPLADETokenizedEncoder(SAETokenizedEncoder):
    """The base encoder for SAESPLADE"""

    aggregation: Param[str] = "amax"
    """The type of the aggregation, usually using amax for splade"""

    def aggregate(self, inds: torch.Tensor, vals: torch.Tensor):
        """
        inds shape [bs, length, k] or None,
        vals shape [bs, length, k]
        the vals is already masked and relued.
        return the aggregated value of shape [bs, sae_width]
        """
        if inds is None:
            if self.aggregation == "amax":
                return torch.max(vals, dim=1).values
            else:
                raise NotImplementedError
        bs = vals.shape[0]
        inds = inds.reshape(bs, -1)  # shape [bs, length * k]
        vals = vals.flatten()  # shape [bs * length * k]
        values = torch.zeros(bs * self.sae_width).to(vals.device).to(vals.dtype)
        flat_inds = (
            torch.arange(bs).unsqueeze(-1).to(vals.device) * self.sae_width + inds
        ).flatten()  # shape [bs * length * k]
        values.scatter_reduce_(0, flat_inds, vals, reduce=self.aggregation)
        return values.reshape(bs, self.sae_width)

    def forward_tokenized(self, tokenized):
        sae_output: SAETokensRepresentationOutput = self.encoder(
            tokenized,
            override_k=self.override_k,
        )
        inds = sae_output.inds  # shape [bs, length, k] or None
        vals = sae_output.vals  # shape [bs, length, k]

        if self.encoder.normalize:
            # if the input of the SAE is normalized, the resulting activation is
            # too small, reconstruct the value to its original norm.
            # if we use the aux_vals as a auxillary loss in splade, we also need
            # to do the same thing for that
            vals = vals * self.encoder.mean_norm  # reconstruct the original magnitude

        if self.mask_punctuation:
            tokenized = self.apply_punctuation_mask(sae_output.tokenized)
            vals = vals * tokenized.mask.unsqueeze(-1)
        else:
            tokenized = sae_output.tokenized
        aggregated = torch.log1p(self.aggregate(inds, vals))
        return TextsRepresentationOutput(
            value=aggregated,  # shape [bs, sae_width]
            tokenized=tokenized,
        )


# For SPLADE baseline
class SPLADEEncoder(TokenizedTextEncoder):

    encoder: Param[HFMaskedLanguageModel]
    """The encoder from Hugging Face"""

    length: Param[int]
    """The length of the tokenized text to encode"""

    mask_punctuation: Param[bool] = True
    """Whether we mask the punctuation of the documents
    """

    override_k: Param[int] = -1
    """In the splade training,
    we apply a different k with the original version of the training.
    Default with k = -1 which means no override k options.
    """

    aggregation: Param[str] = "amax"
    """The type of the aggregation, usually using amax for splade"""

    def __initialize__(self, options):
        super().__initialize__(options)
        self.skiplist = [
            self.tokenizer.tokenizer.tokenizer.encode(symbol, add_special_tokens=False)[
                0
            ]
            for symbol in string.punctuation
        ]
        self.vocab_size = self.encoder.model.config.vocab_size
        self.pad_token_id = self.tokenizer.tokenizer.tokenizer.pad_token_id

    def apply_punctuation_mask(self, tokenized: TokenizedTexts):
        mask = [
            [(x not in self.skiplist) and (x != self.pad_token_id) for x in d]
            for d in tokenized.ids.cpu().tolist()
        ]
        mask = (
            torch.tensor(mask).to(dtype=tokenized.mask.dtype).to(tokenized.mask.device)
        )  # shape [bs, length]
        return TokenizedTexts(
            tokens=tokenized.tokens,
            ids=tokenized.ids,
            lens=tokenized.lens,
            mask=mask,
            token_type_ids=tokenized.token_type_ids,
        )

    @property
    def dimension(self):
        return self.vocab_size

    def tokenize(self, inputs):
        """Using the base encoder's tokenizer to tokenize"""
        return self.tokenizer.tokenize(inputs, TokenizerOptions(self.length))

    def forward(self, inputs):
        tokenized = self.tokenize(inputs)
        return self.forward_tokenized(tokenized)

    def aggregate(self, inds: torch.Tensor, vals: torch.Tensor):
        """
        inds shape [bs, length, k] or None,
        vals shape [bs, length, k]
        the vals is already masked and relued.
        return the aggregated value of shape [bs, sae_width]
        """
        if inds is None:
            if self.aggregation == "amax":
                return torch.max(vals, dim=1).values
            else:
                raise NotImplementedError
        bs = vals.shape[0]
        inds = inds.reshape(bs, -1)  # shape [bs, length * k]
        vals = vals.flatten()  # shape [bs * length * k]
        values = torch.zeros(bs * self.vocab_size).to(vals.device).to(vals.dtype)
        flat_inds = (
            torch.arange(bs).unsqueeze(-1).to(vals.device) * self.vocab_size + inds
        ).flatten()  # shape [bs * length * k]
        values.scatter_reduce_(0, flat_inds, vals, reduce=self.aggregation)
        return values.reshape(bs, self.vocab_size)

    def forward_tokenized(self, tokenized):
        # shape [bs, length, vocab_size]
        splade_output = self.encoder(tokenized).logits
        tokenized = tokenized.to(self.encoder.model.device)
        if self.mask_punctuation:
            tokenized = self.apply_punctuation_mask(tokenized)

        # apply the mask and relu
        splade_output = (splade_output * tokenized.mask.unsqueeze(-1)).relu()
        if self.override_k > 0:
            # shape [bs, length, k]
            vals, inds = torch.topk(splade_output, k=self.override_k, dim=-1)
            aggregated = torch.log1p(self.aggregate(inds, vals))
        else:
            aggregated = torch.log1p(self.aggregate(None, splade_output))
        return TextsRepresentationOutput(
            value=aggregated,  # shape [bs, sae_width]
            tokenized=tokenized,
        )

    def static(self):
        """The HFMaskedLanguageModel's params are learnable"""
        return False


# The dual encoder
class SAESPLADE(DotDense):
    """Compare to the traditional one, we add the alpha multiplier
    Maybe not needed"""

    scale: Param[bool] = True
    """Whether scaling the values of the representation"""

    scale_init: Param[float] = 1.0
    """The initial value of the scaling value,
    normally should be 1, but we can set some smaller values to avoid
    overflow in fp16.
    """

    def __initialize__(self, options):
        super().__initialize__(options)
        if self.scale:
            self.alpha = nn.Parameter(torch.tensor(self.scale_init))

    def encode_queries(self, records):
        q_encoded: TextsRepresentationOutput = self._query_encoder(records)
        value = q_encoded.value * self.alpha if self.scale else q_encoded.value
        return TextsRepresentationOutput(
            value=value,  # shape [bs, sae_width]
            tokenized=q_encoded.tokenized,
        )

    def encode_documents(self, records):
        d_encoded: TextsRepresentationOutput = self.encoder(records)
        value = d_encoded.value * self.alpha if self.scale else d_encoded.value
        return TextsRepresentationOutput(
            value=value,  # shape [bs, sae_width]
            tokenized=d_encoded.tokenized,
        )

    def merge_queries(self, queries: List[TextsRepresentationOutput]):
        if len(queries) == 0:
            return queries[0]
        # assume that all the query are padded to the same
        # length in the tokenizer
        value = torch.cat([query.value for query in queries])
        # Only redo the ids and the mask,
        # the others are not important for validation
        if queries[0].tokenized.ids is not None:
            ids = torch.cat([query.tokenized.ids for query in queries])
        if queries[0].tokenized.mask is not None:
            mask = torch.cat([query.tokenized.mask for query in queries])
        return TextsRepresentationOutput(
            value=value,
            tokenized=TokenizedTexts(
                tokens=None, ids=ids, lens=None, mask=mask, token_type_ids=None
            ),
        )


# the init task
class ColBERTSAEBiasInitialization(LightweightTask):
    """Initializing the SAE's pre-encoder-bias with the mean of several
    document"""

    encoder: Param[SAEAdapter]
    """The original model, use it calculate the original embeddings
    to initialize the bias.
    """

    tokenizer: Param[TokenizerBase]
    """The tokenizer to tokenize the text"""

    d_sampler: Param[DocumentSampler]
    """The document sampler for the model
    assume all the d_sampler are process directly in one batch
    for the model
    """

    device: Meta[Device]
    """The device"""

    batch_size: Meta[int] = 256
    """The batch size of the intialization"""

    def execute(self) -> None:
        self.device.execute(self.device_execute)

    def device_execute(self, device_information: DeviceInformation):
        self.tokenizer.initialize(ModuleInitMode.DEFAULT.to_options())
        self.encoder.initialize(ModuleInitMode.DEFAULT.to_options())

        # put to eval model to calculate the mean representation
        self.encoder.to(device_information.device).eval()
        _, doc_iter = self.d_sampler()
        d_records = [record for record in doc_iter]
        pad_token_id = self.tokenizer.tokenizer.tokenizer.pad_token_id
        skiplist = [
            self.tokenizer.tokenizer.tokenizer.encode(symbol, add_special_tokens=False)[
                0
            ]
            for symbol in string.punctuation
        ]
        if isinstance(self.encoder.model, HFColBERTTokensEncoder):
            hs = self.encoder.model.model.hf_config.colbert.dim
        else:
            hs = self.encoder.model.model.hf_config.hidden_size
        with torch.no_grad():
            # calculate the mean norm
            gb_mean_norm = torch.zeros(1).to(device_information.device)
            gb_mean_bias = torch.zeros(hs).to(device_information.device)
            gb_count = 0
            for sub_set in batchiter(self.batch_size, d_records):
                tokenized = self.tokenizer.tokenize(
                    sub_set, options=TokenizerOptions(max_length=180)
                )
                encoded = self.encoder.model(tokenized)
                sae_input = encoded.value
                punct_mask = [
                    [(x not in skiplist) and (x != pad_token_id) for x in d]
                    for d in encoded.tokenized.ids.cpu().tolist()
                ]
                punct_mask = (
                    torch.tensor(punct_mask)
                    .to(dtype=encoded.tokenized.mask.dtype)
                    .to(encoded.tokenized.mask.device)
                )
                unpad_values, _ = unpad(sae_input, punct_mask)

                b_mean_bias = unpad_values.mean(dim=0)
                b_mean_norm = (unpad_values - b_mean_bias).norm(dim=-1).mean()
                b_count = unpad_values.shape[0]

                gb_ratio = gb_count / (gb_count + b_count)
                gb_mean_norm = gb_ratio * gb_mean_norm + (1 - gb_ratio) * b_mean_norm
                gb_mean_bias = gb_ratio * gb_mean_bias + (1 - gb_ratio) * b_mean_bias

                gb_count += b_count

        self.encoder.mean_bias.data = gb_mean_bias
        self.encoder.mean_norm.data = gb_mean_norm
        logger.info(
            f"initializing SAE pre_norm muliplier to the mean of {gb_count} \
            document's embeddings, the mean of the norm is {gb_mean_norm}"
        )

        logger.info(
            f"initializing decoder bias to the mean of {gb_count} \
            document's embeddings, the norm of the bias is {gb_mean_bias.norm()}"
        )

        # put back to the train mode!
        self.encoder.train()
