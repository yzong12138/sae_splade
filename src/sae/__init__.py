from attrs import define
from typing import Optional

import torch
from xpmir.neural.dual import DualVectorListener
from xpmir.learning.context import TrainerContext

from xpmir.neural.interaction.common import (
    SimilarityInput,
)
from xpmir.utils.misc import opt_slice
from xpmir.utils.torch import to_device


class InteractionVectorListener(DualVectorListener):
    """A special dual vector listener designed for interaction models
    It can receive also the interaction matrix from the query and document,
    the interaction matrix is used to get the indice of the matching tokens
    """

    def __call__(
        self,
        context: TrainerContext,
        queries: torch.Tensor,
        documents: torch.Tensor,
        matching_indice: torch.Tensor,
    ):
        """Hook handler

        Args:
            context (TrainerContext): The training context
            queries (torch.Tensor): The query vectors
            documents (torch.Tensor): The document vectors
            matching_indice (torch.Tensor): the matching indice,
                of shape [Bq, Lq, Bd] for batchwise, and
                of shape [B, Lq] for pairwise
        Raises:
            NotImplementedError: _description_
        """
        raise NotImplementedError(f"__call__ in {self.__class__}")


@define
class SAESimilarityInput(SimilarityInput):
    # need to check the __getitem__ in order to use the full retriever
    # need to getitem of the inds, vals, value. Not even for the mask.

    sae_input: Optional[torch.Tensor] = None
    """The tensor before doing the reconstruction of SAE
    shape [bs, length, encoder_dim]
    """

    sparsity: Optional[torch.Tensor] = None
    """The sparsity of the each encoded token
    shape [tokens, ], so N.A. when getitem.
    """

    inds: Optional[torch.Tensor] = None
    """The indices of the latents after the SAE encoder
    shape [bs, length, k]
    """

    vals: Optional[torch.Tensor] = None
    """The values of the latents after the SAE encoder
    shape [bs, length, k]
    """

    aux_inds: Optional[torch.Tensor] = None
    """The aux indice for the sparse vectors after the SAE encoding
    [bs, length, aux_k]
    """

    aux_vals: Optional[torch.Tensor] = None
    """The aux vals for the sparse vectors after the SAE encoding
    [bs, length, aux_k]
    """

    before_sparsity: Optional[torch.Tensor] = None
    """The sparsity before applying the topk
    shape [tokens, ], so N.A. when getitem.
    """

    latent_app_doc: Optional[torch.Tensor] = None
    """the matrix represent the if a latent activates within
    documents (not at the token level), the of shape [bs, sae_width]
    """

    def __len__(self):
        return len(self.value)

    def __getitem__(self, index):
        return SAESimilarityInput(
            opt_slice(self.value, index),
            opt_slice(self.mask, index),
            opt_slice(self.sae_input, index),
            None,
            opt_slice(self.inds, index),
            opt_slice(self.vals, index),
            opt_slice(self.aux_inds, index),
            opt_slice(self.aux_vals, index),
            None,
            opt_slice(self.latent_app_doc, index),
        )

    def to(self, device):
        return SAESimilarityInput(
            to_device(self.value, device),
            to_device(self.mask, device),
            to_device(self.sae_input, device),
            None,
            to_device(self.inds, device),
            to_device(self.vals, device),
            to_device(self.aux_rcst, device),
            None,
            to_device(self.latent_app_doc, device),
        )


# flake8: noqa: F401
from sae.hooks import (
    SAEReconstructionRegu,
    SAEHierarchicalReconstructionRegu,
    SAEMatryoshkaReconstructionRegu,
    QDFlopsRegularizer,
    SAESparsityRegu,
    SAEDeadNodeLogger,
    SAELatentAppearanceLogger,
    DualSAELatentAppearanceLogger,
    SAEGradientAdjustHook,
    SAEGradientClippingHook,
)
from sae.encoder import (
    HFTokensEncoderNoType,
    HFTokensEncoderNoTypeTransform,
)
from sae.model import (
    SAETokenizedEncoder,
    SAESPLADETokenizedEncoder,
    SPLADEEncoder,
    SAESPLADE,
    SAEAdapter,
    FastTopKSAEAdapter,
    ColBERTSAEBiasInitialization,
)
