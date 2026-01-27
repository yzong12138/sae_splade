from typing import List
import numpy as np
import torch
import torch.nn.functional as F
from experimaestro import Param, Meta

from xpmir.neural.dual import DualVectorListener
from xpmir.learning.metrics import ScalarMetric
from xpmir.learning.optim import GradientHook, GradientClippingHook
from xpmir.learning.context import TrainerContext, Loss

from sae.model import SAETokenizedEncoder
from utils.kernels import TritonDecoderAutograd
from utils.kernel_flex import triton_hierarchical_sae_loss
from sae import InteractionVectorListener
from sae import SAESimilarityInput, SimilarityInput
from xpmir.utils.utils import easylog
from utils.utils import unpad


logger = easylog()


# ---- Regularizers
class SAEReconstructionRegu(InteractionVectorListener):
    """The sae reconstruction loss"""

    coeff: Param[float]
    """The coefficient for this regularization"""

    coeff_aux: Param[float]
    """The coefficient for the auxillary loss if have"""

    normalize: Param[bool] = True
    """The sae_input are not necessarily have the unit norm.
    If this value is true, we force the sae_input to be unit norm
    and adjust the reconstructed values"""

    def __call__(
        self,
        context: TrainerContext,
        queries: SimilarityInput,
        documents: SAESimilarityInput,
        matching_indice: torch.Tensor,
    ):
        mask = documents.mask  # [bs, length, hs]
        sae_input, _ = unpad(documents.sae_input, mask)  # [tokens, hs]
        inds, _ = unpad(documents.inds, mask)
        vals, _ = unpad(documents.vals, mask)
        aux_inds, _ = unpad(documents.aux_inds, mask)
        aux_vals, _ = unpad(documents.aux_vals, mask)

        # the sae model
        sae = context.state.model.encoder.sae

        if self.normalize:
            # keep every term at the same contribution to the final loss.
            sae_input_norm = sae_input.norm(dim=-1, keepdim=True).detach()
            sae_input = sae_input / (sae_input_norm + 1e-5)
            vals = vals / (sae_input_norm + 1e-5)
            aux_vals = aux_vals / (sae_input_norm + 1e-5)

        # decode here
        rcst = TritonDecoderAutograd.apply(inds, vals, sae.W_dec.T) + sae.b_dec
        aux_rcst = TritonDecoderAutograd.apply(aux_inds, aux_vals, sae.W_dec.T)

        loss = torch.sum((sae_input.detach() - rcst) ** 2, dim=-1).mean()
        context.add_loss(
            Loss(
                "SAE-Reconstruction-loss",
                loss,
                self.coeff,
            )
        )

        aux_rcst = aux_rcst + rcst.detach()
        aux_loss = (
            torch.sum((sae_input.detach() - aux_rcst) ** 2, dim=-1).mean()
        ).nan_to_num(0)
        context.add_loss(
            Loss(
                "SAE-Recons-Aux-loss",
                aux_loss,
                self.coeff_aux,
            )
        )


class SAEHierarchicalReconstructionRegu(InteractionVectorListener):

    coeff: Param[float]
    """The coefficient for this regularization"""

    coeff_aux: Param[float]
    """The coefficient for the auxillary loss if have"""

    normalize: Param[bool] = True
    """The sae_input are not necessarily have the unit norm.
    If this value is true, we force the sae_input to be unit norm
    and adjust the reconstructed values"""

    def __call__(
        self,
        context: TrainerContext,
        queries: SimilarityInput,
        documents: SAESimilarityInput,
        matching_indice: torch.Tensor,
    ):
        mask = documents.mask  # [bs, length, hs]
        sae_input, _ = unpad(documents.sae_input, mask)  # [tokens, hs]
        inds, _ = unpad(documents.inds, mask)
        vals, _ = unpad(documents.vals, mask)
        aux_inds, _ = unpad(documents.aux_inds, mask)
        aux_vals, _ = unpad(documents.aux_vals, mask)

        # the input tokens should the power of 16.
        new_b = (sae_input.shape[0] // 16) * 16
        sae_input = sae_input[:new_b]
        inds = inds[:new_b]
        vals = vals[:new_b]
        aux_inds = aux_inds[:new_b]
        aux_vals = aux_vals[:new_b]

        # the sae model
        sae = context.state.model.encoder.sae

        if self.normalize:
            # keep every term at the same contribution to the final loss.
            sae_input_norm = sae_input.norm(dim=-1, keepdim=True).detach()
            sae_input = sae_input / (sae_input_norm + 1e-5)
            vals = vals / (sae_input_norm + 1e-5)
            aux_vals = aux_vals / (sae_input_norm + 1e-5)

        loss, rcst = triton_hierarchical_sae_loss(
            indices=inds,  # [tokens, k]
            weight=sae.W_dec,  # [sae_width, hs]
            vals=vals,  # [tokens, k]
            bias=sae.b_dec,  # [hs]
            target=sae_input.detach(),  # [tokens, hs]
        )
        loss = loss * sae_input.shape[1]
        context.add_loss(
            Loss(
                "SAE-Reconstruction-loss",
                loss,
                self.coeff,
            )
        )
        if self.coeff_aux > 0:
            aux_rcst = TritonDecoderAutograd.apply(aux_inds, aux_vals, sae.W_dec.T)
            aux_rcst = aux_rcst + rcst.detach()
            aux_loss = (
                torch.sum((sae_input.detach() - aux_rcst) ** 2, dim=-1).mean()
            ).nan_to_num(0)
            context.add_loss(
                Loss(
                    "SAE-Recons-Aux-loss",
                    aux_loss,
                    self.coeff_aux,
                )
            )


class SAEMatryoshkaReconstructionRegu(InteractionVectorListener):

    coeff: Param[float]
    """The coefficient for this regularization"""

    coeff_aux: Param[float]
    """The coefficient for the auxillary loss if have"""

    normalize: Param[bool] = True
    """The sae_input are not necessarily have the unit norm.
    If this value is true, we force the sae_input to be unit norm
    and adjust the reconstructed values"""

    matryoshaka_range: Param[List[int]] = [2048, 6144, 14336, 30720, 65536]
    """The range of the groups for Matryoshka SAE"""

    def __call__(
        self,
        context: TrainerContext,
        queries: SimilarityInput,
        documents: SAESimilarityInput,
        matching_indice: torch.Tensor,
    ):
        mask = documents.mask  # [bs, length, hs]
        sae_input, _ = unpad(documents.sae_input, mask)  # [tokens, hs]
        inds, _ = unpad(documents.inds, mask)
        vals, _ = unpad(documents.vals, mask)
        aux_inds, _ = unpad(documents.aux_inds, mask)
        aux_vals, _ = unpad(documents.aux_vals, mask)

        # the sae model
        sae = context.state.model.encoder.sae

        if self.normalize:
            # keep every term at the same contribution to the final loss.
            sae_input_norm = sae_input.norm(dim=-1, keepdim=True).detach()
            sae_input = sae_input / (sae_input_norm + 1e-5)
            vals = vals / (sae_input_norm + 1e-5)
            aux_vals = aux_vals / (sae_input_norm + 1e-5)

        # initialize the cumulate reconstructed vectors
        rcst = torch.zeros_like(sae_input) + sae.b_dec
        # initialize the cumulate loss
        l2_losses = torch.tensor([]).to(vals.device)

        # masking the latents not in the given range
        for i in range(len(self.matryoshaka_range)):
            s = 0 if i == 0 else self.matryoshaka_range[i - 1]
            e = self.matryoshaka_range[i]
            rcst = rcst + TritonDecoderAutograd.apply(
                inds, vals * ((inds >= s) & (inds < e)), sae.W_dec.T
            )
            l2_loss = torch.sum((sae_input.detach() - rcst) ** 2, dim=-1).mean()
            l2_losses = torch.cat([l2_losses, l2_loss.unsqueeze(0)])
        loss = torch.mean(l2_losses)
        context.add_loss(
            Loss(
                "SAE-Reconstruction-loss",
                loss,
                self.coeff,
            )
        )

        if self.coeff_aux > 0:
            aux_rcst = TritonDecoderAutograd.apply(aux_inds, aux_vals, sae.W_dec.T)
            aux_rcst = aux_rcst + rcst.detach()
            aux_loss = (
                torch.sum((sae_input.detach() - aux_rcst) ** 2, dim=-1).mean()
            ).nan_to_num(0)
            context.add_loss(
                Loss(
                    "SAE-Recons-Aux-loss",
                    aux_loss,
                    self.coeff_aux,
                )
            )


class SAESparsityRegu(InteractionVectorListener):
    """The SAE sparisity loss, based on l0"""

    coeff: Param[float]
    """The coefficient for this regularization"""

    def __call__(
        self,
        context: TrainerContext,
        queries: SimilarityInput,
        documents: SAESimilarityInput,
        matching_indice: torch.Tensor,
    ):
        sparsity = documents.sparsity.to(dtype=torch.float16)
        mean_sparse = torch.mean(sparsity)
        if self.coeff > 0:
            context.add_loss(
                Loss(
                    "l0-sparsity-nnz",
                    mean_sparse,
                    self.coeff,
                )
            )
        else:
            context.add_metric(
                ScalarMetric(
                    "l0-sparsity-nnz",
                    float(mean_sparse),
                    1,
                )
            )

        if documents.before_sparsity is not None:
            b_sp = documents.before_sparsity.to(dtype=torch.float16)
            b_sp_mean = torch.mean(b_sp)
            context.add_metric(
                ScalarMetric(
                    "l0-sparsity-nnz-before-topk",
                    float(b_sp_mean),
                    1,
                )
            )


# --- Only used for logging, not as a regularizer.
class QDFlopsRegularizer(DualVectorListener):
    def __call__(self, info: TrainerContext, queries, documents):
        # Assuming we are in batchwise distillation scenario
        # the len(documents) = len (query) * k
        # k represent the number of pos + hard neg for each query
        queries = queries.value  # shape [bs, vocab]
        documents = documents.value  # shape [bs*k, vocab]
        bs = len(queries)
        k = int(len(documents) / bs)
        assert k * len(queries) == len(documents)

        with torch.no_grad():
            # Expected qd-flops on this sample -- include self
            qdflops_count_w_self = (
                (queries > 0).float().mean(0) * (documents > 0).float().mean(0)
            ).sum()
            info.metrics.add(
                ScalarMetric(
                    "qdflops_count_w_self",
                    qdflops_count_w_self.item(),
                    1,
                )
            )

            # Expected qd-flops on this sample -- not include self
            d_count_sum = (documents > 0).float().sum(0)  # shape [vocab]
            d_count_per_batch = (
                (documents > 0).float().reshape(bs, k, -1).sum(1)
            )  # shape [bs, vocab]
            d_mean_count_wo_current = (d_count_sum.unsqueeze(0) - d_count_per_batch) / (
                k * (bs - 1)
            )  # shape [bs, vocab]
            qdflops_count_wo_self = (
                ((queries > 0) * d_mean_count_wo_current)  # shape [bs, vocab]
                .mean(0)
                .sum()
            )
            info.metrics.add(
                ScalarMetric(
                    "qdflops_count_wo_self",
                    qdflops_count_wo_self.item(),
                    1,
                )
            )


class SAEDeadNodeLogger(InteractionVectorListener):
    """Log the statistics of the dead nodes after a certain number of
    batches
    """

    num_tokens: Meta[int] = 20_000_000
    """log the ratio of the dead nodes with not activate for this consecutive
    number of batches"""

    def __call__(self, context, queries, documents, matching_indice=None):
        if isinstance(context.state.model, SAETokenizedEncoder):
            counts = context.state.model.encoder.sae.node_last_activate.data
        else:
            counts = context.state.model.encoder.encoder.sae.node_last_activate.data
        context.add_metric(
            ScalarMetric(
                "dead_nodes_ratio",
                float((counts > self.num_tokens).sum() / counts.shape[0]),
                1,
            )
        )


class SAELatentAppearanceLogger(InteractionVectorListener):
    """Log the statistics of the latents appearance of in the batch of training
    SAE
    """

    log_hist_step: Meta[int] = 400
    """the frequency of log out the histogram"""

    log_nums_doc_latent_ratio: Meta[List] = [
        2 / 768,
        8 / 768,
        32 / 768,
        128 / 768,
        512 / 768,
    ]
    """The latents appear > this number of documents with be noted"""

    def __call__(
        self,
        context: TrainerContext,
        queries: SimilarityInput,
        documents: SAESimilarityInput,
        matching_indice: torch.Tensor,
    ):
        latent_app_doc = documents.latent_app_doc  # shape [bs, sae_width]
        assert latent_app_doc is not None
        count_in_batch = latent_app_doc.sum(dim=0)
        latent_in_batch = (count_in_batch > 0).sum()

        context.add_metric(
            ScalarMetric(
                "Num latents in batch",
                float(latent_in_batch),
                1,
            )
        )
        # log out the number of latents which appear in more than certain
        # threshold of documents
        for t in self.log_nums_doc_latent_ratio:
            nums = ((count_in_batch / latent_app_doc.shape[0]) > t).sum()
            context.add_metric(
                ScalarMetric(
                    f"Num latents in more than {t:.1%} of total doc",
                    float(nums),
                    1,
                )
            )

        act_nums_doc = latent_app_doc.sum(dim=-1).to(dtype=torch.float16)
        act_nums_doc_mean = torch.mean(act_nums_doc)
        context.add_metric(
            ScalarMetric(
                "l0-sparsity-nnz-doc-activated",
                float(act_nums_doc_mean),
                1,
            )
        )

        # log out histgram
        if context.steps % self.log_hist_step == 0:
            # build the histogram of the appearance
            hist_latent_app_doc = (
                count_in_batch[count_in_batch > 0].cpu().detach().numpy()
            )
            context.writer.add_histogram(
                "train/latent_appearance_in_doc",
                hist_latent_app_doc,
                context.steps,
                # every integer from 1 to bs
                bins=np.linspace(
                    0.5, latent_app_doc.shape[0] + 0.5, latent_app_doc.shape[0] + 1
                ),
            )


class DualSAELatentAppearanceLogger(DualVectorListener):
    """Log the statistics of the latents appearance of in the batch of training
    documents when training a dual sparse model, e.g. SPLADE
    """

    log_hist_step: Meta[int] = 400
    """the frequency of log out the histogram"""

    log_nums_doc_latent_ratio: Meta[List] = [
        2 / 768,
        8 / 768,
        32 / 768,
        128 / 768,
        384 / 768,
        512 / 768,
    ]
    """The latents appear > this number of documents with be noted"""

    def __call__(self, context, queries, documents):
        latent_app_doc = documents.value > 0  # shape [bs, sae_width]
        count_in_batch = latent_app_doc.sum(dim=0)  # shape [sae_width]
        latent_in_batch = (count_in_batch > 0).sum()

        context.add_metric(
            ScalarMetric(
                "Num latents in batch",
                float(latent_in_batch),
                1,
            )
        )
        # log out the number of latents which appear in more than certain
        # threshold of documents
        for t in self.log_nums_doc_latent_ratio:
            nums = ((count_in_batch / latent_app_doc.shape[0]) > t).sum()
            context.add_metric(
                ScalarMetric(
                    f"Num latents in more than {t:.1%} of total doc",
                    float(nums),
                    1,
                )
            )

        # log out histgram
        if context.steps % self.log_hist_step == 0:
            # build the histogram of the appearance
            hist_latent_app_doc = (
                count_in_batch[count_in_batch > 0].cpu().detach().numpy()
            )
            context.writer.add_histogram(
                "train/latent_appearance_in_doc",
                hist_latent_app_doc,
                context.steps,
                # every integer from 1 to bs
                bins=np.linspace(
                    0.5, latent_app_doc.shape[0] + 0.5, latent_app_doc.shape[0] + 1
                ),
            )


# --- The learning hooks. Adjust the gradients during the training.
def remove_parallel_component_pt(grad, weight_data):
    """Weight_data and grad shape: [sae_dim, hiddens_dim]"""
    weight_normalized = F.normalize(weight_data, dim=-1)  # [sae_dim, hiddens_dim]
    parallel_component = torch.einsum("sh,sh->s", grad, weight_normalized)  # [sae_dim]
    return grad - parallel_component[:, None] * weight_normalized


class SAEGradientAdjustHook(GradientHook):
    """In this gradient adjusting hook, we remove the parallel
    part of the gradient for the decoder bias (pre-bias)
    and normalize the norm of the decoder bias afterwards

    assume the main the SAETokenizedTextEncoder
    """

    def before(self, main):
        # Maybe there is a better weight to find the module
        # shape [sae_width, hs]
        module_to_adjust = main.module.encoder.sae.W_dec
        module_to_adjust.grad = remove_parallel_component_pt(
            module_to_adjust.grad, module_to_adjust.data
        )

    def after(self, main):
        # shape [sae_width, hs]
        module_to_adjust = main.module.encoder.sae.W_dec
        module_to_adjust.data = F.normalize(module_to_adjust.data, dim=-1)


class SAEGradientClippingHook(GradientClippingHook):
    """The SAE gradient clipping hook
    which doesn't clip the threshold"""

    use_norm: Param[bool] = False
    """If this is true, then use the clipping based on norm"""

    def before(self, main):
        if self.use_norm:
            torch.nn.utils.clip_grad_norm_(main.module.parameters(), self.max_norm)
        else:
            torch.nn.utils.clip_grad_value_(main.module.parameters(), self.max_norm)
