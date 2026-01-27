import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.kernels import (
    triton_sparse_dense_matmul,
    triton_sparse_transpose_dense_matmul,
)
import math


class FastTopKSAE(nn.Module):
    """The topK SAE with sparse kernels
    Topk SAE's doesn't have the b_enc in general
    https://github.com/bartbussmann/BatchTopK/blob/main/sae.py
    """

    def __init__(
        self,
        sae_width,
        hidden_size,
        k,
        aux_k,
        dead_steps_threshold,
    ):
        super().__init__()
        self.k = k
        self.sae_width = sae_width
        self.hidden_size = hidden_size
        self.aux_k = aux_k
        self.dead_steps_threshold = dead_steps_threshold
        self.register_buffer(
            "node_last_activate", torch.zeros(sae_width, dtype=torch.long)
        )

        # remove the a=math.sqrt(5), mode="fan_out" if we want pure kaiming uniform
        # currently it is the uniform of nn.Linear
        self.W_enc = nn.Parameter(
            torch.nn.init.kaiming_uniform_(
                torch.empty(hidden_size, sae_width), a=math.sqrt(5), mode="fan_out"
            )
        )
        self.b_enc = nn.Parameter(nn.Parameter(torch.zeros(sae_width)))
        self.W_dec = nn.Parameter(
            torch.nn.init.kaiming_uniform_(
                torch.empty(sae_width, hidden_size), a=math.sqrt(5), mode="fan_out"
            )
        )
        self.b_dec = nn.Parameter(torch.zeros(hidden_size))
        self.init_weight()

    def init_weight(self):
        self.W_dec.data = self.W_enc.data.T
        # make W_enc contiguous on hidden_size
        self.W_enc.data = self.W_enc.data.T.contiguous().T
        self.W_dec.data = F.normalize(self.W_dec.data, dim=-1)
        # make W_dec contiguous on hidden_size
        self.W_dec.data = self.W_dec.data.contiguous()

    def auxk_mask_fn(self, x):
        dead_mask = self.node_last_activate > self.dead_steps_threshold
        return x.data * dead_mask

    def forward(self, x, override_k):
        if override_k > 0:
            self.k = override_k

        class TritonEncoderAutogradAux(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x, pre_bias, weight, enc_bias):
                x = x - pre_bias
                latents_pre_act = x @ weight + enc_bias
                # shape [bs, k]
                vals, inds = torch.topk(latents_pre_act, self.k, dim=-1)
                # set num nonzero stat #
                self.node_last_activate += x.shape[0]
                self.node_last_activate[inds.reshape(-1).unique()] = 0
                # end stats #

                # no need to save the vals for backwards
                aux_vals, aux_inds = torch.topk(
                    self.auxk_mask_fn(latents_pre_act), self.aux_k, dim=-1
                )
                ctx.save_for_backward(x, weight, inds, aux_inds)

                return inds, vals, aux_inds, aux_vals, latents_pre_act

            @staticmethod
            def backward(ctx, _, grad_vals, __, grad_aux_vals, ___):
                # The two _ are the inds, which have no gradients.
                x, weight, inds, aux_inds = ctx.saved_tensors
                inds = torch.cat((inds, aux_inds), dim=-1)
                grad_vals = torch.cat((grad_vals, grad_aux_vals), dim=-1)

                grad_sum = torch.zeros(
                    self.sae_width, dtype=torch.float32, device=grad_vals.device
                )
                grad_sum.scatter_add_(
                    -1, inds.flatten(), grad_vals.flatten().to(torch.float32)
                )  # shape [sae_width]
                return (
                    triton_sparse_dense_matmul(inds, grad_vals, weight.T),
                    -(grad_sum @ weight.T),
                    triton_sparse_transpose_dense_matmul(
                        inds, grad_vals, x, N=self.sae_width
                    ).T,
                    grad_sum,
                )

        if self.k == self.sae_width:
            # Only in splade or colbert finetuning
            vals = (x - self.b_dec) @ self.W_enc + self.b_enc  # shape [bs, sae_width]
            with torch.no_grad():
                self.node_last_activate += x.shape[0]
                self.node_last_activate[vals.mean(dim=0).nonzero().squeeze(-1)] = 0
            inds = None
            aux_inds = None
            aux_vals = None
        else:
            inds, vals, aux_inds, aux_vals, pre_act = TritonEncoderAutogradAux.apply(
                x,
                self.b_dec,
                self.W_enc,
                self.b_enc,
            )

        vals = torch.relu(vals)
        if aux_vals is not None:
            aux_vals = torch.relu(aux_vals)
        # if do_recon:
        #     # the original reconstruction
        #     x_rcst = TritonDecoderAutograd.apply(inds,vals,self.W_dec.T)+self.b_dec
        #     # the auxillary reconstruction
        #     aux_vals = torch.relu(aux_vals)
        #     aux_rcst = TritonDecoderAutograd.apply(aux_inds, aux_vals, self.W_dec.T)
        # else:
        #     x_rcst = None
        #     aux_rcst = None

        # some other logging infos
        with torch.no_grad():
            # real_sparsity
            real_sparsity = (vals > 0).sum(-1)  # shape [bs]
            # sparsity before topk (only if we apply topk)
            if self.k < self.sae_width:
                before_topk_sparsity = (pre_act > 0).sum(-1)  # shape [bs]
            else:
                before_topk_sparsity = None

        return (
            inds,  # shape [tokens, k]
            vals,  # shape [tokens, k]
            aux_inds,  # shape [tokens, aux_k]
            aux_vals,  # shape [tokens, aux_k]
            # only for logging
            real_sparsity,
            before_topk_sparsity,
        )
