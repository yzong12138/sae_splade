# the file contains the trainer for the SAE training and SAE-SPLADE training
from typing import List
import torch
import torch.nn as nn
from torch.functional import Tensor
from numpy.random.mtrand import RandomState as RandomState

import sys
from experimaestro import Param, Config
from xpmir.letor.records import DocumentRecords
from xpmir.letor.trainers import LossTrainer
from xpmir.learning.context import TrainerContext, Loss
from xpmir.rankers import LearnableScorer
from xpmir.utils.utils import foreach, easylog
from xpmir.utils.iter import (
    MultiprocessSerializableIterator,
)
from xpmir.letor.trainers.batchwise import BatchwiseTrainer, BatchwiseLoss
from dataset.documents import InfiniteDocumentSampler
from dataset.records import ListwiseDistillationProductRecords
from dataset.samplers import DistillationInBatchNegativesSampler
from sae import InteractionVectorListener

logger = easylog()

# ----- SAE Training Trainers
class DocumentOnlySAETrainer(LossTrainer):

    sampler: Param[InfiniteDocumentSampler]

    def initialize(self, random, context):
        super().initialize(random, context)
        self.sampler_iter = MultiprocessSerializableIterator(
            self.sampler.document_batch_iter(self.batch_size)
        )

    def train_batch(self, records: DocumentRecords):
        # here the model is our SAE model, but we only call its encode_document

        # we create a subclass of ColBERTJumpReLUSaeAdapter which we do
        # the sampling inside the encode_documents and pass it for training.
        # For the normal one we used it for the validation only

        # How to make the parameters shared between these two encoder?
        document_sim = self.model(
            list(records.documents),
        )

        if torch.isnan(document_sim.vals).any() or torch.isinf(document_sim.vals).any():
            self.logger.error("nan or inf detected in the vals. Aborting.")
            sys.exit(1)

        if (
            torch.isnan(document_sim.aux_vals).any()
            or torch.isinf(document_sim.aux_vals).any()
        ):
            self.logger.error("nan or inf detected in the aux_vals. Aborting.")
            sys.exit(1)

        if self.context is not None:
            foreach(
                self.context.hooks(InteractionVectorListener),
                lambda hook: hook(self.context, None, document_sim, None),
            )


# ----- SAE-SPLADE Finetuning Trainers
class DistillationBatchwiseLoss(Config, nn.Module):
    """The abstract loss for pairwise distillation"""

    weight: Param[float] = 1.0
    NAME = "?"

    def initialize(self, ranker: LearnableScorer):
        pass

    def process(
        self, student_scores: Tensor, teacher_scores: Tensor, info: TrainerContext
    ):
        loss = self.compute(student_scores, teacher_scores, info)
        info.add_loss(Loss(f"batchwise-{self.NAME}", loss, self.weight))

    def compute(
        self, student_scores: Tensor, teacher_scores: Tensor, context: TrainerContext
    ) -> torch.Tensor:
        """
        Compute the loss

        Arguments:

            student_scores: A (batch x nway) tensor
            teacher_scores: A (batch x nway) tensor
        """
        raise NotImplementedError()


class DistillationBatchwiseKLLoss(DistillationBatchwiseLoss):
    """
    Follow the code of the colbertv2 to do a distillation over
    a batch of 'negative' for each query
    """

    NAME = "Distil-Batch-KL"

    def initialize(self, ranker):
        super().initialize(ranker)
        self.loss = nn.KLDivLoss(reduction="batchmean", log_target=True)

    def compute(
        self, student_scores: Tensor, teacher_scores: Tensor, info: TrainerContext
    ) -> torch.Tensor:
        log_teacher_scores = torch.nn.functional.log_softmax(teacher_scores, dim=-1)
        log_student_scores = torch.nn.functional.log_softmax(student_scores, dim=-1)
        return self.loss(log_student_scores, log_teacher_scores)


class DistillationBatchwiseMSELoss(DistillationBatchwiseLoss):
    """
    Follow the MSE distillation over a pair of documents for each query.
    """

    NAME = "delta-MSE"

    def initialize(self, ranker):
        super().initialize(ranker)
        self.loss = nn.MSELoss()

    def compute(
        self, student_scores: Tensor, teacher_scores: Tensor, info: TrainerContext
    ) -> torch.Tensor:
        pos_teacher = teacher_scores[:, 0].unsqueeze(-1)
        pos_student = student_scores[:, 0].unsqueeze(-1)

        return self.loss(
            (pos_student - student_scores)[:, 1:], (pos_teacher - teacher_scores)[:, 1:]
        )


class DistillationInBatchNegativeTrainer(BatchwiseTrainer):

    sampler: Param[DistillationInBatchNegativesSampler]
    """A batch-wise sampler but contain"""

    lossfn: Param[BatchwiseLoss]
    """The in batch negative loss"""

    lossfn_distillation: Param[List[DistillationBatchwiseLoss]]
    """The distillation loss"""

    need_ibn: Param[bool] = True
    """Whether we apply ibn loss"""

    need_distil: Param[bool] = True
    """Whether we apply the distillation loss"""

    def __validate__(self) -> None:
        assert self.need_distil or self.need_ibn

    def initialize(self, random: RandomState, context: TrainerContext):
        super().initialize(random, context)
        self.lossfn.initialize(context)
        for distil_lossfn in self.lossfn_distillation:
            distil_lossfn.initialize(self.ranker)

    def train_batch(self, batch: ListwiseDistillationProductRecords):
        # Get the next batch and compute the scores for each query/document
        # Get the training examples
        records = ListwiseDistillationProductRecords()
        records.add_topics(*batch.unique_topics)
        records.add_documents(*[d.document for d in batch.unique_documents])

        # Get the scores
        rel_scores = self.ranker(records, self.context)
        # try:
        #     alpha = self.ranker.alpha
        # except:
        #     alpha = 1

        if torch.isnan(rel_scores).any() or torch.isinf(rel_scores).any():
            self.logger.error("nan or inf relevance score detected. Aborting.")
            sys.exit(1)

        # batch score shape [bs_q, bs_d] where bs_d = bs_q * nway and bs_q = bs
        batch_scores = rel_scores.reshape(
            len(batch.unique_queries), len(batch.unique_documents)
        )
        ibn_input = batch.ibn_relevance_view(batch_scores)  # shape [bs, (bs-1)*nway+1]
        ibn_target = batch.ibn_relevance().to(
            ibn_input.device
        )  # shape [bs, (bs-1)*nway+1]

        distil_input = batch.distillation_view(batch_scores)  # shape [bs, nway]
        distil_target = (
            torch.tensor([d.score for d in batch.unique_documents])
            .reshape_as(distil_input)
            .to(distil_input.device)
        )  # shape [bs, nway]

        if self.need_ibn:
            self.lossfn.process(ibn_input, ibn_target, self.context)
        if self.need_distil:
            for distil_lossfn in self.lossfn_distillation:
                distil_lossfn.process(distil_input, distil_target, self.context)
