# the file contains the trainer for the document only version
import torch
from typing import Any, Optional, List
import numpy as np

from experimaestro import Param, Meta
from xpmir.letor import Random
from datamaestro_text.data.ir import DocumentRecord, Topics

from xpmir.letor.records import DocumentRecords
from xpmir.documents.samplers import DocumentSampler
from xpmir.learning import Sampler
from xpmir.utils.utils import easylog
from xpmir.utils.iter import (
    SerializableIterator,
    InfiniteSkippingIterator,
)

logger = easylog()


class NoRestoreInfiniteSkippingIterator(InfiniteSkippingIterator):
    """Avoid to skipping the records.
    Recommend to use when the input iterator is large and randomly ordered, e.g.
    the MIRACL dataset SAE pretraining with randomly order documents and
    queries.

    Not recommend to use when the input is clearly order, e.g. HeadSampler, etc
    """

    def restore_state(self, state):
        count = state["count"]
        logger.info("Avoid Skipping %d records: Iterate from beginning", count)
        self.position = 0


class DocumentQuerySampler(DocumentSampler):
    """A sampler of randomly giving query and documents
    Used in the ablation with the query training data.
    """

    doc_sampler: Param[DocumentSampler]
    """The document only sampler"""

    topics: Param[Topics]
    """The topics"""

    random: Param[Optional[Random]]
    """Random sampler"""

    max_count: Param[int] = 0
    """Maximum number of documents (if 0, no limit)"""

    max_ratio: Param[float] = 0
    """Maximum ratio of documents (if 0, no limit)"""

    q_proba: Param[float] = -1.0
    """The manually assigned q_proba in the sampling,
    if -1 means according to the corpus size of the query and document."""

    def __post_init__(self):
        self.d_count = self.doc_sampler()[0]
        self.t_count = self.topics.count()
        total_count = self.t_count + self.d_count

        if self.q_proba < 0:
            self.q_proba = self.t_count / total_count

        sampler_count = (self.max_ratio or 1) * total_count
        if self.max_count > 0:
            sampler_count = min(self.max_count, sampler_count)

        self.sampler_count = int(sampler_count)

    def __call__(self):
        return self.sampler_count, iter(self)

    def __iter__(self):
        state = np.random.RandomState() if self.random is None else self.random.state
        q_iter = self.topics.iter()
        d_iter = iter(self.doc_sampler)
        count = 0
        d_count_cur = 0
        q_count_cur = 0
        while count < self.sampler_count:
            r = state.rand()
            if (
                r < self.q_proba and q_count_cur < self.t_count
            ) or d_count_cur >= self.d_count:
                q_count_cur += 1
                count += 1
                yield next(q_iter)
            elif (
                r >= self.q_proba and d_count_cur < self.d_count
            ) or q_count_cur >= self.t_count:
                d_count_cur += 1
                count += 1
                yield next(d_iter)
            else:
                raise StopIteration


class MultipleDocumentSampler(DocumentSampler):
    """Used in multilingual document samplers"""

    doc_samplers: Param[List[DocumentSampler]]
    """The list of document only sampler"""

    max_count: Param[int] = 0
    """Maximum number of documents (if 0, no limit)"""

    max_ratio: Param[float] = 0
    """Maximum ratio of documents (if 0, no limit)"""

    def __post_init__(self):
        # the count of each sampler
        self.d_counts = torch.tensor(
            [doc_sampler()[0] for doc_sampler in self.doc_samplers]
        )
        self.total_count = int(torch.sum(self.d_counts))

        # the count of this sampler
        sampler_count = (self.max_ratio or 1) * self.total_count
        if self.max_count > 0:
            sampler_count = min(self.max_count, sampler_count)
        self.sampler_count = int(sampler_count)

    def __call__(self):
        return self.sampler_count, iter(self)

    def __iter__(self):
        count_cur = 0
        d_iters = [iter(doc_sampler) for doc_sampler in self.doc_samplers]
        d_counts_cur = torch.zeros_like(self.d_counts)

        while count_cur < self.sampler_count:
            # create the sampling distribution
            # if a sampler if already finished, don't sample from that
            valid_counts = (d_counts_cur < self.d_counts) * self.d_counts
            if valid_counts.sum() == 0:
                raise StopIteration
            valid_proba = valid_counts / valid_counts.sum()
            index = int(valid_proba.multinomial(num_samples=1))
            count_cur += 1
            d_counts_cur[index] += 1
            yield next(d_iters[index])


class InfiniteDocumentSampler(Sampler):
    """Loop over the DocumentSampler infinite times for i.e., start of a new
    epoch"""

    doc_sampler: Param[DocumentSampler]

    restore_state: Meta[bool] = True
    """Whether we restore the state: set to false if we need to skip a lot of
    items while the order is not important, e.g. when the doc_sampler is the
    combination of all MIRACL corpus (size > 100M)
    """

    def initialize(self, random):
        super().initialize(random)

    def document_iter(self) -> SerializableIterator[DocumentRecord, Any]:
        if self.restore_state:
            return InfiniteSkippingIterator(self.doc_sampler)
        else:
            return NoRestoreInfiniteSkippingIterator(self.doc_sampler)

    def document_batch_iter(self, size) -> SerializableIterator[DocumentRecords, Any]:
        """Batchwise iterator

        Can be subclassed by some classes to be more efficient"""

        class BatchIterator(SerializableIterator):
            def __init__(self, sampler: InfiniteDocumentSampler):
                self.iter = sampler.document_iter()

            def state_dict(self):
                return self.iter.state_dict()

            def load_state_dict(self, state):
                self.iter.load_state_dict(state)

            def __next__(self):
                batch = DocumentRecords()
                for _, record in zip(range(size), self.iter):
                    batch.add(record)
                return batch

        return BatchIterator(self)
