import random
from typing import Any, List, TypeVar
from experimaestro import Param

from xpmir.learning.base import Sampler
from xpmir.letor.samplers import BatchwiseSampler
from dataset.records import ListwiseDistillationProductRecords
from dataset.distillation import ListwiseDistillationSamples, ListwiseDistillationSample


from xpmir.utils.iter import (
    SerializableIterator,
    SkippingIterator,
    SerializableIteratorAdapter,
)

from xpmir.letor.records import BatchwiseRecords

T = TypeVar("T")


class MultiSourceSkippingIterator(SerializableIterator):
    def __init__(self, iterators: List[SerializableIterator[T]]):
        self.iterators = iterators
        self.iterators_idx = list(range(len(self.iterators)))
        self.state = None

    def load_state_dict(self, state):
        self.state = state

    def state_dict(self):
        return {str(idx): it.state_dict() for idx, it in enumerate(self.iterators)}

    def restore_state(self, states):
        for idx, state in states.items():
            self.iterators[int(idx)].restore_state(state)

    def __next__(self) -> T:
        idx = random.choice(self.iterators_idx)
        return next(self.iterators[idx])


class DistillationListwiseSampler(Sampler):
    """Abstract class for pairwise samplers which output a set of (query,
    positive, and another list of document)"""

    samples: Param[ListwiseDistillationSamples]
    """the distillation samples"""

    def listwise_sampler(self) -> SerializableIterator[ListwiseDistillationSample, Any]:
        return SkippingIterator.make_serializable(iter(self.samples))


class MultiSourceDistillationListwiseSampler(Sampler):
    """Random choosing the source of dataset for sampling in multilingual
    scenario"""

    samples_list: Param[List[ListwiseDistillationSamples]]
    """the distillation samples"""

    def listwise_sampler(self) -> SerializableIterator[ListwiseDistillationSample, Any]:
        return MultiSourceSkippingIterator(
            [iter(samples) for samples in self.samples_list]
        )


class DistillationInBatchNegativesSampler(BatchwiseSampler):
    """An in-batch negative sampler constructured from a listwise one,
    we consider only the negatives from the other queries"""

    sampler: Param[DistillationListwiseSampler]
    """The base listwise sampler"""

    def initialize(self, random):
        super().initialize(random)
        self.sampler.initialize(random)

    def batchwise_iter(
        self, batch_size: int
    ) -> SerializableIterator[BatchwiseRecords, Any]:
        def iter(list_iter):
            while True:
                batch = ListwiseDistillationProductRecords()
                for _, record in zip(range(batch_size), list_iter):
                    batch.add_topics(record.query)
                    batch.add_documents(*record.documents)
                yield batch

        return SerializableIteratorAdapter(self.sampler.listwise_sampler(), iter)
