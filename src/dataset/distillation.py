# The distillation dataset
from typing import List, Iterator, NamedTuple, Iterable
import json
import torch
import random
from experimaestro import Config, Param
from datamaestro.data import File
from datamaestro_text.data.ir.base import (
    TopicRecord,
    DocumentRecord,
    ScoredItem,
    IDItem,
    create_record,
    TextItem,
    InternalIDItem,
)

from xpmir.letor.samplers.hydrators import SampleHydrator, DocumentStore, TextStore
from xpmir.rankers import ScoredDocument
from xpmir.utils.iter import (
    SkippingIterator,
    SerializableIteratorTransform,
)


class ListwiseDistillationSample(NamedTuple):
    query: TopicRecord
    """The query"""

    documents: List[DocumentRecord]
    """List of documents with teacher scores"""


class ListwiseDistillationSamples(Config, Iterable[ListwiseDistillationSample]):
    def __iter__(self) -> Iterator[ListwiseDistillationSample]:
        raise NotImplementedError()

    @property
    def nway(self):
        raise NotImplementedError()


class ListwiseHydrator(ListwiseDistillationSamples, SampleHydrator):
    """Hydrate ID-based samples with document and/or query content"""

    samples: Param[ListwiseDistillationSamples]
    """The distillation samples without texts for query and documents"""

    def transform(self, sample: ListwiseDistillationSample):
        topic, documents = sample.query, sample.documents

        if transformed := self.transform_topics([topic]):
            topic = transformed[0]

        if transformed := self.transform_documents(documents):
            documents = tuple(
                ScoredDocument(d, sd[ScoredItem].score)
                for d, sd in zip(transformed, documents)
            )

        return ListwiseDistillationSample(topic, documents)

    def __iter__(self) -> Iterator[ListwiseDistillationSample]:
        iterator = iter(self.samples)
        return SerializableIteratorTransform(
            SkippingIterator.make_serializable(iterator), self.transform
        )

    @property
    def nway(self):
        return self.samples.nway


class RandomIdxSerializableIteratorTransform(SerializableIteratorTransform):
    def __init__(self, iterator, transform, idx_max):
        super().__init__(iterator, transform)
        self.idx_max = idx_max

    def __next__(self):
        idx = random.randint(0, self.idx_max - 1)
        return self.transform(next(self.iterator), idx)


class MultiLingualListwiseHydrator(ListwiseHydrator):

    documentstore: Param[List[DocumentStore]]
    """The list of the document store for different languages"""

    querystore: Param[List[TextStore]]
    """The list of the query store for different languages"""

    def transform_topics(self, topics: List[TopicRecord], lang_idx: int):
        if self.querystore[lang_idx] is None:
            return None
        return [
            create_record(
                id=topic[IDItem].id, text=self.querystore[lang_idx][topic[IDItem].id]
            )
            for topic in topics
        ]

    def transform_documents(self, documents: List[DocumentRecord], lang_idx: int):
        if self.documentstore[lang_idx] is None:
            return None
        results = []
        for document in documents:
            if document.has(TextItem):
                results.append(document)
            elif document.has(InternalIDItem):
                results.append(
                    self.documentstore[lang_idx].document_int(
                        document[InternalIDItem].id
                    )
                )
            elif document.has(IDItem):
                results.append(
                    self.documentstore[lang_idx].document_ext(document[IDItem].id)
                )
            else:
                raise RuntimeError("Cannot handle this")
        return results

    def transform(self, sample: ListwiseDistillationSample, lang_idx: int):
        topic, documents = sample.query, sample.documents

        if transformed := self.transform_topics([topic], lang_idx):
            topic = transformed[0]

        if transformed := self.transform_documents(documents, lang_idx):
            documents = tuple(
                ScoredDocument(d, sd[ScoredItem].score)
                for d, sd in zip(transformed, documents)
            )

        return ListwiseDistillationSample(topic, documents)

    def __iter__(self) -> Iterator[ListwiseDistillationSample]:
        iterator = iter(self.samples)
        return RandomIdxSerializableIteratorTransform(
            SkippingIterator.make_serializable(iterator),
            self.transform,
            len(self.documentstore),
        )


class JSONBasedBatchDistillationSamples(ListwiseDistillationSamples, File):
    """A JSON based batchwise distillation samples dataset
    current moment, it contains the ids. The ids are of type int, but they are
    external ids when transform to string
    """

    nway_num: Param[int] = 64
    """the number of distillation samples to use for each query"""

    def __iter__(self) -> Iterator[ListwiseDistillationSample]:
        return self.iter()

    def iter(self) -> Iterator[ListwiseDistillationSample]:
        def iterate():
            with self.path.open("rt") as fp:
                for line in fp:
                    sample = json.loads(line)
                    query = create_record(id=str(sample[0]))

                    documents = []
                    documents.append(  # append the positive
                        DocumentRecord(
                            IDItem(str(sample[1][0])), ScoredItem(float(sample[1][1]))
                        )
                    )
                    num_neg_samples = min(len(sample) - 1, self.nway) - 1
                    indices = torch.randperm(len(sample) - 2) + 2
                    for i, _ in zip(indices, range(num_neg_samples)):
                        documents.append(
                            DocumentRecord(
                                IDItem(str(sample[i][0])),
                                ScoredItem(float(sample[i][1])),
                            )
                        )
                    yield ListwiseDistillationSample(query, documents)

        return SkippingIterator(iterate())

    @property
    def nway(self):
        return self.nway_num
