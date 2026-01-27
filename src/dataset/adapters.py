# The code to transfer the Adhoc dataset generated from MS MARCO passage v1 to
# the other mMARCO datasets with different languages.
from pathlib import Path
import shutil
from experimaestro import Param, Task, Annotated, pathgenerator
from datamaestro_text.data.ir import (
    Adhoc,
    IDItem,
    TextItem,
)
from xpmir.datasets.adapters import DocumentSubset
from datamaestro_text.data.ir.trec import TrecAdhocAssessments
from datamaestro_text.data.ir.csv import Topics as CSVTopics


class MMSubTopicAdhoc(Task):
    """Dataset with extracted queries and assessments, based on the original
    corpus"""

    eng_adhoc: Param[Adhoc]
    """The english subadhoc task"""

    mm_adhoc: Param[Adhoc]
    """The full multilingual adhoc task
    The qrel between the eng_adhoc and mm_adhoc is equivalent. The same passage
    under different language have the same hash id.
    """

    assessments: Annotated[Path, pathgenerator("assessments.tsv")]
    """Generated assessments file"""

    topics: Annotated[Path, pathgenerator("topics.tsv")]
    """Generated topics file"""

    def task_outputs(self, dep) -> Adhoc:
        return dep(
            Adhoc.C(
                id="",  # No need to have a more specific id since it is generated
                topics=dep(CSVTopics.C(id="", path=self.topics)),
                assessments=dep(TrecAdhocAssessments.C(id="", path=self.assessments)),
                documents=self.mm_adhoc.documents,
            )
        )

    def execute(self):
        topic_ids = [topic[IDItem].id for topic in self.eng_adhoc.topics.iter()]
        self.topics.parent.mkdir(parents=True, exist_ok=True)
        with self.topics.open("wt") as fp:
            for topic_id in topic_ids:
                topic = self.mm_adhoc.topics.topic_ext(topic_id)
                fp.write(f"""{topic_id}\t{topic[TextItem].text}\n""")

        with self.assessments.open("wt") as fp:
            for qrels in self.mm_adhoc.assessments.iter():
                if qrels.topic_id in set(topic_ids):
                    for qrel in qrels.assessments:
                        fp.write(f"""{qrels.topic_id} 0 {qrel.doc_id} {qrel.rel}\n""")


class MMSubCollectionAdhoc(Task):
    """Dataset with extracted (given) queries and assessments, and the corpus is
    selected based on a given id files.
    """

    eng_sub_docs: Param[DocumentSubset]
    """The english subadhoc task"""

    mm_adhoc: Param[Adhoc]
    """The full multilingual adhoc task
    The qrel between the eng_adhoc and mm_adhoc is equivalent. The same passage
    under different language have the same hash id.
    """

    docids_path: Annotated[Path, pathgenerator("docids.txt")]
    """The file containing the document identifiers of the collection"""

    def task_outputs(self, dep) -> Adhoc.C:
        return Adhoc.C(
            id="",  # No need to have a more specific id since it is generated
            topics=self.mm_adhoc.topics,
            assessments=self.mm_adhoc.assessments,
            documents=dep(
                DocumentSubset.C(
                    id="", base=self.mm_adhoc.documents, docids_path=self.docids_path
                )
            ),
        )

    def execute(self):
        shutil.copy(self.eng_sub_docs.docids_path, self.docids_path)
