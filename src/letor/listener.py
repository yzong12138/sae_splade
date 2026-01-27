import torch
from typing import List, Optional, Tuple, Dict, Any, DefaultDict
from tqdm import tqdm
import math
import json
from experimaestro import Param, Meta
from datamaestro_text.data.ir import Documents, Adhoc, IDItem
import ir_measures

from xpmir.rankers import Retriever, ScoredDocument
from xpmir.evaluation import get_evaluator
from xpmir.learning.context import TrainState
from xpmir.neural.dual import DualRepresentationScorer
from xpmir.learning.batchers import Batcher
from xpmir.learning.learner import LearnerListenerStatus, Learner
from xpmir.letor.learner import ValidationListener, ValidationModuleLoader
from xpmir.utils.utils import easylog, foreach
from xpmir.letor.records import DocumentRecord, TopicRecord
from xpmir.learning import ModuleInitMode
from xpmir.letor import Device


logger = easylog()


def get_run(retriever: Retriever, dataset: Adhoc):
    """Returns the scored documents for each topic in a dataset"""
    ir_results, flops_results = retriever.retrieve_all(
        {topic[IDItem].id: topic for topic in dataset.topics.iter()}
    )
    return {
        qid: {sd.document[IDItem].id: sd.score for sd in scoredocs}
        for qid, scoredocs in ir_results.items()
    }, flops_results


def evaluate(retriever: Retriever, dataset: Adhoc, measures: List[str], details=False):
    """Evaluate a retriever on a given dataset

    :param retriever: The retriever to evaluate that also returns the FLOPs.
    :param dataset: The dataset on which to evaluate
    :param measures: The list of measures to compute (using ir_measures)
    :param details: if query-level metrics should be reported, defaults to False
    :return: The metrics (if details is False) or a tuple (metrics, detailed metrics)
    """
    evaluator = get_evaluator(
        [ir_measures.parse_measure(m) for m in measures], dataset.assessments
    )
    run, flops = get_run(retriever, dataset)

    aggregators = {m: m.aggregator() for m in evaluator.measures}
    details = DefaultDict(lambda: {}) if details else None
    for metric in evaluator.iter_calc(run):
        aggregators[metric.measure].add(metric.value)
        if details is not None:
            details[str(metric.measure)][metric.query_id] = metric.value

    metrics = {str(m): agg.result() for m, agg in aggregators.items()}
    if details is not None:
        return metrics, flops, details

    return metrics, flops


class FullRetrieverRescorerWithFLOPs(Retriever):
    """Scores all the documents from a collection
    And output the flops between the query and document representation

    Used in validation
    """

    documents: Param[Documents]
    """The set of documents to consider"""

    scorer: Param[DualRepresentationScorer]
    """The scorer (a dual representation scorer)"""

    batchsize: Param[int] = 0
    batcher: Meta[Batcher] = Batcher.C()
    device: Meta[Optional[Device]] = None

    def initialize(self):
        self.query_batcher = self.batcher.initialize(self.batchsize)
        self.document_batcher = self.batcher.initialize(self.batchsize)
        self.scorer.initialize(ModuleInitMode.DEFAULT.to_options())

        # Compute with the scorer
        if self.device is not None:
            self.scorer.to(self.device.value)

    def _retrieve(
        self,
        batch: List[ScoredDocument],
        query: str,
        scoredDocuments: List[ScoredDocument],
    ):
        scoredDocuments.extend(self.scorer.rsv(query, batch))

    def encode_queries(self, queries: List[Tuple[str, str]], encoded: List[Any], pbar):
        """Encode queries and append the tensor of encoded queries to the encoded

        Args:
            queries (List[Tuple[str, str]]): The input queries (id/text)
            encoded (List[Tuple[List[str], torch.Tensor]]): Full list of topics ??
            it should be the List[torch.Tensor]
        """

        encoded.append(self.scorer.encode_queries([text for _, text in queries]))
        pbar.update(len(queries))
        return encoded

    def score(
        self,
        documents: List[DocumentRecord],
        queries: List,
        scored_documents: List[List[ScoredDocument]],
        query_mean_flops,  # shape [embed_size]
        cum_flops,  # List
        pbar,
    ):
        """Score documents for a set of queries

        Every time the score process a batch of document together with whole set
        of queries

        scored_documents is filled with document batches, i.e. it contains [
        [s(q_0, d_0), ..., s(q_n, d0)], ..., [s(q_0, d_m), ..., s(q_n, d_m)] ]
        --> list of m*n

        :param documents: the batch of documents

        :param queries: List of queries

        :param scored_documents: (output) current lists of scored documents (one
            per query)
        """
        # Encode documents
        encoded = self.scorer.encode_documents(documents)

        # Process query by query
        new_scores = [[] for _ in documents]
        for ix in range(len(queries)):
            # Get a range of query records
            query = queries[ix : (ix + 1)]

            # Returns a query x document matrix
            scores = self.scorer.score_product(query.to(encoded.device), encoded, None)

            # Adds up to the lists
            scores = scores.flatten().detach()
            for ix, (document, score) in enumerate(zip(documents, scores)):
                new_scores[ix].append(ScoredDocument(document, float(score)))
                pbar.update(1)

        # Add each result to the full document list
        scored_documents.extend(new_scores)

        # calculate the mean flops. Doesn't ignore the corresponding positive one.
        flops = (
            query_mean_flops
            * (encoded.value > 0).float().sum(0)
            / self.documents.documentcount
        ).sum()
        cum_flops.append(flops)

    def retrieve(self, record: TopicRecord) -> List[ScoredDocument]:
        # Only use retrieve_all
        return self.retrieve_all({"_": record})["_"]

    def retrieve_all(
        self, queries: Dict[str, TopicRecord]
    ) -> Dict[str, List[ScoredDocument]]:
        """Input is a dictionary of query {id:text},
        return the a dictionary of {query_id: List of ScoredDocuments under the query}
        """

        self.scorer.eval()
        all_queries = list(queries.items())

        with torch.no_grad():
            # Encode all queries
            # each time the batcher will just encode a batchsize of queries
            # and then concat them together
            with tqdm(total=len(all_queries), desc="Encoding queries") as pbar:
                enc_queries = self.query_batcher.reduce(
                    all_queries, self.encode_queries, [], pbar
                )
            enc_queries = self.scorer.merge_queries(
                enc_queries
            )  # shape (len(queries), dimension)

            # Encode documents and score them and cumulate over the flops
            scored_documents: List[List[ScoredDocument]] = []
            # the mean flops of the query over the validation set
            query_mean_flops = (enc_queries.value > 0).float().mean(0)
            # the flops is stored in a tuple of the
            flops_values: List = []

            with tqdm(
                total=len(all_queries) * self.documents.documentcount,
                desc="Scoring documents",
            ) as pbar:
                self.document_batcher.process(
                    self.documents,
                    self.score,
                    enc_queries,
                    scored_documents,
                    query_mean_flops,
                    flops_values,
                    pbar,
                )

        qids = [qid for qid, _ in all_queries]
        ir_score = {
            qid: [sd[ix] for sd in scored_documents] for ix, qid in enumerate(qids)
        }
        flop_score = sum(flops_values)
        return ir_score, flop_score


class ParetoFLOPIRValidationListener(ValidationListener):
    """We store the checkpoint with the best flops value and the best
    evaluation score on IR. We design a score to calculate the based on:

    log(flops * beta) * alpha + 1 / mrr.

    This value is the smaller the better. We also leverage a constant to move
    this value in order to provide a better range of values, e.g. split by 0.

    In conclusion, the value is validated by: C - log(flops * beta) * alpha - 1 / mrr.
    """

    retriever: Param[FullRetrieverRescorerWithFLOPs]
    """The retriever for validation which compute the flops"""

    alpha: Param[float] = 0.1
    """The coeff to multiple the log flops"""

    beta: Param[float] = 4.5
    """The coeff to multiple the intial flops"""

    const: Param[float] = 3.0
    """The constant to minus the value to be central positioned"""

    def validate_score(self, ir_metric, flops_metric):
        # also add the +inf at the end. ==> pure lp_metric
        # avoid dividing by 0.
        eps = 1.0e-5
        return self.const - (
            1 / (max(ir_metric, eps))
            + math.log(max(flops_metric * self.beta, eps)) * self.alpha
        )

    def monitored(self):
        # get all the names of the storing metrics
        base_ir_metrics = [key for key, store in self.metrics.items() if store]

        F_keys = [f"pareto_{ir_metric}" for ir_metric in base_ir_metrics]
        return base_ir_metrics + F_keys

    def init_task(self, learner: "Learner", dep):
        all_stored_key_names = self.monitored()
        return {
            key: dep(
                ValidationModuleLoader.C(
                    value=learner.model,
                    listener=self,
                    key=key,
                    path=self.bestpath / key / TrainState.MODEL_PATH,
                )
            )
            for key in all_stored_key_names
        }

    def __call__(self, state: TrainState):
        if self.should_stop(state.epoch - 1) == LearnerListenerStatus.STOP:
            return LearnerListenerStatus.STOP

        foreach(
            self.hooks,
            lambda hook: hook.before(self.context),
        )

        if state.epoch % self.validation_interval == 0:
            # Compute validation metrics
            means, flops, details = evaluate(
                self.retriever, self.dataset, list(self.metrics.keys()), True
            )
            flops = float(flops)
            self.context.writer.add_scalar(
                f"{self.id}/validation_flops",
                flops,
                state.step,
            )

            # should have only one which is keep in this scenario on the pure ir-metrics
            for metric, keep in self.metrics.items():
                value = means[metric]
                # add the scaler
                self.context.writer.add_scalar(
                    f"{self.id}/{metric}/mean", value, state.step
                )

                # build a list of the values and the names
                metric_values = [value]  # pure ir
                metric_names = [metric]  # pure ir
                if keep:
                    validate_score = self.validate_score(value, flops)
                    self.context.writer.add_scalar(
                        f"{self.id}/pareto_{metric}/mean",
                        validate_score,
                        state.step,
                    )
                    metric_values.append(validate_score)
                    metric_names.append(f"pareto_{metric}")

                if state.epoch >= self.warmup:
                    for metric_name, metric_value in zip(metric_names, metric_values):
                        topstate = self.top.get(metric_name, None)
                        if topstate is None or metric_value > topstate["value"]:
                            # Save the new top JSON
                            self.top[metric_name] = {
                                "value": metric_value,
                                "epoch": self.context.epoch,
                            }

                            if keep:
                                logger.info(
                                    f"Saving the checkpoint {state.epoch}"
                                    f" for metric {metric_name}"
                                )
                                self.context.copy(self.bestpath / metric_name)

            # Update information
            with self.info.open("wt") as fp:
                json.dump(self.top, fp)

        foreach(
            self.hooks,
            lambda hook: hook.after(self.context),
        )

        # Don't apply the early stop for the moment.
        return LearnerListenerStatus.NO_DECISION
