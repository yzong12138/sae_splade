# currently the important training hyperparameters are inside the experimental plan
from attrs import Factory, field
from typing import List, Union
from itertools import product
from functools import cached_property as attrs_cached_property
from experimaestro import Param
from experimaestro.experiments.configuration import ConfigurationBase
from xpmir.papers import configuration
from xpmir.papers.helpers import LauncherSpecification
from xpmir.papers.helpers.optim import TransformerOptimization
from xpmir.letor import Random
from xpmir.learning.schedulers import LinearWithWarmup
from xpmir.learning.devices import Device, CudaDevice
from xpmir.learning.optim import (
    AdamW,
    get_optimizers,
    ParameterOptimizer,
    RegexParameterFilter,
)
from xpmir.neural.dual import ScheduledFlopsRegularizer
from sae import (
    SAEReconstructionRegu,
    SAEHierarchicalReconstructionRegu,
    SAEMatryoshkaReconstructionRegu,
)
from functools import cached_property


class LinearWithWarmupWithDecay(LinearWithWarmup):
    """Linear warmup then stable followed by decay"""

    num_steps_to_decay: Param[int]
    """Number of steps to decay"""

    def lr_lambda(self, current_step: int, num_training_steps: int):
        # Still warming up
        if current_step < self.num_warmup_steps:
            return float(current_step) / float(max(1, self.num_warmup_steps))
        elif current_step < self.num_steps_to_decay:
            return 1
        # Not warming up: the ratio is between 1 (after warmup) and 0 (at the end)
        factor = max(
            0.0,
            float(num_training_steps - current_step)
            / float(max(1, num_training_steps - self.num_steps_to_decay)),
        )

        # Shift/scale so it is between 1 and min factor
        return (factor + self.min_factor) / (1.0 + self.min_factor)


class AdamWCustomBeta(AdamW):
    beta1: Param[float] = 0.9
    beta2: Param[float] = 0.999

    def __call__(self, parameters):
        from torch.optim import AdamW

        return AdamW(
            parameters,
            lr=self.lr,
            betas=(self.beta1, self.beta2),
            weight_decay=self.weight_decay,
            eps=self.eps,
        )


@configuration()
class SAEOptimization(TransformerOptimization):
    """This optimization propose a optimizer with a potentially a different beta1"""

    adam_beta1: float = 0.9
    """The beta1 of the adam optimizer"""

    def get_optimizer(self, regularization, lr):
        # Set weight decay to 0 if no regularization
        weight_decay = self.weight_decay if regularization else 0

        if self.optimizer_name == "adam-w":
            return AdamWCustomBeta.C(
                lr=lr,
                weight_decay=weight_decay,
                eps=self.eps,
                beta1=self.adam_beta1,
            )
        else:
            raise ValueError(f"Cannot handle optimizer named {self.optimizer_name}")


@configuration
class SPLADESAEOptimization(TransformerOptimization):
    """This optimization propose a optimizer with a different lr for the
    threshold. Modify a little bit the architecture to test various schedulers
    in parallel.

    Currently, the learning rate and the scheduler min factor and number of
    steps to decay is tested across various options.
    """

    adam_beta1: float = 0.9
    """The beta1 of the adam optimizer"""

    lr: List[float] = [1.0e-5, 2.0e-5]
    """The list of the learning rate to test"""

    warmup_min_factor: List[float] = [0, 1]
    """The min factor for the warmup"""

    num_steps_to_decay: List[int] = [10000, 80000]
    """The number of steps which keeps a high learning rate then decay"""

    @cached_property
    def scheduler_instance_list(self):
        scheduler_list = []
        for min_factor, decay_point in product(
            self.warmup_min_factor, self.num_steps_to_decay
        ):
            scheduler = (
                LinearWithWarmupWithDecay.C(
                    num_warmup_steps=self.num_warmup_steps,
                    num_steps_to_decay=decay_point,
                    min_factor=min_factor,
                )
                .tag("lr_decay", decay_point)
                .tag("lr_min", min_factor)
                if self.scheduler
                else None
            )
            scheduler_list.append(scheduler)
        return scheduler_list

    def get_optimizer(self, regularization, lr):
        # Set weight decay to 0 if no regularization
        weight_decay = self.weight_decay if regularization else 0

        if self.optimizer_name == "adam-w":
            return AdamWCustomBeta.C(
                lr=lr,
                weight_decay=weight_decay,
                eps=self.eps,
                beta1=self.adam_beta1,
            )
        else:
            raise ValueError(f"Cannot handle optimizer named {self.optimizer_name}")

    def prepare_one_optimizer(self, lr, scheduler_instance):
        if not self.re_no_l2_regularization:
            return get_optimizers(
                [
                    ParameterOptimizer.C(
                        scheduler=scheduler_instance,
                        optimizer=self.get_optimizer(True, lr),
                    ).tag("lr", lr),
                ]
            )

        return get_optimizers(
            [
                ParameterOptimizer.C(
                    scheduler=scheduler_instance,
                    optimizer=self.get_optimizer(False, lr),
                    filter=RegexParameterFilter(includes=self.re_no_l2_regularization),
                ),
                ParameterOptimizer.C(
                    scheduler=scheduler_instance,
                    optimizer=self.get_optimizer(True, lr),
                ),
            ]
        )

    @cached_property
    def optimizer_list(self):
        optimizer_list = []
        for lr, scheduler_instance in product(self.lr, self.scheduler_instance_list):
            optimizer_list.append(self.prepare_one_optimizer(lr, scheduler_instance))
        return optimizer_list


@configuration
class ValidationSample:
    seed: int = 123
    size: int = 200  # negative means no validation


@configuration()
class Indexation(LauncherSpecification):
    requirements: str = "duration=6 days & cpu(cores=8)"

    max_docs: int = 0
    """Maximum number of indexed documents – should be 0 when not debugging"""


@configuration()
class Preprocessing:
    requirements: str = "duration=12h & cpu(cores=12)"


@configuration()
class Retrieval:
    # better put the k value here for first stage
    batch_size: int = 256
    requirements: str = "duration=5 days & cuda(mem=24G)"

    # evaluation dataset
    eval_on_beir: bool = False
    """If true, we evaluate on msmarco + beir
    else, we evaluate on msmarco + lotte search
    """

    eval_on_mcl: bool = False
    """If true, for the multilingual settings,
    we evaluate also on miracl"""


@configuration()
class SAETrainingConfig:
    """Most of the configs are using list as they are the looping parameters"""

    # Base model basics
    hf_id: List[str] = ["distilbert/distilbert-base-uncased"]
    """Identifier for the base model"""

    layers: List[Union[int, str]] = [-1]
    """The layers for the SAE training
    -1 means the full layers; if contains "h", means using the full layers + the
    transform projection of the MLM head.
    """

    # SAE basics
    topk_style: str = "TopK"
    """The hyperparameters for the TopK SAE,
    values in TopK, HTopK, MTopK for the moment"""

    ks: List[int] = [8]
    """The list of the k we use to train the SAE models"""

    sae_width_list: List[int] = [65536]
    """The options of the SAE training values"""

    mat_range_sets: List[List[int]] = [[2048, 6144, 14336, 30720, 65536]]
    """The range of the Matryoshka embedding"""

    # SAE normalization option
    norm_sae_input_opt: List[bool] = [False]
    """Whether the input of the SAE is normalized"""

    norm_loss_opt: List[bool] = [False]
    """Whether each reconstructed vector and input vector are rescaled to the
    same magnitude for loss calculation. E.g. No matter the different input norm,
    the input vector is rescaled to norm 1 for every token vector"""

    # The training options
    rcst_coeff: List[float] = [1]
    """The hyperparameters for the reconstruction loss
    Normal value is 1.
    """

    aux_rcst_coeff: List[float] = [0.0625]
    """
    The hyperparamter for the auxillary reconstruction loss
    Normal value is 0.0625, for Hierarchical TopK it is 0.
    """

    sae_train_with_query_opt: List[bool] = [False]
    """During the SAE pretraining, do we still involve the query pretraining
    also?"""

    @property
    def rcst_hooks(self):
        rcst_hooks = []
        if self.topk_style == "TopK":
            for rcst, aux_multi, nl in product(
                self.rcst_coeff, self.aux_rcst_coeff, self.norm_loss_opt
            ):
                rcst_hooks.append(
                    SAEReconstructionRegu.C(
                        coeff=rcst, coeff_aux=rcst * aux_multi, normalize=nl
                    )
                    # .tag("rcst", rcst)
                    # .tag("aux", rcst * aux_multi)
                    .tag("norm_loss", nl)
                )
        elif self.topk_style == "HTopK":
            for rcst, aux_multi, nl in product(
                self.rcst_coeff, self.aux_rcst_coeff, self.norm_loss_opt
            ):
                rcst_hooks.append(
                    SAEHierarchicalReconstructionRegu.C(
                        coeff=rcst, coeff_aux=rcst * aux_multi, normalize=nl
                    )
                    # .tag("rcst", rcst)
                    # .tag("aux", rcst * aux_multi)
                    .tag("norm_loss", nl)
                )
        elif self.topk_style == "MTopK":
            for rcst, aux_multi, nl, mat_range in product(
                self.rcst_coeff,
                self.aux_rcst_coeff,
                self.norm_loss_opt,
                self.mat_range_sets,
            ):
                rcst_hooks.append(
                    SAEMatryoshkaReconstructionRegu.C(
                        coeff=rcst,
                        coeff_aux=rcst * aux_multi,
                        normalize=nl,
                        matryoshaka_range=mat_range,
                    )
                    # .tag("rcst", rcst)
                    # .tag("aux", rcst * aux_multi)
                    .tag("norm_loss", nl)
                )
        else:
            raise NotImplementedError
        return rcst_hooks


@configuration()
class SAESPLADETrainingConfig:
    """The SAE-SPLADE training configurations"""

    # training configs
    distil_data_path: str = "/home/zong/colbert_distillation/samples.json"

    nway: int = 8
    """the number of distillation samples for one query"""

    # regularization configs: much have the same length
    d_flops: List[float] = [0.04]
    q_flops: List[float] = [0.06]
    flops_lambda_warmup: int = 6000

    # k options for the SPLADE model
    override_k_option: List[int] = [2, 4, 8, 16, 32, 65536]
    """The override k that use for the SPLADE finetuning"""

    joint_remove: List[List[int]] = []
    """Define the pairs of the sae_k and splade_k need to be removed in the
    experimental plan to avoid using all the cross combinations"""

    @property
    def flop_hooks(self):
        # Later if we want to use the L1 regu on queries, we can also
        # adapt here
        flop_hooks = []
        for q_coeff, d_coeff in zip(self.q_flops, self.d_flops):
            flop_hooks.append(
                ScheduledFlopsRegularizer.C(
                    lambda_q=q_coeff,
                    lambda_d=d_coeff,
                    lambda_warmup_steps=self.flops_lambda_warmup,
                )
                .tag("q_flops", q_coeff)
                .tag("d_flops", d_coeff)
            )
        return flop_hooks


@configuration()
class TrainingConfig:
    ml_langs: List[str] = ["ar", "es", "fr", "ja", "ru", "zh"]
    sae: SAETrainingConfig = Factory(SAETrainingConfig)
    sae_splade: SAESPLADETrainingConfig = Factory(SAESPLADETrainingConfig)


@configuration()
class Learner:
    validation_top_k: int = field(default=100)
    validation_interval: int = field(default=32)
    sae_optimization: SAEOptimization = Factory(SAEOptimization)
    ir_optimization: SPLADESAEOptimization = Factory(SPLADESAEOptimization)
    requirements: str = "duration=4 days & cuda(mem=24G) * 2"
    sample_rate: float = 1.0
    """Sample rate for triplets"""
    sample_max: int = 0
    """Maximum number of samples considered (before shuffling). 0 for no limit."""


@configuration()
class SAESPLADEConfiguration(ConfigurationBase):
    """The SAE SPLADE training configurations
    First contains the SAE-training, then with the SAE-SPLADE training.
    """

    # misc
    gpu: bool = True
    """Use GPU for computation"""

    seed: int = 0
    """The seed used for experiments"""

    dev_test_size: int = 0
    """Development test size (0 to leave it like this)"""

    @attrs_cached_property
    def random(self):
        return Random.C(seed=self.seed)

    @attrs_cached_property
    def device(self) -> Device:
        return CudaDevice.C() if self.gpu else Device.C()

    train_config: TrainingConfig = Factory(TrainingConfig)
    validation: ValidationSample = Factory(ValidationSample)
    indexation: Indexation = Factory(Indexation)
    learning: Learner = Factory(Learner)
    retrieval: Retrieval = Factory(Retrieval)
    preprocessing: Preprocessing = Factory(Preprocessing)

    logging: List[str] = ["norm_sae"]
    """The tagging name of the final report"""
