from dataclasses import dataclass
from pathlib import Path
import json
import logging
import os
from typing import NamedTuple, Type, Union

import torch
import torch.nn as nn
from experimaestro import Param
from transformers import AutoConfig, PreTrainedModel, AutoModel
from transformers.models.bert import BertModel
from transformers.models.distilbert import DistilBertModel
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions
from transformers.utils import cached_file

from xpmir.learning.optim import ModuleInitMode, ModuleInitOptions
from xpmir.text.huggingface.base import HFModelConfigFromId

HFConfigName = Union[str, os.PathLike]


@dataclass
class TransformersColBERTOutput(BaseModelOutputWithPoolingAndCrossAttentions):
    """Also include the orginal bert model's output before projection"""

    bert_last_hidden_state: torch.FloatTensor = None


class ColBERTConfig(NamedTuple):
    """ColBERT configuration when loading a pre-trained ColBERT model"""

    dim: int
    query_maxlen: int
    similarity: str
    attend_to_mask_tokens: bool
    data: dict

    @staticmethod
    def from_pretrained(pretrained_model_name_or_path: HFConfigName, out_dim: int):
        resolved_config_file = cached_file(
            pretrained_model_name_or_path, "artifact.metadata"
        )
        config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
        with open(resolved_config_file, "rt") as fp:
            data = json.load(fp)
            kwargs = {
                key: data[key]
                for key in ColBERTConfig._fields
                if key != "data" and key != "dim"
            }
            config.colbert = ColBERTConfig(**kwargs, data=data, dim=out_dim)
        return config


class DistilColBERTConfig(NamedTuple):
    """ColBERT configuration when training from scratch"""

    dim: int

    @staticmethod
    def from_pretrained(pretrained_model_name_or_path: HFConfigName):
        config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
        config.colbert = DistilColBERTConfig(dim=128)
        return config


class ColBERTModel(PreTrainedModel):
    """ColBERT model"""

    DEFAULT_OUTPUT_SIZE = 128

    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        self.linear = nn.Linear(config.hidden_size, config.colbert.dim, bias=False)

    def forward(self, ids, **kwargs):
        output = self.bert(ids, **kwargs)
        bert_last_hidden_state = output.last_hidden_state
        output.last_hidden_state = self.linear(output.last_hidden_state)
        return TransformersColBERTOutput(
            bert_last_hidden_state=bert_last_hidden_state,
            **output,
        )

    @classmethod
    def from_config(cls, config):
        return super(ColBERTModel, cls)._from_config(config)


class DistilColBERTModel(PreTrainedModel):
    """ColBERT model"""

    DEFAULT_OUTPUT_SIZE = 128

    def __init__(self, config):
        super().__init__(config)
        self.distilbert = DistilBertModel(config)
        self.linear = nn.Linear(config.hidden_size, config.colbert.dim, bias=False)

    def forward(self, ids, **kwargs):
        output = self.distilbert(ids, **kwargs)
        bert_last_hidden_state = output.last_hidden_state.clone()
        output.last_hidden_state = self.linear(output.last_hidden_state)
        return TransformersColBERTOutput(
            bert_last_hidden_state=bert_last_hidden_state,
            **output,
        )

    @classmethod
    def from_config(cls, config):
        return super(DistilColBERTModel, cls)._from_config(config)


class HFColBERTConfigFromId(HFModelConfigFromId):
    out_dim: Param[int] = 128

    def get_config(
        self,
        options: ModuleInitOptions,
        autoconfig: Type[AutoModel],
        automodel: Type[AutoConfig],
    ):
        model_id_or_path = self.model_id

        # Use saved models
        if model_path := os.environ.get("XPMIR_TRANSFORMERS_CACHE", None):
            path = (
                Path(model_path)
                / Path(f"{automodel.__module__}.{automodel.__qualname__}")
                / Path(self.model_id)
            )
            if path.is_dir():
                logging.warning("Using saved model from %s", path)
                model_id_or_path = path
            else:
                logging.warning(
                    "Could not find saved model in %s, using HF loading", path
                )

        # Load the model configuration
        config = autoconfig.from_pretrained(model_id_or_path, out_dim=self.out_dim)

        # Return it
        return config, model_id_or_path

    def __call__(
        self,
        options: ModuleInitOptions,
        autoconfig: Type[AutoConfig],
        automodel: Type[AutoModel],
    ):
        config, model_id_or_path = self.get_config(options, autoconfig, automodel)

        if options.mode == ModuleInitMode.NONE or options.mode == ModuleInitMode.RANDOM:
            logging.info("Random initialization of HF model")
            return config, automodel.from_config(config)

        logging.info(
            "Loading model from HF (%s) with model %s.%s",
            self.model_id,
            automodel.__module__,
            automodel.__name__,
        )
        return config, automodel.from_pretrained(
            model_id_or_path, config=config, ignore_mismatched_sizes=True
        )
