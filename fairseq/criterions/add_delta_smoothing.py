# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass, field
from collections import defaultdict
from tqdm import tqdm
import fairseq.criterions.utils as crit_utils 

import torch
import torch.nn.functional as F
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from fairseq.tasks.translation import TranslationTask
from fairseq.tasks.language_modeling import LanguageModelingTask
from omegaconf import II


@dataclass
class AddDeltaCriterionConfig(FairseqDataclass):
    label_smoothing: float = field(
        default=0.0,
        metadata={"help": " a delta such that new_count = (old_count+delta)/(N+|V|*delta)"},
    )
    sentence_avg: bool = II("optimization.sentence_avg")


@register_criterion("add_delta_smoothing", dataclass=AddDeltaCriterionConfig)
class AddDeltaSmoothingCriterion(FairseqCriterion):
    def __init__(self, task, label_smoothing, sentence_avg):
        super().__init__(task)
        self.delta = label_smoothing
        self.sentence_avg = sentence_avg
        self.dataset = crit_utils.get_dataset_from_task(task)
        self.dict_size = len(task.dictionary)
        self.ignored_indices = [self.padding_idx]

        crit_utils.get_fqs(self)
        smoothed = torch.zeros(size=[self.dict_size], device=torch.device("cuda"))
        for token in range(self.dict_size):
            if token not in self.ignored_indices:
                smoothed[token] = self.fqs[token] + self.delta
        #smoothed[task.dictionary.unk] = 1 - torch.sum(smoothed)
        smoothed = smoothed/torch.sum(smoothed)
        self.smoothed = smoothed

        crit_utils.get_kl_terms(self)


    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample["net_input"], target=sample["target"])
        loss, _ = self.compute_loss(model, net_output, sample, reduce=reduce)
        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        logging_output = {
            "loss": loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }
        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, sample, reduce=True):
        loss = crit_utils.lambda_loss(self, model, net_output, sample, reduce=True)
        return loss, loss


    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        # we divide by log(2) to convert the loss from base e to base 2
        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        if sample_size != ntokens:
            metrics.log_scalar(
                "nll_loss", loss_sum / ntokens / math.log(2), ntokens, round=3
            )
            metrics.log_derived(
                "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
            )
        else:
            metrics.log_derived(
                "ppl", lambda meters: utils.get_perplexity(meters["loss"].avg)
            )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True


