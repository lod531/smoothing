# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass, field
from collections import defaultdict
from tqdm import tqdm

import torch
import torch.nn.functional as F
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
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
        task.load_dataset("train")
        dataset = task.datasets["train"].tgt
        self.get_counts(dataset)

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
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = model.get_targets(sample, net_output).view(-1)
        desired_size = lprobs.shape[:-1] + torch.Size([1])

        repeated_empirical = self.empirical.repeat(desired_size[-1])
        uniform = torch.ones(size=[self.max_token+1],
                                device=torch.device("cuda")).float()
        uniform = uniform/torch.sum(uniform)
        

        uniform_repeated = uniform.repeat(desired_size)

        kl_uniform_loss = F.kl_div(
                input = lprobs,
                target = uniform_repeated,
                reduction="sum" if reduce else "none")

        kl_uniform_loss = kl_uniform_loss * (self.max_token * self.delta)/self.N

        loss = F.nll_loss(
            lprobs,
            target,
            ignore_index=self.padding_idx,
            reduction="sum" if reduce else "none",
        )
        loss = loss + kl_uniform_loss
        return loss, loss

    def get_counts(self, dataset):
        fqs = defaultdict(int)
        print("Calculating Good-Turing stats:")
        for sentence in tqdm(dataset):
            for token in sentence:
                fqs[token.item()] += 1

        max_token = max(list(fqs.keys()))
        N = sum(fqs.values())

        empirical = torch.cuda.FloatTensor(size=[max_token+1])
        for token, fq in fqs.items():
            empirical[token] = fq
        self.empirical = empirical/N

        lambda_neg = torch.cuda.FloatTensor([0])
        lambda_pos = torch.cuda.FloatTensor([0])

        r_pos = torch.cuda.FloatTensor(size=[max_token+1])
        r_neg = torch.cuda.FloatTensor(size=[max_token+1])

        C = torch.cuda.FloatTensor([1/(N+max_token*self.delta)])
        for token, fq in fqs.items():
            h_w = (C-1/N)*fq + self.delta*C
            if h_w > 0:
                r_pos[token] = h_w
            else:
                r_neg[token] = h_w
        self.lambda_pos = torch.sum(r_pos)
        self.lambda_neg = torch.sum(r_neg)
        self.r_pos = r_pos/self.lambda_pos
        self.r_neg = r_neg/self.lambda_neg
        self.fqs = fqs
        self.N = N
        self.max_token = max_token

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
