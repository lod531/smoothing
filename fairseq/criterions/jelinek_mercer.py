# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pickle
import sys
import math
import os
import fairseq.criterions.utils as crit_utils
from ast import literal_eval as make_tuple
from typing import List
from dataclasses import dataclass, field
from collections import defaultdict
from tqdm import tqdm
from fairseq.tasks.translation import TranslationTask
from fairseq.tasks.language_modeling import LanguageModelingTask


import torch
import torch.nn.functional as F
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from omegaconf import II

@dataclass
class JelinekMercerSmoothingCriterionConfig(FairseqDataclass):
    alphas: str = field(
        default="(1)", metadata={"help": "weights - alphas[0] = weight for unigram dist etc."}
    )
    jelinek_n: int = field(
        default=1,
        metadata={"help": "n-gram version of Kneser-Ney"},
    )
    sentence_avg: bool = II("optimization.sentence_avg")


@register_criterion("jelinek_mercer_smoothing", dataclass=JelinekMercerSmoothingCriterionConfig)
class JelinekMercerSmoothingCriterion(FairseqCriterion):
    def __init__(self, task, jelinek_n, alphas, sentence_avg):
        super().__init__(task)
        self.n = jelinek_n
        self.alphas = make_tuple(alphas)
        assert len(self.alphas) == self.n+1
        assert sum(self.alphas) == 1
        self.sentence_avg = sentence_avg
        self.dataset = crit_utils.get_dataset_from_task(task)
        self.dict_size = len(task.dictionary)
        self.ignored_indices = [self.padding_idx]
        self.fqs, self.N = crit_utils.get_fqs(self)
        self.unigram = self.get_empirical()
        self.uniform = torch.ones(size=[self.dict_size], device=torch.device("cuda"), dtype=torch.float)
        self.uniform = self.uniform/torch.sum(self.uniform)

        self.KL_div_uniform = 0
        self.KL_div_unigram = 0
        self.KL_n_terms = 0


        self.ngram_probs = {}
        for i in range(1, self.n):
            self.ngram_probs[i] = crit_utils.get_ngram_stats(self, self.dataset, i)
        self.get_kl_terms()

    def get_kl_terms(self):
        idx_type = torch.long
        val_type = torch.float16
        self.kl_terms = defaultdict(lambda: {})
        for i in range(1, self.n):
            print()
            print("transferring to GPU memory")
            for context, counts in tqdm(self.ngram_probs[i].items()):
                indices = list(counts.keys())
                values = list(counts.values())
                idx = torch.tensor(indices, device=torch.device("cuda"), dtype=idx_type)
                val = torch.tensor(values, device=torch.device("cuda"), dtype=val_type)
                self.kl_terms[i][context] = {"val":val, "idx":idx}

    def get_empirical(self):
        empirical = torch.zeros(size=[self.dict_size], device=torch.device("cuda"), dtype=torch.float)
        for token, fq in self.fqs.items():
            if token not in self.ignored_indices:
                empirical[token] = fq
        empirical = empirical/torch.sum(empirical)
        return empirical

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        #net_output = model(**sample["net_input"], target=sample["target"])
        net_output = model(**sample["net_input"])
        loss, _ = self.compute_loss(model, net_output, sample, reduce=reduce)
        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        logging_output = {
            "loss": loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
            "test": 2
        }
        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))


        uniform_tile = self.uniform.expand(lprobs.shape[0], -1)
        unigram_tile = self.unigram.expand(lprobs.shape[0], -1)
        uniform_loss = uniform_tile * (-lprobs)
        KL_div_uniform = F.kl_div(input=lprobs, target=uniform_tile, reduction="sum")
        KL_div_unigram = F.kl_div(input=lprobs, target=unigram_tile, reduction="sum")
        self.KL_div_uniform += KL_div_uniform.item()
        self.KL_div_unigram += KL_div_unigram.item()
        self.KL_n_terms += lprobs.shape[0]

        unigram_tile = self.alphas[1]*unigram_tile

        target = model.get_targets(sample, net_output).view(-1)
        nll_loss = F.nll_loss(
            lprobs,
            target,
            ignore_index=self.padding_idx,
            reduction="sum" if reduce else "none",
        )
        #nll_coefs = F.one_hot(torch.flatten(target), num_classes=lprobs.shape[-1])
        #loss = nll_loss*self.alphas[2] + KL_div_unigram*self.alphas[1] + KL_div_uniform*self.alphas[0]
        #nll_loss = (nll_coefs*(-lprobs)).sum()
        uniform_loss = -lprobs.sum()*(self.alphas[0]/lprobs.shape[-1])
        unigram_loss = (unigram_tile*(-lprobs)).sum()
        loss = nll_loss*self.alphas[2] + unigram_loss + uniform_loss
        #tokens = crit_utils.tokenize_tensor(self, labels, self.n)
        #coeffs = torch.zeros(size=lprobs.shape, device=torch.device("cuda"), dtype=torch.float16)
        #for i in range(len(tokens)):
        #    token = tokens[i]
        #    # 0-order backoff a.k.a. uniform
        #    coeffs[i, :] += self.alphas[0]*self.uniform
        #    for j in range(self.n-1):
        #        context = tuple(token[-1-j:-1])
        #        if context in self.kl_terms[j+1]:
        #            kl_stuff = self.kl_terms[j+1][context]
        #            coeffs[i, kl_stuff["idx"]] += self.alphas[j+1]*kl_stuff["val"]
        #    coeffs[i, labels[i]] += 1*self.alphas[-1]
        #loss = torch.sum(coeffs * (-lprobs))
        return loss, nll_loss


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

def get_size(obj, seen=None):
    """Recursively finds size of objects"""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])
    return size

