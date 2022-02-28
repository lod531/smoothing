# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pickle
import sys
import math
import os
import fairseq.criterions.utils as crit_utils
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
class GoodTuringSmoothingCriterionConfig(FairseqDataclass):
    good_turing_n: int = field(
        default=1,
        metadata={"help": "n-gram version of Kneser-Ney"},
    )
    sentence_avg: bool = II("optimization.sentence_avg")


@register_criterion("good_turing_smoothing", dataclass=GoodTuringSmoothingCriterionConfig)
class GoodTuringSmoothingCriterion(FairseqCriterion):
    def __init__(self, task, good_turing_n, sentence_avg):
        super().__init__(task)
        self.n = good_turing_n
        self.sentence_avg = sentence_avg
        self.dataset = crit_utils.get_dataset_from_task(task)
        self.dict_size = len(task.dictionary)
        self.ignored_indices = [self.padding_idx]
        self.fqs, self.N = crit_utils.get_fqs(self)
        self.scc = crit_utils.get_scc(self.fqs)
        #self.empirical = self.get_empirical()
        self.tokenized = crit_utils.tokenize(self, self.dataset, self.n)
        self.contexts = crit_utils.get_contexts(self.tokenized)
        self.counts = defaultdict(lambda: defaultdict(int))
        self.ngram_probs = {}
        self.empirical = crit_utils.get_ngram_stats(self, self.dataset, self.n)
        #self.backoff = crit_utils.get_ngram_stats(self, self.dataset, self.n-1)
        for token in self.tokenized:
            token = tuple(token)
            context = token[:-1]
            word = token[-1]
            self.counts[context][word] += 1
        #hist = defaultdict(int)
        #stats = defaultdict(int)
        #for context in self.contexts:
        #    counts = self.counts[context]
        #    for count in counts.values():
        #        hist[count] += 1
        #    stats[sum(counts.values())] += 1

        #test = 0
        #test2 = 0
        #for i in range(1, 10):
        #    test += hist[i]
        #for i in range(1, 100):
        #    test2 += stats[i]

        self.kl_terms = {}
        self.smoothed_count = 0
        for context in tqdm(self.contexts):
            self.kl_terms[context] = self.get_kl_terms(context)
        print("SMOOTHED CONTEXTS")
        print(self.smoothed_count)
        print("TOTAL CONTEXTS")
        print(len(self.contexts))

    def get_kl_terms(self, context):
        #test = 0
        #for token, count in self.counts[context].items():
        #    if count == 1:
        #        test += 1
        #counts = self.counts[context]
        #scc = defaultdict(int)
        #for _, count in counts.items():
        #    scc[count] += 1
        #import pdb; pdb.set_trace()

        dist, p0 = simpleGoodTuringProbs(self.counts[context])
        idx_type = torch.long
        val_type = torch.float16
        if p0 == 0.0 or 1.0:
            idx = torch.tensor([], device=torch.device("cuda"), dtype=idx_type)
            val = torch.tensor([], device=torch.device("cuda"), dtype=val_type)
        else:
            self.smoothed_count += 1
            dist[self.task.dictionary.unk()] += p0
            for token, prob in dist.items():
                dist[token] = prob - self.empirical[context][token]
            indices = list(dist.keys())
            values = list(dist.values())
            idx = torch.tensor(indices, device=torch.device("cuda"), dtype=idx_type)
            val = torch.tensor(values, device=torch.device("cuda"), dtype=val_type)
        return {"val":val, "idx":idx}

    def get_empirical(self):
        empirical = torch.zeros(size=[self.dict_size], device=torch.device("cuda"))
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
        }
        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = model.get_targets(sample, net_output).view(-1)

        labels = torch.flatten(target).tolist()
        tokens = crit_utils.tokenize_tensor(self, labels, self.n)
        coeffs = torch.zeros(size=lprobs.shape, device=torch.device("cuda"), dtype=torch.float16)
        for i in range(len(tokens)):
            token = tokens[i]
            context = tuple(token[:-1])
            if context in self.contexts:
                kl_stuff = self.kl_terms[context]
                coeffs[i, kl_stuff["idx"]] = kl_stuff["val"]
            coeffs[i, labels[i]] += 1
        loss = torch.sum(coeffs * (-lprobs))
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

# Copyright 2009-2011 by Max Bane
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""
This module provides an implementation of Gale and Sampson's (1995/2001) "Simple
Good Turing" algorithm. The main function is simpleGoodTuringProbs(), which
takes a dictionary of species counts and returns the estimated population
frequencies of the species, as estimated by the Simple Good Turing method. To
use this module, you must have scipy and numpy installed.

Also included is a function that uses pylab and matplotlib to draw a useful
scatterplot for comparing the empirical frequencies against the Simple Good
Turing estimates.

Depends on reasonably recent versions of scipy and numpy.

Version 0.3: June 21, 2011
    First github version.

Version 0.2: November 12, 2009. 
    Added __version__ string.
    Added check for 0 counts.
    Don't pollute namespace with "import *".
    Added loglog keyword argument to plotFreqVsGoodTuring().
Version 0.1: November 11, 2009.

REFERENCES:
    William Gale and Geoffrey Sampson. 1995. Good-Turing frequency estimation
    without tears. Journal of Quantitative Linguistics, vol. 2, pp. 217--37.
    
    See also the corrected reprint of same on Sampson's web site.
"""

__version__ = "0.3"

from scipy import linalg
from numpy import c_, exp, log, inf, NaN, sqrt

def simpleGoodTuringProbs(counts, confidenceLevel=1.96):
    """
    Given a dictionary mapping keys (species) to counts, returns a dictionary
    mapping those same species to their smoothed probabilities, according to
    Gale and Sampson's (1995/2001 reprint) "Simple Good-Turing" method of
    smoothing. The optional confidenceLevel argument should be a multiplier of
    the standard deviation of the empirical Turing estimate (default 1.96,
    corresponding to a 95% confidence interval), a parameter of the algorithm
    that controls how many datapoints are smoothed loglinearly (see Gale and
    Sampson 1995).
    """
    # Gale and Sampson (1995/2001 reprint)
    if 0 in list(counts.values()):
        raise ValueError('Species must not have 0 count.')
    totalCounts = float(sum(counts.values()))   # N (G&S)
    countsOfCounts = defaultdict(int)
    for token, count in counts.items():
        countsOfCounts[count] += 1
    sortedCounts = sorted(countsOfCounts.keys())
    assert(totalCounts == sum([r*n for r,n in countsOfCounts.items()]))

    p0 = countsOfCounts[1] / totalCounts
    #print('p0 = %f' % p0)

    Z = __sgtZ(sortedCounts, countsOfCounts)

    # Compute a loglinear regression of Z[r] on r
    rs = list(Z.keys())
    zs = list(Z.values())
    a, b = __loglinregression(rs, zs)

    # Gale and Sampson's (1995/2001) "simple" loglinear smoothing method.
    rSmoothed = {}
    useY = False
    for r in sortedCounts:
        # y is the loglinear smoothing
        y = float(r+1) * exp(a*log(r+1) + b) / exp(a*log(r) + b)

        # If we've already started using y as the estimate for r, then
        # contine doing so; also start doing so if no species was observed
        # with count r+1.
        if r+1 not in countsOfCounts:
            if not useY:
                #print('Warning: reached unobserved count before crossing the '\
                #      'smoothing threshold.')
                pass
            useY = True

        if useY:
            rSmoothed[r] = y
            continue

        # x is the empirical Turing estimate for r
        x = (float(r+1) * countsOfCounts[r+1]) / countsOfCounts[r]

        Nr = float(countsOfCounts[r])
        Nr1 = float(countsOfCounts[r+1])

        # t is the width of the 95% (or whatever) confidence interval of the
        # empirical Turing estimate, assuming independence.
        t = confidenceLevel * \
            sqrt(\
                float(r+1)**2 * (Nr1 / Nr**2) \
                              * (1. + (Nr1 / Nr))\
            )

        # If the difference between x and y is more than t, then the empirical
        # Turing estimate x tends to be more accurate. Otherwise, use the
        # loglinear smoothed value y.
        if abs(x - y) > t:
            rSmoothed[r] = x
        useY = True
        rSmoothed[r] = y

    # normalize and return the resulting smoothed probabilities, less the
    # estimated probability mass of unseen species.
    sgtProbs = defaultdict(float)
    smoothTot = 0.0
    for r, rSmooth in rSmoothed.items():
        smoothTot += countsOfCounts[r] * rSmooth
    for species, spCount in counts.items():
        sgtProbs[species] = (1.0 - p0) * (rSmoothed[spCount] / smoothTot)

    return sgtProbs, p0

def __sgtZ(sortedCounts, countsOfCounts):
    # For each count j, set Z[j] to the linear interpolation of i,j,k, where i
    # is the greatest observed count less than i and k is the smallest observed
    # count greater than j.
    Z = {}
    for (jIdx, j) in enumerate(sortedCounts):
        if jIdx == 0:
            i = 0
        else:
            i = sortedCounts[jIdx-1]
        if jIdx == len(sortedCounts)-1:
            k = 2*j - i
        else:
            k = sortedCounts[jIdx+1]
        Z[j] = 2*countsOfCounts[j] / float(k-i)
    return Z

def __loglinregression(rs, zs):
    coef = linalg.lstsq(c_[log(rs), (1,)*len(rs)], log(zs))[0]
    a, b = coef
    #print('Regression: log(z) = %f*log(r) + %f' % (a,b))
    if a > -1.0:
        #print('Warning: slope is > -1.0')
        pass
    return a, b

