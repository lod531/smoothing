# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import nltk
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
class KatzSmoothingCriterionConfig(FairseqDataclass):
    katz_k: int = field(
        default=0,
        metadata={"help": "Tokens occuring up to and including k times will be smoothed."},
    )
    sentence_avg: bool = II("optimization.sentence_avg")




@register_criterion("katz_smoothing", dataclass=KatzSmoothingCriterionConfig)
class KatzSmoothingCriterion(FairseqCriterion):
    def __init__(self, task, katz_k, sentence_avg):
        super().__init__(task)
        self.k = katz_k
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
        # shape of (#tokens in batch, self.max_token+1)
        # in other words each row gives predicted probabilities
        # for entire vocabulary
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = model.get_targets(sample, net_output).view(-1)
        # #tokens in batch, 1
        # the 1 is there for torch.repeat
        desired_size = lprobs.shape[:-1] + torch.Size([1])

        # repeated_empirical = self.empirical.repeat(desired_size[-1])
        # uniform = torch.ones(size=[self.max_token+1],
        #                         device=torch.device("cuda")).float()
        # uniform = uniform/torch.sum(uniform)
        # 

        # uniform_repeated = uniform.repeat(desired_size)

        # kl_uniform_loss = F.kl_div(
        #         input = lprobs,
        #         target = uniform_repeated,
        #         reduction="sum" if reduce else "none")

        # kl_uniform_loss = kl_uniform_loss * (self.max_token * self.delta)/self.N

        #We want a KL loss per token

        # each row is an instance of r_pos (and r_neg lol) 
        # shape is (#tokens in batch, self.max_token+1)
        r_pos_repeated = self.r_pos.repeat(desired_size)
        r_neg_repeated = self.r_neg.repeat(desired_size)

        # flatten the observed samples for indexing
        flat_samples = torch.flatten(sample["target"])
        #number_of_samples = flat_samples.shape[0]
        # cum_kl_loss = torch.cuda.FloatTensor([0])
        # for i in range(0, number_of_samples):
        #     # token is the actual word that occurred
        #     token = flat_samples[i]
        #     # probs is the probabilities output by the model
        #     # given token has occurred
        #     probs = lprobs[i,:]
        #     kl_pos = F.kl_div(
        #             input=probs,
        #             target = self.r_pos,
        #             reduction="sum" if reduce else "none")
        #     kl_neg = F.kl_div(
        #             input=probs,
        #             target = self.r_neg,
        #             reduction="sum" if reduce else "none")
        #     kl_loss = kl_pos*self.lambda_pos + kl_neg*self.lambda_neg
        #     kl_loss = kl_loss * self.alphas[i]
        #     cum_kl_loss += kl_loss


        # to double check the order of P and Q in kl_dv
        # see https://pytorch.org/docs/master/generated/torch.nn.KLDivLoss.html#torch.nn.KLDivLoss
        # target = y_true, so it's KL(target || input), so
        # input = model distribution

        # empirical kl
        repeated_empirical = self.empirical.repeat(desired_size)
        kl_emp = F.kl_div(
                input = lprobs,
                target = repeated_empirical,
                reduction="none")


        # this should just go pair by pair of tensors from
        # lprobs and target, returning
        # target[i]*(log(target[i]) - lprobs[i])
        # kl_div won't apply log to lprobs[i] because it
        # expects lprobs to be already in log space,
        # hence lprobs instead of probs
        # also We have reduction="none" because
        # We need to apply different scaling factors
        # to losses belonging to different tokens
        # kl_pos[a, b] = 
        # r_pos_repeated[a, b] * (torch.log(r_pos_repeated[a, b]) - lprobs[a, b])
        kl_pos = F.kl_div(
                input = lprobs,
                target = r_pos_repeated,
                reduction="none")
        kl_neg = F.kl_div(
                input = lprobs,
                target = r_neg_repeated,
                reduction="none")

        # the lambda_pos and lambda_neg are constants
        # regardless of token, so We can multiply entire
        # matrices
        kl_pos = kl_pos * self.lambda_pos
        kl_neg = kl_neg * self.lambda_neg


        # the alpha_j depends on the word w_j, 
        # so We select from alphas using flat_samples
        # to get alphas of tokens which have occurred
        #relevant_alphas = self.alphas[flat_samples]
        # unsqueeze and expand so that the shape of
        # the kl losses and relevant_alphas is the same
        # this way relevant_alphas[i,:] = row vector of
        # alpha[token]
        #relevant_alphas = torch.unsqueeze(relevant_alphas, 1)
        #relevant_alphas = relevant_alphas.expand(-1, kl_pos.shape[-1])
        # can add kl losses since We just need to keep the per-token
        # kl losses distinct

        # sum rows across the first dimension
        # i.e. just sum over the rows.
        # result is a vector of size (# of tokens in batch)
        #kl_pos = torch.sum(kl_pos, dim=1)
        #kl_neg = torch.sum(kl_neg, dim=1)
        kl_loss = kl_pos + kl_neg
        # scale by the appropriate alphas via an element-wise multiply
        #kl_loss = kl_loss * relevant_alphas
        # reduce everything down to a scalar
        # nll_loss literally just returns -lprobs[token] lol
        loss = F.nll_loss(
            lprobs,
            target,
            ignore_index=self.padding_idx,
            reduction="none",
        )

        #loss = loss + torch.sum(kl_loss)
        #neg_losses = torch.sum(kl_neg, dim=1)
        #pos_losses = torch.sum(kl_neg, dim=1)
        #loss = torch.sum(kl_loss) + torch.sum(kl_emp)
        loss = torch.sum(loss) + torch.sum(kl_loss)

        return loss, loss

    def get_counts(self, dataset):
        fqs = defaultdict(int)
        print("Calculating Good-Turing stats:")
        #nltk_set = nltk.FreqDist()
        for sentence in tqdm(dataset):
            for token in sentence:
                fqs[token.item()] += 1
                #nltk_set.update([token.item()])

        max_token = max(list(fqs.keys()))
        voc_size = len(fqs.keys())
        N = sum(fqs.values())

        # good_turing probs, p0 is the amount of mass allocated to 
        # words with 0 frequency, and scc is smoothed
        # counts of counts
        gt_probs, p0 = simpleGoodTuringProbs(fqs)
        scc = defaultdict(int)
        for token, frequency in fqs.items():
            scc[frequency] += 1
        # gt_probs = nltk.probability.SimpleGoodTuringProbDist(nltk_set)

        empirical = torch.cuda.FloatTensor(size=[max_token+1])
        for token, fq in fqs.items():
            empirical[token] = fq
        self.empirical = empirical/N

        lambda_neg = torch.cuda.FloatTensor([0])
        lambda_pos = torch.cuda.FloatTensor([0])

        r_pos = torch.cuda.FloatTensor(size=[max_token+1])
        r_neg = torch.cuda.FloatTensor(size=[max_token+1])

        hws = {}
        for token, fq in fqs.items():
            if fq > self.k:
                pass
            else:
                # this is a mess and it kind of can't be helped.
                # eq 55 in overleaf
                rstar = (fq+1)*scc[fq+1]/scc[fq]
                ls = rstar/fq - (self.k+1)*scc[self.k+1]/scc[1]
                rs = 1/(1-(self.k+1)*scc[self.k+1]/scc[1])
                d_r = ls*rs
                h_w = (d_r-1)*fq/N
                hws[token] = h_w
                if h_w > 0:
                    r_pos[token] = h_w
                else:
                    r_neg[token] = h_w
        lambda_pos = torch.sum(r_pos)
        lambda_neg = torch.sum(r_neg)
        r_pos = r_pos/lambda_pos
        r_neg = r_neg/lambda_neg

        # now We need the a_j terms lol
        alphas = torch.cuda.FloatTensor(size=[max_token+1])
        for token, fq in fqs.items():
            if fq==0 or fq > self.k:
                alphas[token]=0
            else:
                numerator = hws[token]*N 
                denominator = fq*(lambda_pos*r_pos[token] + lambda_neg*r_neg[token])
                alphas[token] = numerator/denominator

        # for token, fq in fqs.items():
        #     if(fq in scc.keys() and (fq+1) in scc.keys()):
        #         left = (fq+1)*(scc[fq+1]/scc[fq])
        #         right = fq*(1+alphas[token]*(lambda_pos*r_pos[token] + lambda_neg*r_neg[token]))
        #         print(left)
        #         print(right.item())
        #         import pdb; pdb.set_trace()


        self.alphas = alphas
        self.lambda_pos = lambda_pos
        self.lambda_neg = lambda_neg
        self.r_pos = r_pos
        self.r_neg = r_neg
        self.N = N
        self.max_token = max_token
        self.voc_size = voc_size
        self.fqs = fqs


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
    print('p0 = %f' % p0)

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
                print('Warning: reached unobserved count before crossing the '\
                      'smoothing threshold.')
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
    sgtProbs = {}
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
    print('Regression: log(z) = %f*log(r) + %f' % (a,b))
    if a > -1.0:
        print('Warning: slope is > -1.0')
    return a, b


