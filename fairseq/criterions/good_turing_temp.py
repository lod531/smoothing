# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections import defaultdict
from tqdm import tqdm

import math
from dataclasses import dataclass

import torch.nn.functional as F
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from omegaconf import II


def get_good_turing_counts(dataset):
    fqs = defaultdict(int)
    print("Calculating Good-Turing stats:")
    for sentence in tqdm(dataset):
        for token in sentence:
            fqs[token.item()] += 1

    scc = defaultdict(int)
    for token, fq in fqs.items():
        scc[fq] += 1

    max_token = max(list(fqs.keys()))
    N = sum(fqs.values())

    delta = 10000
    add_delta_probs = {}
    for token, fq in fqs.items():
        add_delta_probs[token] = (fq+delta)/(N+delta*max_token)

    gt_probs, p0 = simpleGoodTuringProbs(fqs)

    hws = {}
    for token, gt_prob in gt_probs.items():
        hws[token] = gt_prob - fqs[token]/N

    yeet = 0
    for token, fq in fqs.items():
        gt_count = gt_probs[token]*N
        if gt_count > fq+50 or gt_count < fq-50:
            yeet += 1


    lambda_pos = 0
    lambda_neg = 0
    r_pos = {}
    r_neg = {}
    for token, hw in hws.items():
        if hw > 0:
            r_pos[token] = hw
            lambda_pos += hw
        else:
            r_neg[token] = hw
            lambda_neg += hw

    for token, pos in r_pos.items():
        r_pos[token] = pos/lambda_pos

    for token, neg in r_neg.items():
        r_neg[token] = neg/lambda_neg

    katz_probs = {}
    k = 5
    katz_total = 0
    katz_items = 0
    for token, fq in fqs.items():
        if fq > 5:
            katz_probs[token] = fq/N
        else:
            rstar = (fq+1)*scc[fq+1]/scc[fq]
            ls = rstar/fq - (k+1)*scc[k+1]/scc[1]
            rs = 1/(1-(k+1)*scc[k+1]/scc[1])
            d_r = ls*rs
            empirical = fq/N
            katz_probs[token] = d_r*fq/N
            katz_total += abs(katz_probs[token]-fq/N)
            katz_items += 1
    print(katz_total*3949114) 
    print(katz_items)
    import pdb;pdb.set_trace()
            

    #sanity_1 = sum(r_pos.values())
    #sanity_2 = sum(r_neg.values())
    #sanity_3 = min(r_pos.values())
    #sanity_4 = min(r_neg.values())
    #sanity_5 = sum(gt_probs.values()) + p0
    return {"fqs":fqs, "lambda_pos":lambda_pos, "lambda_neg":lambda_neg,
            "r_pos":r_pos, "r_neg":r_neg, "add_delta_probs":add_delta_probs,
            "gtp": gt_probs, "ktp": katz_probs}



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


