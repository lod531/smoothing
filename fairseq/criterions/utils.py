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
from fairseq.tasks.translation import TranslationTask
from fairseq.tasks.language_modeling import LanguageModelingTask
from omegaconf import II


def get_fqs(crit):
    fqs = defaultdict(int)
    print("Calculating frequency stats:")
    for sentence in tqdm(crit.dataset):
        for token in sentence:
            fqs[token.item()] += 1
    return fqs, sum(fqs.values())

def get_scc(fqs):
    scc = defaultdict(int)
    for token, count in fqs.items():
        scc[count] += 1
    return scc

def get_dataset_from_task(task):
    task.load_dataset("train")
    if isinstance(task, TranslationTask):
        dataset = task.datasets["train"].tgt
    elif isinstance(task, LanguageModelingTask):
        dataset = task.datasets["train"].dataset.dataset
    return dataset

def get_kl_terms(crit, smoothed, empirical):
    #ignored_indices = [crit.padding_idx]
    h_w = smoothed - empirical

    lambda_neg = torch.cuda.FloatTensor([0])
    lambda_pos = torch.cuda.FloatTensor([0])

    r_pos = torch.zeros(size=[crit.dict_size], device=torch.device("cuda"))
    r_neg = torch.zeros(size=[crit.dict_size], device=torch.device("cuda"))

    r_pos[h_w >= 0] = h_w[h_w >= 0]
    r_neg[h_w < 0] = h_w[h_w < 0]
    lambda_pos = torch.sum(r_pos)
    lambda_neg = torch.sum(r_neg)
    r_pos = r_pos/lambda_pos
    r_neg = r_neg/lambda_neg

    return {"lambda_pos":lambda_pos, "lambda_neg":lambda_neg, 
            "r_pos":r_pos, "r_neg":r_neg, "empirical":empirical}
    #crit.lambda_pos = lambda_pos
    #crit.lambda_neg = lambda_neg
    #crit.r_pos = r_pos
    #crit.r_neg = r_neg
    #crit.empirical = empirical

def lambda_loss(crit, model, net_output, sample, reduce=True):
    lprobs = model.get_normalized_probs(net_output, log_probs=True)
    lprobs = lprobs.view(-1, lprobs.size(-1))
    target = model.get_targets(sample, net_output).view(-1)
    # the 1 is there for torch.repeat
    desired_size = lprobs.shape[:-1] + torch.Size([1])

    r_pos_repeated = crit.r_pos.repeat(desired_size)
    r_neg_repeated = crit.r_neg.repeat(desired_size)

    kl_pos = -r_pos_repeated*lprobs
    kl_neg = -r_neg_repeated*lprobs

    kl_pos = kl_pos * crit.lambda_pos
    kl_neg = kl_neg * crit.lambda_neg

    kl_loss = kl_pos + kl_neg
    torch_nll = F.nll_loss(
        lprobs,
        target,
        ignore_index=crit.padding_idx,
        reduction="none",
    )
    loss = torch.sum(torch_nll) + torch.sum(kl_loss)
    return loss

def tokenize(crit, dataset, n):
    assert(n>0)
    res = []
    BOS = crit.dict.bos()
    for sentence in dataset:
        sentence = sentence.tolist()
        for end in range(len(sentence)):
            start = end-(n-1)
            # +1 because I want slices to include
            # the token at [end], but [start:end]
            # excludes end
            end = end+1
            if start < 0:
                pad = [BOS]*abs(start)
                token = pad + sentence[:end]
            else:
                token = sentence[start:end]
            res.append(token)
    return res

def get_ngram_stats(crit, dataset, n):
    assert(n>0)
    res = defaultdict(lambda: defaultdict(float))
    BOS = crit.dict.bos()
    print()
    print("gathering stats for n=" + str(n))
    for sentence in tqdm(dataset):
        sentence = sentence.tolist()
        for end in range(len(sentence)):
            start = end-(n-1)
            # +1 because I want slices to include
            # the token at [end], but [start:end]
            # excludes end
            end = end+1
            if start < 0:
                pad = [BOS]*abs(start)
                token = pad + sentence[:end]
            else:
                token = sentence[start:end]
            #res[context][word] += 1
            token = tuple(token)
            res[token[:-1]][token[-1]] += 1
    for context, counts in res.items():
        total = sum(counts.values())
        for word, count in counts.items():
            counts[word] = count/total
        assert sum(counts.values()) < 1.000001 and sum(counts.values()) > 0.99999
    return res

# crit is a criterion
# tens is the tensor
# n is the order, as in ngram
def tokenize_tensor(crit, tens, n):
    assert(n>0)
    res = []
    BOS = crit.dict.bos()
    for end in range(len(tens)):
        start = end-(n-1)
        # +1 because I want slices to include
        # the token at [end], but [start:end]
        # excludes end
        end = end+1
        if start < 0:
            pad = [BOS]*abs(start)
            token = pad + tens[:end]
        else:
            token = tens[start:end]
        res.append(token)
    return res

def get_contexts(data):
    # the input to this is expected to be the output of tokenize, basically.
    contexts = set()
    for token in data:
        contexts.add(tuple(token[:-1]))
    return contexts

def get_empirical(crit):
    empirical = torch.zeros(size=[crit.dict_size], device=torch.device("cuda"))
    for token, fq in fqs.items():
        if token not in ignored_indices:
            empirical[token] = fq
    empirical = empirical/torch.sum(empirical)
    return empirical

def filter_by_context(data, context):
# data - list of tokens
# context - tuple, as in (word, word, word)
    res_with_contexts = []
    res_unigram = []
    context = list(context)
    for token in data:
        if token[:-1] == context:
            res_with_contexts.append(token)
            res_unigram.append(token[-1])
    return res_with_contexts, res_unigram


