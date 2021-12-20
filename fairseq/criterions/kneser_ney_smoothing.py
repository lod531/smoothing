# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pickle
import sys
import math
import os
import kenlm
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
class KneserNeySmoothingCriterionConfig(FairseqDataclass):
    kneser_d: float = field(
        default=0.0,
        metadata={"help": "0<=d<=1 constant discount count"},
    )   
    kneser_n: int = field(
        default=1,
        metadata={"help": "n-gram version of Kneser-Ney"},
    )   
    sentence_avg: bool = II("optimization.sentence_avg")


@register_criterion("kneser_ney_smoothing", dataclass=KneserNeySmoothingCriterionConfig)
class KneserNeySmoothingCriterion(FairseqCriterion):
    def __init__(self, task, kneser_d, kneser_n, sentence_avg):
        super().__init__(task)
        self.d = kneser_d
        self.n = kneser_n
        self.sentence_avg = sentence_avg
        self.dataset = crit_utils.get_dataset_from_task(task)
        self.dict_size = len(task.dictionary)
        self.ignored_indices = [self.padding_idx]
        self.fqs, self.N = crit_utils.get_fqs(self)
        self.empirical = self.get_empirical()
        cont_sets = defaultdict(set)
        self.tokenized = crit_utils.tokenize(self, self.dataset, self.n)
        self.contexts = crit_utils.get_contexts(self.tokenized)
        
        # alpha, gamma to keep track of alpha, gamma as in
        # https://dash.harvard.edu/bitstream/handle/1/25104739/tr-10-98.pdf page 17
        # An Empirical Study of Smoothing Techniques for Language Modeling
        self.gamma = defaultdict(set)
        self.psmooth1 = defaultdict(set)
        self.psmooth2 = defaultdict(set)
        self.filtered_u = defaultdict(lambda: defaultdict(int))

        self.alpha = defaultdict(lambda: defaultdict(int))
        self.gamma2 = defaultdict(set)
        self.psmooth_num = defaultdict(lambda: defaultdict(set))
        self.psmooth_denum = defaultdict(set)
        # code for manually calculating kneser-ney
        for token in self.tokenized:
            token = tuple(token)
            context = token[:-1]
            word = token[-1]
            self.alpha[context][word] += 1
            self.gamma2[context].add(word)
            # psmooth_numerator
            # this is tricky
            # We'll want to find the cases where the numerator is non-zero
            # for the p_smooth, given context (ish).
            # So We'll index in first using token[1:-1] to narrow the search
            # down to sets actually relevant, and then We'll iterate over
            # the non-zero sets which are determined by token[-1]
            self.psmooth_num[token[1:-1]][word].add(token[0])
            self.psmooth_denum[token[1:-1]].add(token[:1] + token[-1:])

            self.gamma[token[:-1]].add(token[-1:])
            self.psmooth1[token[1:]].add(token[:1])
            self.psmooth2[token[1:-1]].add(token[:1] + token[-1:])
            self.filtered_u[token[:-1]][token[-1]] += 1
        #self.ngrams = kenlm.Model("/cluster/home/andriusb/fq/kenlm/build/3gram.arpa")
        # so what, per context We'll need a 
        self.kl_terms = {}
        self.smoothed = {}
        for context in tqdm(self.contexts):
            self.kl_terms[context] = self.get_kl_terms(context)
        #print()
        #import os, psutil; print(psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2)

    def get_kl_terms(self, context):
        dist= defaultdict(float)
        #dist= torch.zeros(size=[self.dict_size], device=torch.device("cuda"))
        n_tokens = sum(self.alpha[context].values())
        for token, count in self.alpha[context].items():
            #dist[hash(token)] = (count-self.d)/n_tokens
            # same as doing -count 
            dist[token] = -self.d/n_tokens

        gamma = (self.d/n_tokens)*len(self.gamma2[context])
        #smooth denominator
        sm_denum = len(self.psmooth_denum[context[1:]])
        relevant_numerators = self.psmooth_num[context[1:]]
        for word, wset in relevant_numerators.items():
            dist[word] += gamma*len(wset)/sm_denum
        # at this point dist = smoothed - empirical
        pos_indices = []
        neg_indices = []
        pos_values = []
        neg_values = []
        for index, value in dist.items():
            if value > 0:
                pos_indices.append(index)
                pos_values.append(value)
            else:
                neg_indices.append(index)
                neg_values.append(value)

        indices = [list(dist.keys())]
        values = list(dist.values())

        idx_type = torch.long
        val_type = torch.bfloat16
        r_pos_val = torch.tensor(pos_values, device=torch.device("cuda"), dtype=val_type)
        r_neg_val = torch.tensor(neg_values, device=torch.device("cuda"), dtype=val_type)
        r_pos_idx = torch.tensor(pos_indices, device=torch.device("cuda"), dtype=idx_type)
        r_neg_idx = torch.tensor(neg_indices, device=torch.device("cuda"), dtype=idx_type)
        lambda_pos = torch.sum(r_pos_val)
        lambda_neg = torch.sum(r_neg_val)
        r_pos_val = r_pos_val/lambda_pos
        r_neg_val = r_neg_val/lambda_neg
        # handling edge case where
        if len(torch.nonzero(lambda_pos)) == 0:
            r_pos_val = torch.tensor([], device=torch.device("cuda"), dtype=val_type)
            r_pos_idx = torch.tensor([], device=torch.device("cuda"), dtype=idx_type)
        if len(torch.nonzero(lambda_neg)) == 0:
            r_neg_val = torch.tensor([], device=torch.device("cuda"), dtype=val_type)
            r_neg_idx = torch.tensor([], device=torch.device("cuda"), dtype=idx_type)
        #if torch.isnan(r_pos_val).any() or torch.isnan(r_neg_val).any() or torch.isnan(lambda_pos).any() or torch.isnan(lambda_neg).any():
        #    import pdb; pdb.set_trace()
        del dist
        del pos_indices
        del neg_indices
        del pos_values
        del neg_values
        return {"r_pos_val":r_pos_val, "r_neg_val":r_neg_val, 
                "r_pos_idx":r_pos_idx, "r_neg_idx":r_neg_idx,
                "lambda_pos":lambda_pos, "lambda_neg":lambda_neg}

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
        desired_size = lprobs.shape[:-1] + torch.Size([1])

        labels = torch.flatten(target).tolist()
        tokens = crit_utils.tokenize_tensor(self, labels, self.n)
        kl_loss = torch.tensor([0], device=torch.device("cuda")).float()
        for i in range(len(tokens)):
            token = tokens[i]
            context = tuple(token[:-1])
            if context in self.contexts:
                kl_stuff = self.kl_terms[context]
                r_pos = kl_stuff["r_pos_val"]
                r_neg = kl_stuff["r_neg_val"]
                r_pos_idx = kl_stuff["r_pos_idx"]
                r_neg_idx = kl_stuff["r_neg_idx"]

                lprob = lprobs[i,:]
                kl_pos = -r_pos*lprob[r_pos_idx]
                kl_neg = -r_neg*lprob[r_neg_idx]
                kl_pos = kl_pos * kl_stuff["lambda_pos"]
                kl_neg = kl_neg * kl_stuff["lambda_neg"]

                kl_loss += torch.sum(kl_pos) + torch.sum(kl_neg)
            
        torch_nll = F.nll_loss(
            lprobs,
            target,
            ignore_index=self.padding_idx,
            reduction="sum",
        )
        loss = torch_nll + kl_loss
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


#    # code using kenlm
#    def get_smoothed_ken(self, context):
#        dist= torch.zeros(size=[self.dict_size], requires_grad=False, device=torch.device("cuda"))
#        for i in range(self.dict_size):
#            bos=False
#            eos=False
#            sentence = self.task.dictionary.string(torch.tensor(context + (i,)))
#            if context[0] == self.task.dictionary.bos():
#                bos=True
#            if i == self.task.dictionary.eos():
#                eos=True
#            #if len(sentence.split()) != self.n and not (eos or bos):
#            if len(sentence.split()) != self.n or bos or eos :
#                pass
#            *_, last = self.ngrams.full_scores(sentence, bos=bos, eos=eos)
#            dist[i] = 10**last[0]
#        return dist

#        # old kneser code
#        #dist= torch.zeros(size=[self.dict_size], requires_grad=False)
#        dist= torch.zeros(size=[self.dict_size], device=torch.device("cuda"))
#        for token, count in self.filtered_u[context].items():
#            dist[token] = count-self.d
#
#        n_tokens = sum(self.filtered_u[context].values())
#        dist = dist/n_tokens
#        # number of unique words given context
#        uc = len(self.gamma[context])
#        for i in range(self.dict_size):
#            num = len(self.psmooth1[context[1:] + (i,)])
#            denom = len(self.psmooth2[tuple(context[1:])])
#            if num == 0: 
#                pass
#            else:
#                dist[i] += self.d/n_tokens*uc*num/denom
#
#
