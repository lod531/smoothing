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
        # shape of (#tokens in batch, self.max_token+1)
        # in other words each row gives predicted probabilities
        # for entire vocabulary
        lprobs = lprobs.view(-1, lprobs.size(-1))
        # target shape = (#of tokens in batch, |V|)
        target = model.get_targets(sample, net_output).view(-1)
        kl_lprobs = lprobs[:, self.non_zero_indexes]
        # #tokens in batch, 1
        # the 1 is there for torch.repeat
        desired_size = lprobs.shape[:-1] + torch.Size([1])

        r_pos_repeated = self.r_pos.repeat(desired_size)
        r_neg_repeated = self.r_neg.repeat(desired_size)

        # flatten the observed samples for indexing
        flat_samples = torch.flatten(sample["target"])
        # to double check the order of P and Q in kl_dv
        # see https://pytorch.org/docs/master/generated/torch.nn.KLDivLoss.html#torch.nn.KLDivLoss
        # target = y_true, so it's KL(target || input), so
        # input = model distribution
        #kl_emp = F.kl_div(
        #        input = kl_lprobs,
        #        target = repeated_empirical,
        #        reduction="none")

        repeated_empirical = self.empirical.repeat(desired_size)
        kl_emp = -repeated_empirical * kl_lprobs


        kl_pos = -r_pos_repeated*kl_lprobs
        kl_neg = -r_neg_repeated*kl_lprobs

        kl_pos = kl_pos * self.lambda_pos
        kl_neg = kl_neg * self.lambda_neg


        kl_loss = kl_pos + kl_neg
        #loss = F.nll_loss(
        #    lprobs,
        #    target,
        #    ignore_index=self.padding_idx,
        #    reduction="none",
        #)
        # select the negative log probabilities according to
        # which tokens have occurrerd
        test = -lprobs[range(lprobs.shape[0]), flat_samples]
        #loss = torch.sum(kl_emp) + torch.sum(kl_loss)
        #loss = torch.sum(test) + torch.sum(kl_loss)
        loss = torch.sum(test)

        return loss, loss

    def get_counts(self, dataset):
        fqs = defaultdict(int)
        print("Calculating frequency stats:")
        for sentence in tqdm(dataset):
            for token in sentence:
                fqs[token.item()] += 1

        # will need this since it's the shape of model parameters
        # in the unigram case
        max_token = max(list(fqs.keys()))
        voc_size = len(fqs.keys())
        N = sum(fqs.values())

        # for testing purposes
        empirical = torch.zeros(size=[max_token+1], device=torch.device("cuda"))
        for token, fq in fqs.items():
            empirical[token] = fq
        self.empirical = empirical/N

        lambda_neg = torch.cuda.FloatTensor([0])
        lambda_pos = torch.cuda.FloatTensor([0])

        r_pos = torch.zeros(size=[max_token+1], device=torch.device("cuda"))
        r_neg = torch.zeros(size=[max_token+1], device=torch.device("cuda"))

        #C = torch.cuda.FloatTensor([1/(N+voc_size*self.delta)])
        for token, fq in fqs.items():
            #h_w = (C-1/N)*fq + self.delta*C
            h_w = (fq+self.delta)/(N+voc_size*self.delta) - fq/N
            #print("add-delta " + str(add_delta_probs[token]))
            #print("h_w + p: " + str((h_w + fq/N).item()))
            if h_w > 0:
                r_pos[token] = h_w
            else:
                r_neg[token] = h_w
        lambda_pos = torch.sum(r_pos)
        lambda_neg = torch.sum(r_neg)
        r_pos = r_pos/lambda_pos
        r_neg = r_neg/lambda_neg

        # which indexes of the unigram weight vector correspond
        # to words which actually occur?
        self.non_zero_indexes = list(fqs.keys())

        self.empirical = self.empirical[self.non_zero_indexes]
        self.lambda_pos = lambda_pos
        self.lambda_neg = lambda_neg
        # ignore words that don't occur since optimization will
        # dump probability into those to minimize the loss
        # due to negative KL term in lambda_neg
        self.r_pos = r_pos[self.non_zero_indexes]
        self.r_neg = r_neg[self.non_zero_indexes]
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


#        # now We need the a_j terms lol
#        alphas = torch.zeros(size=[max_token+1], device=torch.device("cuda"))
#        for token, fq in fqs.items():
#            if fq==0:
#                alphas[token]=0
#            else:
#                alphas[token] = self.delta/(fq*((fq+self.delta)/(N+self.delta*voc_size)-fq/N))
#
#        ## sanity check:
#        normalizer = voc_size*self.delta + N
#        for token, fq in fqs.items():
#            temp = fq + fq*alphas[token]*(lambda_pos*r_pos[token]
#                                                +lambda_neg*r_neg[token])
#            test_prob = (temp/normalizer).item()
#            true_prob = (fq+self.delta)/normalizer
#            if test_prob/true_prob>1.01 or test_prob/true_prob<0.99:
#                print(test_prob)
#                print(true_prob)
#                import pdb; pdb.set_trace()
#
#        for token, fq in fqs.items():
#            # must always equal self.delta
#            test = +fq*alphas[token]*(lambda_pos*r_pos[token] + lambda_neg*r_neg[token])
#            if self.delta/test > 1.01 or self.delta/test < 0.99:
#                print(test.item())
#                import pdb; pdb.set_trace()
#
#        lbda = 0
#        for token, fq in fqs.items():
#            lbda += fq*(1+alphas[token]*(lambda_pos*r_pos[token]+lambda_neg*r_neg[token]))
#
#        for token, fq in fqs.items():
#            delta_prob = (fq+self.delta)/(voc_size*self.delta + N)
#            test_prob = fq*(1+alphas[token]*(lambda_pos*r_pos[token]+lambda_neg*r_neg[token]))/lbda
#
#            if test_prob/delta_prob >1.01 or test_prob/delta_prob<0.99:
#                import pdb; pdb.set_trace()
#                print("delta_prob")
#                print(delta_prob)
#                print("test_prob")
#                print(test_prob.item())
#
#        # eq 23
#        for token, fq in fqs.items():
#            test_prob = fq*(1+alphas[token]*(lambda_pos*r_pos[token] + lambda_neg*r_neg[token]))/lbda
#            delta_prob = (fq+self.delta)/(voc_size*self.delta + N)
#            if test_prob > delta_prob+0.001 or test_prob < delta_prob-0.001:
#                print(test_prob.item())
#                print(true_prob)
#                import pdb; pdb.set_trace()
#
#
