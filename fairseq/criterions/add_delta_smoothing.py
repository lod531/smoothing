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
        relevant_alphas = self.alphas[flat_samples]
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
        kl_pos = torch.sum(kl_pos, dim=1)
        kl_neg = torch.sum(kl_neg, dim=1)
        kl_loss = kl_pos + kl_neg
        # scale by the appropriate alphas via an element-wise multiply
        kl_loss = kl_loss * relevant_alphas
        # reduce everything down to a scalar
        loss = F.nll_loss(
            lprobs,
            target,
            ignore_index=self.padding_idx,
            reduction="sum" if reduce else "none",
        )

        loss = loss + torch.sum(kl_loss)
        #neg_losses = torch.sum(kl_neg, dim=1)
        #pos_losses = torch.sum(kl_neg, dim=1)
        #loss = kl_loss + torch.sum(kl_emp)

        return loss, loss

    def get_counts(self, dataset):
        fqs = defaultdict(int)
        print("Calculating Good-Turing stats:")
        for sentence in tqdm(dataset):
            for token in sentence:
                fqs[token.item()] += 1

        max_token = max(list(fqs.keys()))
        voc_size = len(fqs.keys())
        N = sum(fqs.values())

        # for testing purposes
        empirical = torch.cuda.FloatTensor(size=[max_token+1])
        for token, fq in fqs.items():
            empirical[token] = fq
        self.empirical = empirical/N

        lambda_neg = torch.cuda.FloatTensor([0])
        lambda_pos = torch.cuda.FloatTensor([0])

        r_pos = torch.cuda.FloatTensor(size=[max_token+1])
        r_neg = torch.cuda.FloatTensor(size=[max_token+1])

        C = torch.cuda.FloatTensor([1/(N+voc_size*self.delta)])

        for token, fq in fqs.items():
            h_w = (C-1/N)*fq + self.delta*C
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

        # now We need the a_j terms lol
        alphas = torch.cuda.FloatTensor(size=[max_token+1])
        for token, fq in fqs.items():
            if fq==0:
                alphas[token]=0
            else:
                alphas[token] = self.delta/(fq*((fq+self.delta)/(N+self.delta*voc_size)-fq/N))

        ## sanity check:
        normalizer = voc_size*self.delta + N
        for token, fq in fqs.items():
            temp = fq + fq*alphas[token]*(lambda_pos*r_pos[token]
                                                +lambda_neg*r_neg[token])
            test_prob = (temp/normalizer).item()
            true_prob = (fq+self.delta)/normalizer
            if test_prob/true_prob>1.01 or test_prob/true_prob<0.99:
                print(test_prob)
                print(true_prob)
                import pdb; pdb.set_trace()

        for token, fq in fqs.items():
            # must always equal self.delta
            test = +fq*alphas[token]*(lambda_pos*r_pos[token] + lambda_neg*r_neg[token])
            if self.delta/test > 1.01 or self.delta/test < 0.99:
                print(test.item())
                import pdb; pdb.set_trace()

        lbda = 0
        for token, fq in fqs.items():
            lbda += fq*(1+alphas[token]*(lambda_pos*r_pos[token]+lambda_neg*r_neg[token]))

        for token, fq in fqs.items():
            delta_prob = (fq+self.delta)/(voc_size*self.delta + N)
            test_prob = fq*(1+alphas[token]*(lambda_pos*r_pos[token]+lambda_neg*r_neg[token]))/lbda

            if test_prob/delta_prob >1.01 or test_prob/delta_prob<0.99:
                import pdb; pdb.set_trace()
                print("delta_prob")
                print(delta_prob)
                print("test_prob")
                print(test_prob.item())

        # eq 23
        for token, fq in fqs.items():
            test_prob = fq*(1+alphas[token]*(lambda_pos*r_pos[token] + lambda_neg*r_neg[token]))/lbda
            delta_prob = (fq+self.delta)/(voc_size*self.delta + N)
            if test_prob > delta_prob+0.001 or test_prob < delta_prob-0.001:
                print(test_prob.item())
                print(true_prob)
                import pdb; pdb.set_trace()

        #print("all passed")

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


