from fairseq.models import BaseFairseqModel, register_model

import torch

# Note: the register_model "decorator" should immediately precede the
# definition of the Model class.

@register_model('unigram')
class Unigram(BaseFairseqModel):

    def __init__(self, unique_items):
        super().__init__()

        self.softmax = torch.nn.Softmax(dim=0)
        self.logsoftmax = torch.nn.LogSoftmax(dim=0)
        self.weights = torch.nn.Parameter(torch.randn(size=[max(unique_items.values())+1], 
                                                    requires_grad=True, 
                                                    device=torch.device('cuda')))


        


    @classmethod
    def build_model(cls, args, task):
        # Fairseq initializes models by calling the ``build_model()``
        # function. This provides more flexibility, since the returned model
        # instance can be of a different type than the one that was called.
        # In this case we'll just return a SimpleLSTMModel instance.

        # Initialize our Encoder and Decoder.

        # Print the model architecture.
        task.load_dataset("train")
        dataset = task.datasets["train"].tgt 
        unique_tokens = {}
        for sentence in dataset:
            for token in sentence:
                unique_tokens[token.item()] = token.item()

        return Unigram(unique_tokens)

    # We could override the ``forward()`` if we wanted more control over how
    # the encoder and decoder interact, but it's not necessary for this
    # tutorial since we can inherit the default implementation provided by
    # the FairseqEncoderDecoderModel base class, which looks like:
    #
    # def forward(self, src_tokens, src_lengths, prev_output_tokens):
    #     encoder_out = self.encoder(src_tokens, src_lengths)
    #     decoder_out = self.decoder(prev_output_tokens, encoder_out)
    #     return decoder_out

    def forward(self, src_tokens, src_lengths, prev_output_tokens, **kwargs): 
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (LongTensor): lengths of each source sentence of shape
                `(batch)`

        Returns:
            dict:
                - **encoder_out** (tuple): a tuple with two elements, where the
                  first element is the last encoder layer's output and the
                  second element is the same quantity summed with the input
                  embedding (used for attention). The shape of both tensors is
                  `(batch, src_len, embed_dim)`.
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
        """
       # probs = self.softmax(self.weights)
        probs = self.logsoftmax(self.weights)

        #probs = self.weights/torch.sum(self.weights)
        #probs = self.dist.probs
        desired_shape = list(kwargs["target"].shape) + [1]
        res = probs.repeat(desired_shape).to(torch.device("cuda"))
        return res


from fairseq.models import register_model_architecture

# The first argument to ``register_model_architecture()`` should be the name
# of the model we registered above (i.e., 'simple_lstm'). The function we
# register here should take a single argument *args* and modify it in-place
# to match the desired architecture.

@register_model_architecture('unigram', 'unigram')
def tutorial_simple_lstm(args):
    pass
