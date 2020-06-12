import copy
from typing import List
import math

import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils import rnn
from torch.nn import ModuleList

from allennlp.nn import util

from misc import constant as CST


class MultiLayerMMA(nn.Module):
    def __init__(self, num_layers, in_features, num_heads, dim_feedforward=512, dropout=0.1, activation="relu",
                 share_single_mutual_attender=False, return_repeat=True):
        super(MultiLayerMMA, self).__init__()
        self.return_repeat = return_repeat
        encoder_layer = MutualAttendEncoderLayer(in_features, num_heads, dim_feedforward=dim_feedforward,
                                                 dropout=dropout, activation=activation,
                                                 share_single_mutual_attender=share_single_mutual_attender)
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.head_norm = nn.LayerNorm(in_features)
        self.tail_norm = nn.LayerNorm(in_features)

    def forward(self, head_mentions_embeddings, tail_mentions_embeddings, ht_pair_pos,
                head_mentions_indices_mask, tail_mentions_indices_mask, NRHET, need_weights=False):
        N, R, H, E, T = NRHET
        # (H, N*R, E)
        head_mentions_embeddings = head_mentions_embeddings.permute(1, 0, 2)
        # (T, N*R, E)
        tail_mentions_embeddings = tail_mentions_embeddings.permute(1, 0, 2)

        attended_head, attended_tail = head_mentions_embeddings, tail_mentions_embeddings
        for i in range(self.num_layers):
            _need = need_weights if i == (self.num_layers-1) else False
            attended_head, hAt_weights, attended_tail, tAh_weights = \
                self.layers[i](attended_head, attended_tail,
                               head_mentions_indices_mask,
                               tail_mentions_indices_mask,
                               NRHET, _need)
        if self.head_norm:
            attended_head = self.head_norm(attended_head)
        if self.tail_norm:
            attended_tail = self.tail_norm(attended_tail)

        del head_mentions_embeddings
        del tail_mentions_embeddings

        attended_head = attended_head.transpose(1, 0)
        attended_tail = attended_tail.transpose(1, 0)
        if not self.return_repeat:
            return attended_head.contiguous(), hAt_weights, attended_tail.contiguous(), tAh_weights
        # Cartesian Product of head and tail mentions
        # (N*R, H*T, E)
        attended_head = attended_head.repeat(1, 1, T).view(N*R, H*T, E)
        # (N*R, T*H, E)
        attended_tail = attended_tail.repeat(1, H, 1)
        assert attended_head.size() == attended_tail.size(), f"attended_head.size() == attended_tail.size(): " \
            f"{attended_head.size()} != {attended_tail.size()}"
        # ht_pair_pos (N, R, H*T)
        assert attended_head.size(1) == ht_pair_pos.size(2), f"attended_head.size(1) == ht_pair_pos.size(2):" \
            f" {attended_head.size(1)} != {ht_pair_pos.size(2)}"
        return attended_head, attended_tail


class MutualAttendEncoderLayer(nn.Module):
    def __init__(self, in_features, num_heads, dim_feedforward=512, dropout=0.1, activation="relu",
                 share_single_mutual_attender=False):
        super(MutualAttendEncoderLayer, self).__init__()
        self.share_single_mutual_attender = share_single_mutual_attender
        self.head_attend_tail = EncoderLayer(in_features=in_features, num_heads=num_heads,
                                             dim_feedforward=dim_feedforward, dropout=dropout, activation=activation)
        if not self.share_single_mutual_attender:
            self.tail_attend_head = EncoderLayer(in_features=in_features, num_heads=num_heads,
                                                 dim_feedforward=dim_feedforward, dropout=dropout, activation=activation)

    def forward(self, head_mentions_embeddings, tail_mentions_embeddings,
                head_mentions_indices_mask, tail_mentions_indices_mask, NRHET, need_weights=False):
        N, R, H, E, T = NRHET
        NR = N*R
        attended_head, hAt_weights = self.head_attend_tail(head_mentions_embeddings,
                                                           tail_mentions_embeddings,
                                                           key_padding_mask=(
                                                                       1 - tail_mentions_indices_mask.view(NR, T)).to(
                                                               dtype=torch.bool),
                                                           need_weights=need_weights)
        if not self.share_single_mutual_attender:
            attended_tail, tAh_weights = self.tail_attend_head(tail_mentions_embeddings,
                                                               head_mentions_embeddings,
                                                               key_padding_mask=(1 - head_mentions_indices_mask.view(NR,
                                                                                                                     H)).to(
                                                                   dtype=torch.bool),
                                                               need_weights=need_weights)
        else:
            attended_tail, tAh_weights = self.head_attend_tail(tail_mentions_embeddings,
                                                               head_mentions_embeddings,
                                                               key_padding_mask=(1 - head_mentions_indices_mask.view(NR,
                                                                                                                     H)).to(
                                                                   dtype=torch.bool),
                                                               need_weights=need_weights)
        return attended_head, hAt_weights, attended_tail, tAh_weights


class EncoderLayer(nn.Module):

    def __init__(self, in_features, num_heads, dim_feedforward, dropout, activation):
        super(EncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(in_features, num_heads, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(in_features, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, in_features)

        self.norm1 = nn.LayerNorm(in_features)
        self.norm2 = nn.LayerNorm(in_features)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def forward(self, tgt, src, key_padding_mask, need_weights=False):
        tgt2, attn_output_weights = self.self_attn(tgt, src, src,
                                                   key_padding_mask=key_padding_mask,
                                                   need_weights=need_weights)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        return tgt, attn_output_weights


class SpanExtractor(nn.Module):

    def __init__(self, in_features, entity_span_pooling):
        """

        :param entity_span_pooling: If `entity_span_pooling` is `mean-max`,
            the concat mean and max presentation will be down-project to original dimension.
        """
        super(SpanExtractor, self).__init__()
        self.entity_span_pooling = entity_span_pooling
        if self.entity_span_pooling == CST.pooling_style['mean-max']:
            self._span_pooling_down_projection = nn.Linear(2 * in_features, in_features)

    @staticmethod
    def _get_entity_span_tokens_embeddings(sentence_repr: torch.Tensor,
                                           entity_span_indices: torch.LongTensor) -> List[torch.Tensor]:
        """
            Most of the codes are extracted from `forward()` method of
            `https://github.com/allenai/allennlp/blob/master/allennlp/modules/span_extractors/self_attentive_span_extractor.py#L45`
        :param sentence_repr: (batch_size, seq_len, embedding_dim)
        :param entity_span_indices: (batch_size, num_spans, 2); last dim `0` for start, `1` for end (inclusive)
        :return: List (length 2) of Tensor:
            span_embeddings: (batch_size, num_spans, max_batch_span_width, embedding_dim)
            span_mask      : (batch_size, num_spans, max_batch_span_width)
        """
        # both of shape (batch_size, num_spans, 1)
        span_starts, span_ends = entity_span_indices.split(1, dim=-1)

        # shape (batch_size, num_spans, 1)
        # These span widths are off by 1, because the span ends are `inclusive`.
        span_widths = span_ends - span_starts

        # We need to know the maximum span width so we can
        # generate indices to extract the spans from the sequence tensor.
        # These indices will then get masked below, such that if the length
        # of a given span is smaller than the max, the rest of the values
        # are masked.
        max_batch_span_width = span_widths.max().item() + 1

        # shape (batch_size, sequence_length, 1)
        # global_attention_logits = self._global_attention(sentence_repr)

        # Shape: (1, 1, max_batch_span_width)
        max_span_range_indices = util.get_range_vector(max_batch_span_width,
                                                       util.get_device_of(sentence_repr)).view(1, 1, -1)
        # Shape: (batch_size, num_spans, max_batch_span_width)
        # This is a broadcasted comparison - for each span we are considering,
        # we are creating a range vector of size max_span_width, but masking values
        # which are greater than the actual length of the span.
        #
        # We're using <= here (and for the mask below) because the span ends are
        # inclusive, so we want to include indices which are equal to span_widths rather
        # than using it as a non-inclusive upper bound.
        span_mask = (max_span_range_indices <= span_widths).float()
        raw_span_indices = span_ends - max_span_range_indices
        # We also don't want to include span indices which are less than zero,
        # which happens because some spans near the beginning of the sequence
        # have an end index < max_batch_span_width, so we add this to the mask here.
        # Shape: (batch_size, num_spans, max_batch_span_width)
        span_mask = span_mask * (raw_span_indices >= 0).float()

        span_indices = torch.nn.functional.relu(raw_span_indices.float()).long()

        # Shape: (batch_size * num_spans * max_batch_span_width)
        flat_span_indices = util.flatten_and_batch_shift_indices(span_indices, sentence_repr.size(1))

        # Shape: (batch_size, num_spans, max_batch_span_width, embedding_dim)
        span_embeddings = util.batched_index_select(sentence_repr, span_indices, flat_span_indices)
        return [span_embeddings, span_mask]

    def forward(self, sentence_repr: torch.Tensor,
                entity_span_indices: torch.LongTensor) -> torch.Tensor:
        """
            This function returns entity span representations,
            given the original sequence representation and span indices ([start, end] end is inclusive)
        :param sentence_repr: (batch_size, seq_len, embed_dim)
        :param entity_span_indices: (batch_size, num_spans, 2)
        :return: Shape: (batch_size, num_spans, embedding_dim)
        """
        # span_embeddings Shape: (batch_size, num_spans, max_batch_span_width, embedding_dim)
        # span_mask,      Shape: (batch_size, num_spans, max_batch_span_width)
        span_embeddings, span_mask = self._get_entity_span_tokens_embeddings(sentence_repr, entity_span_indices)
        span_starts, span_ends = entity_span_indices.split(1, dim=-1)
        # shape (batch_size, num_spans, 1)
        span_widths = span_ends - span_starts + 1  # assign length 1 to spans whose length=1 or padding spans

        if self.entity_span_pooling == CST.pooling_style['max']:
            # Shape: (batch_size, num_spans, max_batch_span_width, embedding_dim)
            span_embeddings = span_embeddings - (1-span_mask.unsqueeze(-1).float())*1e32
            # Shape: (batch_size, num_spans, embedding_dim)
            span_embeddings = torch.max(span_embeddings, dim=2)[0]
            # span_embeddings = _get_span_max_pooling()
        elif self.entity_span_pooling == CST.pooling_style['mean']:  # mean pooling of tokens in entity span
            span_embeddings = span_embeddings * span_mask.unsqueeze(-1).float()
            # Shape: (batch_size, num_spans, embedding_dim)
            span_embeddings = torch.sum(span_embeddings, dim=2) / span_widths.float()  # / 0
            # span_embeddings = _get_span_mean_pooling()
        elif self.entity_span_pooling == CST.pooling_style['mean-max']:
            span_embeddings_max = torch.max(span_embeddings - (1-span_mask.unsqueeze(-1).float())*1e32, dim=2)[0]
            span_embeddings_mean = torch.sum(span_embeddings * span_mask.unsqueeze(-1).float(), dim=2) / span_widths.float()
            span_embeddings = self._span_pooling_down_projection(torch.cat([span_embeddings_max, span_embeddings_mean], dim=2))
        else:
            raise Exception('entity_span_pooling must be in `mean`, `max`, `mean-max`')
        # Shape: (batch_size, num_spans, embedding_dim)
        return span_embeddings


class TwoLayerLinear(nn.Module):
    def __init__(self, input_dim, hidden_dim=None, out_dim=1, bias=(False, True), activation=None):
        super(TwoLayerLinear, self).__init__()
        self.activation = None
        if activation is not None:
            self.activation = activation
        if hidden_dim is None:
            hidden_dim = input_dim
        self._projection = nn.Linear(input_dim, hidden_dim, bias=bias[0])
        # self._attention_logits = nn.Linear(hidden_dim, out_dim, bias=False)
        self._attention_logits = nn.Linear(hidden_dim, out_dim, bias=bias[1])

    def forward(self, input_tensor):
        if self.activation is None:
            states = torch.tanh(self._projection(input_tensor))
        else:
            states = self.activation(self._projection(input_tensor))
        return self._attention_logits(states)


class LockedDropout(nn.Module):
    def __init__(self, dropout):
        super(LockedDropout, self).__init__()
        self.dropout = dropout

    def forward(self, x):
        dropout = self.dropout
        if not self.training:
            return x
        m = x.data.new(x.size(0), 1, x.size(2)).bernoulli_(1 - dropout)
        mask = Variable(m.div_(1 - dropout), requires_grad=False)
        mask = mask.expand_as(x)
        return mask * x


class EncoderLSTM(nn.Module):
    def __init__(self, input_size, num_units, nlayers, concat, bidir, dropout, return_last):
        super(EncoderLSTM, self).__init__()
        self.rnns = []
        for i in range(nlayers):
            if i == 0:
                input_size_ = input_size
                output_size_ = num_units
            else:
                input_size_ = num_units if not bidir else num_units * 2
                output_size_ = num_units
            self.rnns.append(nn.LSTM(input_size_, output_size_, 1, bidirectional=bidir, batch_first=True))
        self.rnns = nn.ModuleList(self.rnns)

        self.init_hidden = nn.ParameterList(
            [nn.Parameter(torch.Tensor(2 if bidir else 1, 1, num_units).zero_()) for _ in range(nlayers)])
        self.init_c = nn.ParameterList(
            [nn.Parameter(torch.Tensor(2 if bidir else 1, 1, num_units).zero_()) for _ in range(nlayers)])

        self.dropout = LockedDropout(dropout)
        self.concat = concat
        self.nlayers = nlayers
        self.return_last = return_last

    # self.reset_parameters()

    def reset_parameters(self):
        for rnn in self.rnns:
            for name, p in rnn.named_parameters():
                if 'weight' in name:
                    p.data.normal_(std=0.1)
                else:
                    p.data.zero_()

    def get_init(self, bsz, i):
        return self.init_hidden[i].expand(-1, bsz, -1).contiguous(), self.init_c[i].expand(-1, bsz, -1).contiguous()

    def forward(self, input, input_lengths=None):
        bsz, slen = input.size(0), input.size(1)
        output = input
        outputs = []
        if input_lengths is not None:
            lens = input_lengths.data.cpu().numpy()

        for i in range(self.nlayers):
            hidden, c = self.get_init(bsz, i)

            output = self.dropout(output)
            if input_lengths is not None:
                output = rnn.pack_padded_sequence(output, lens, batch_first=True, enforce_sorted=False)
            self.rnns[i].flatten_parameters()
            output, hidden = self.rnns[i](output, (hidden, c))

            if input_lengths is not None:
                output, _ = rnn.pad_packed_sequence(output, batch_first=True)
                if output.size(1) < slen:  # used for parallel
                    padding = Variable(output.data.new(1, 1, 1).zero_())
                    output = torch.cat([output, padding.expand(output.size(0), slen - output.size(1), output.size(2))],
                                       dim=1)
            if self.return_last:
                outputs.append(hidden[0].permute(1, 0, 2).contiguous().view(bsz, -1))
            else:
                outputs.append(output)

        if self.concat:
            return torch.cat(outputs, dim=2)
        return outputs[-1]


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    else:
        raise RuntimeError("activation should be relu/gelu, not %s." % activation)


def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for _ in range(N)])


class OurLinear(nn.Module):
    r"""Applies a linear transformation to the incoming data: :math:`y = xA^T + b`
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0.0)
    """
    __constants__ = ['bias', 'in_features', 'out_features']

    def __init__(self, in_features, out_features, bias=True):
        super(OurLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0.0)

    def forward(self, input):
        return F.linear(input, self.weight, self.bias)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )
