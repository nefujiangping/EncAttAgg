import torch
import torch.nn as nn
import torch.nn.functional as F
import copy


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    else:
        raise RuntimeError("activation should be relu/gelu, not %s." % activation)


def _get_clones(module, num_layers):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(num_layers)])


class MultiLayerMMA(nn.Module):
    def __init__(self, num_layers, in_features, num_heads, dim_feedforward=512, dropout=0.1, activation="relu",
                 residual=False, logging=None, share_single_mutual_attender=False, return_repeat=True):
        super(MultiLayerMMA, self).__init__()
        self.return_repeat = return_repeat
        encoder_layer = MutualAttendEncoderLayer(in_features, num_heads, dim_feedforward=dim_feedforward,
                                                 dropout=dropout, activation=activation,
                                                 share_single_mutual_attender=share_single_mutual_attender)
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.head_norm = nn.LayerNorm(in_features)
        self.tail_norm = nn.LayerNorm(in_features)

        self.residual = residual
        if self.residual:
            self.head_gate = nn.Linear(in_features, in_features)
            self.tail_gate = nn.Linear(in_features, in_features)
            if logging is not None:
                logging("Use residual connection. In MutualMultiheadAttn.")

    def forward(self, head_mentions_embeddings, tail_mentions_embeddings, ht_pair_pos,
                head_mentions_indices_mask, tail_mentions_indices_mask, NRHET):
        N, R, H, E, T = NRHET
        # (H, N*R, E)
        head_mentions_embeddings = head_mentions_embeddings.permute(1, 0, 2)
        # (T, N*R, E)
        tail_mentions_embeddings = tail_mentions_embeddings.permute(1, 0, 2)

        attended_head, attended_tail = head_mentions_embeddings, tail_mentions_embeddings
        for i in range(self.num_layers):
            attended_head, attended_tail = self.layers[i](attended_head, attended_tail,
                                                          head_mentions_indices_mask,
                                                          tail_mentions_indices_mask, NRHET)
        if self.head_norm:
            attended_head = self.head_norm(attended_head)
        if self.tail_norm:
            attended_tail = self.tail_norm(attended_tail)

        # attended_head = attended_head.permute(1, 0, 2)
        if self.residual:
            # head_mentions_embeddings = head_mentions_embeddings.transpose(1, 0)
            gate = self.head_gate(head_mentions_embeddings).sigmoid()
            attended_head = gate*attended_head + (1-gate)*head_mentions_embeddings
            del gate
        del head_mentions_embeddings
        # (N*R, T, E)
        # attended_tail = attended_tail.permute(1, 0, 2)
        if self.residual:
            # tail_mentions_embeddings = tail_mentions_embeddings.transpose(1, 0)
            gate = self.tail_gate(tail_mentions_embeddings).sigmoid()
            attended_tail = gate*attended_tail + (1-gate)*tail_mentions_embeddings
            del gate
        del tail_mentions_embeddings

        attended_head = attended_head.transpose(1, 0)
        attended_tail = attended_tail.transpose(1, 0)
        if not self.return_repeat:
            return attended_head.contiguous(), attended_tail.contiguous()
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
                head_mentions_indices_mask, tail_mentions_indices_mask, NRHET):
        N, R, H, E, T = NRHET
        NR = N*R
        attended_head = self.head_attend_tail(head_mentions_embeddings,
                                              tail_mentions_embeddings,
                                              key_padding_mask=(1 - tail_mentions_indices_mask.view(NR, T)).to(dtype=torch.bool))
        if not self.share_single_mutual_attender:
            attended_tail = self.tail_attend_head(tail_mentions_embeddings,
                                                  head_mentions_embeddings,
                                                  key_padding_mask=(1 - head_mentions_indices_mask.view(NR, H)).to(dtype=torch.bool))
        else:
            attended_tail = self.head_attend_tail(tail_mentions_embeddings,
                                                  head_mentions_embeddings,
                                                  key_padding_mask=(1 - head_mentions_indices_mask.view(NR, H)).to(dtype=torch.bool))
        return attended_head, attended_tail


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

    def forward(self, tgt, src, key_padding_mask):
        tgt2 = self.self_attn(tgt, src, src,
                              key_padding_mask=key_padding_mask,
                              need_weights=False)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        return tgt


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
