from typing import Dict, List, Union
import torch.nn as nn
from misc.constant import MutualAttender, IntegrationAttender, PoolingStyle
from models.model_utils import *
from models.modules import *
from models.embedding import DocumentEncoder
from models.lock_dropout import LockedDropout


class Embedding(nn.Module):
    def __init__(self, pretrained_model_name_or_path: str = 'bert-base-cased',
                 transformer_type: str = 'bert',
                 num_hidden_layers: int = 12,
                 freeze_embed: bool = False,
                 embedd_dim: int = None,
                 word_dropout: float = 0.1,
                 max_length: int = None,
                 coref_size: int = None,
                 num_entity_type: int = None,
                 entity_type_size: int = None) -> None:
        """
            embedding layer:
                word-piece embedding (then transform to token embedding)
                entity_cluster_embedding (if max_length is not None)
                entity_type_embedding (if num_entity_type is not None)
        :param pretrained_model_name_or_path:
        :param word_dropout:
        :param max_length:
        :param coref_size:
        :param num_entity_type:
        :param entity_type_size:
        """
        super(Embedding, self).__init__()
        self.doc_embed = DocumentEncoder(
            pretrained_model_name_or_path,
            transformer_type=transformer_type,
            num_hidden_layers=num_hidden_layers,
            freeze=freeze_embed)
        self.doc_embed_down_proj = None
        if self.doc_embed.output_dim != embedd_dim:
            self.doc_embed_down_proj = nn.Linear(self.doc_embed.output_dim, embedd_dim, bias=False)
        self.output_dim = embedd_dim
        self.word_dropout = nn.Dropout(word_dropout) if word_dropout > 0.0 else None
        self.entity_cluster_embed = nn.Embedding(max_length, coref_size, padding_idx=0) if max_length else None
        self.entity_type_embed = \
            nn.Embedding(num_entity_type, entity_type_size, padding_idx=0) if num_entity_type else None
        self.output_dim += coref_size if coref_size else 0
        self.output_dim += entity_type_size if entity_type_size else 0
        self.output_dim = self.output_dim

    def forward(self,
                input_ids: torch.LongTensor,
                attention_mask: torch.Tensor,
                t2p_map: torch.LongTensor,
                t2p_map_mask: torch.LongTensor,
                word_mask: torch.LongTensor,
                entity_cluster_ids: torch.LongTensor = None,
                entity_type_ids: torch.LongTensor = None,
                indices: List = None) -> torch.Tensor:
        seq_repr = self.doc_embed(input_ids, attention_mask)
        if self.doc_embed_down_proj is not None:
            seq_repr = self.doc_embed_down_proj(seq_repr)
        # [B, wordpiece-L, H]
        if self.word_dropout is not None:
            seq_repr = self.word_dropout(seq_repr)
        # ===== transform word-piece-level repr to word-level repr
        # B, word_seq_len, pieces_per_word = t2p_map.size()
        # [B, word-L, pieces_per_word, H]
        seq_repr = batched_index_select(seq_repr, t2p_map)
        # [B, word-L, pieces_per_word, H]
        seq_repr = seq_repr * t2p_map_mask.unsqueeze(-1).float()
        # ===== zero-out padding words
        # [B, word-L, H], mean-pooling over word-pieces
        seq_repr = torch.sum(seq_repr, dim=2) / t2p_map_mask.sum(dim=-1, keepdim=True).float()
        seq_repr = seq_repr * word_mask.unsqueeze(-1).float()
        # ===== concat entity_cluster_embed feature
        if self.entity_cluster_embed is not None:
            seq_repr = torch.cat([seq_repr, self.entity_cluster_embed(entity_cluster_ids)], dim=-1)
        # ===== concat entity_type_embed feature
        if self.entity_type_embed is not None:
            seq_repr = torch.cat([seq_repr, self.entity_type_embed(entity_type_ids)], dim=-1)
        assert seq_repr.size(-1) == self.output_dim
        return seq_repr


class BiLSTMEncoder(nn.Module):
    def __init__(self, input_size, num_units, num_layers, concat=False, bidir=True, dropout=0.2, return_last=False):
        super(BiLSTMEncoder, self).__init__()
        self.rnns = []
        for i in range(num_layers):
            if i == 0:
                input_size_ = input_size
                output_size_ = num_units
            else:
                input_size_ = num_units if not bidir else num_units * 2
                output_size_ = num_units
            self.rnns.append(nn.LSTM(input_size_, output_size_, 1, bidirectional=bidir, batch_first=True))
        self.rnns = nn.ModuleList(self.rnns)

        self.init_hidden = nn.ParameterList(
            [nn.Parameter(torch.zeros((2 if bidir else 1, 1, num_units))) for _ in range(num_layers)])
        self.init_c = nn.ParameterList(
            [nn.Parameter(torch.zeros((2 if bidir else 1, 1, num_units))) for _ in range(num_layers)])

        self.dropout = LockedDropout(dropout)
        self.concat = concat
        self.nlayers = num_layers
        self.return_last = return_last
        self.output_dim = num_units * (num_layers if concat else 1)
        self.output_dim = self.output_dim * (2 if bidir else 1)

    def reset_parameters(self):
        for rnn in self.rnns:
            for name, p in rnn.named_parameters():
                if 'weight' in name:
                    p.data.normal_(std=0.1)
                else:
                    p.data.zero_()

    def get_init(self, bsz, i):
        return self.init_hidden[i].expand(-1, bsz, -1).contiguous(), self.init_c[i].expand(-1, bsz, -1).contiguous()

    def forward(self, seq_repr: torch.Tensor, input_lengths=None):
        bsz, seq_len = seq_repr.size(0), seq_repr.size(1)
        output = seq_repr
        outputs = []
        if input_lengths is not None:
            input_lengths = input_lengths.data.cpu().numpy()

        for i in range(self.nlayers):
            hidden, c = self.get_init(bsz, i)

            output = self.dropout(output)
            if input_lengths is not None:
                output = torch.nn.utils.rnn.pack_padded_sequence(
                    output, input_lengths, batch_first=True, enforce_sorted=False)
            self.rnns[i].flatten_parameters()
            output, hidden = self.rnns[i](output, (hidden, c))

            if input_lengths is not None:
                output, _ = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
                if output.size(1) < seq_len:  # used for parallel
                    padding = torch.tensor(output.data.new(1, 1, 1).zero_()).to(output)
                    output = torch.cat(
                        [output, padding.expand(output.size(0), seq_len - output.size(1), output.size(2))], dim=1)
        
            if self.return_last:
                outputs.append(hidden[0].permute(1, 0, 2).contiguous().view(bsz, -1))
            else:
                outputs.append(output)

        if self.concat:
            return torch.cat(outputs, dim=2)
        output = outputs[-1]
        assert output.size(-1) == self.output_dim
        return output


class MentionExtractor(nn.Module):

    def __init__(self, in_features, mention_span_pooling):
        """

        :param mention_span_pooling: If `entity_span_pooling` is `mean-max`,
            the concat mean and max presentation will be down-project to original dimension.
        """
        super(MentionExtractor, self).__init__()
        print("\n\nSpanExtractor\n\n")
        self.mention_span_pooling = PoolingStyle.__getattr__(mention_span_pooling)
        if self.mention_span_pooling == PoolingStyle.mean_max:
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
        max_span_range_indices = get_range_vector(max_batch_span_width, get_device_of(sentence_repr)).view(1, 1, -1)
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
        flat_span_indices = flatten_and_batch_shift_indices(span_indices, sentence_repr.size(1))

        # Shape: (batch_size, num_spans, max_batch_span_width, embedding_dim)
        span_embeddings = batched_index_select(sentence_repr, span_indices, flat_span_indices)
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

        if self.mention_span_pooling == PoolingStyle.max:
            # Shape: (batch_size, num_spans, max_batch_span_width, embedding_dim)
            span_embeddings = span_embeddings - (1-span_mask.unsqueeze(-1).float())*1e32
            # Shape: (batch_size, num_spans, embedding_dim)
            span_embeddings = torch.max(span_embeddings, dim=2)[0]
            # span_embeddings = _get_span_max_pooling()
        elif self.mention_span_pooling == PoolingStyle.mean:  # mean pooling of tokens in entity span
            span_embeddings = span_embeddings * span_mask.unsqueeze(-1).float()
            # Shape: (batch_size, num_spans, embedding_dim)
            span_embeddings = torch.sum(span_embeddings, dim=2) / span_widths.float()  # / 0
            # span_embeddings = _get_span_mean_pooling()
        elif self.mention_span_pooling == PoolingStyle.mean_max:
            span_embeddings_max = torch.max(span_embeddings - (1-span_mask.unsqueeze(-1).float())*1e32, dim=2)[0]
            span_embeddings_mean = torch.sum(
                span_embeddings * span_mask.unsqueeze(-1).float(), dim=2) / span_widths.float()
            span_embeddings = self._span_pooling_down_projection(
                torch.cat([span_embeddings_max, span_embeddings_mean], dim=2))
        else:
            raise Exception('entity_span_pooling must be in `mean`, `max`, `mean-max`')
        # Shape: (batch_size, num_spans, embedding_dim)
        return span_embeddings


class AttenderAggregator(nn.Module):
    def __init__(self, in_features: int,
                 dis_size: int,
                 relation_num: int,
                 use_distance: bool = True,
                 mutual_attender: dict = None,
                 integration_attender: dict = None,
                 overlap=True,
                 logging=None):
        super(AttenderAggregator, self).__init__()
        self.use_distance = use_distance
        self.num_relation = relation_num
        self.in_features = in_features
        self.dis_size = dis_size
        self.overlap = overlap

        rel_features_dim = in_features*3
        if not self.overlap:
            rel_features_dim = in_features*2

        if self.use_distance:
            self.dis_embed = self.dis_embed = nn.Embedding(20, dis_size, padding_idx=10)
            rel_features_dim += 2*self.dis_size

        # params of mutual attender
        self.mutual_attender = MutualAttender.__getattr__(mutual_attender['attender'])  # 'ML_MMA', 'NONE'
        self.mutual_attend_num_layers = mutual_attender['num_layers']
        self.mutual_attend_nhead = mutual_attender['nhead']
        self.mutual_attend_dim_ff = 4*in_features
        self.mutual_drop = mutual_attender['drop']
        self.share_single_mutual_attender = mutual_attender['shared']

        # params of integration attender
        self.integration_attender = IntegrationAttender.__getattr__(integration_attender['attender'])  # 'ML_MA', 'NONE'
        self.integration_num_layers = integration_attender['num_layers']
        self.integration_nhead = integration_attender['nhead']
        self.integration_dim_ff = 4 * rel_features_dim
        self.integration_drop = integration_attender['drop']

        # mutual_attender
        if self.mutual_attender == MutualAttender.ML_MMA:
            self.attention = MultiLayerMMA(num_layers=self.mutual_attend_num_layers, in_features=in_features,
                                           num_heads=self.mutual_attend_nhead,
                                           dim_feedforward=self.mutual_attend_dim_ff,
                                           dropout=self.mutual_drop, activation="relu",
                                           share_single_mutual_attender=self.share_single_mutual_attender,
                                           return_repeat=False)
        elif self.mutual_attender == MutualAttender.NONE:
            pass
        else:
            raise Exception(f'mutual_attender ERROR in {self.__class__.__name__} init')

        # integration_attender
        if self.integration_attender == IntegrationAttender.ML_MA:
            encoder_layer = nn.TransformerEncoderLayer(d_model=rel_features_dim, nhead=self.integration_nhead,
                                                       dim_feedforward=self.integration_dim_ff,
                                                       dropout=self.integration_drop, activation='relu')
            encoder_norm = nn.LayerNorm(rel_features_dim)
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.integration_num_layers,
                                                             norm=encoder_norm)
        elif self.integration_attender == IntegrationAttender.NONE:
            pass
        else:
            raise Exception('integration_attender ERROR in {self.__class__.__name__} init')

        # project mention representation to scores
        hidden_dim = 512
        self.final_proj = TwoLayerLinear(
            rel_features_dim, hidden_dim=hidden_dim, activation=torch.relu, out_dim=self.num_relation)

        # print config
        self.print_config(logging)

    def print_config(self, logging):
        log_str = f"\nClass {self.__class__.__name__} constructor.\n" \
                  f"use_distance: {self.use_distance}\n" \
                  f"overlap: {self.overlap}\n" \
                  f"dis_size: {self.dis_size}\n"
        log_str += "mutual_attend        : {:<15}, num_layer: {:<2}, " \
                   "nhead: {:<3}, dim_ff: {:<5} share: {}\n".format(self.mutual_attender, self.mutual_attend_num_layers,
                                                                    self.mutual_attend_nhead, self.mutual_attend_dim_ff,
                                                                    str(self.share_single_mutual_attender))
        log_str += "integration_attender : {:<15}, num_layer: {:<2}, " \
                   "nhead: {:<3}, dim_ff: {:<5}\n".format(self.integration_attender, self.integration_num_layers,
                                                          self.integration_nhead, self.integration_dim_ff)

        if logging is not None:
            logging(log_str)
        else:
            print(log_str)

    def forward(self,
                mention_span_embeddings: torch.Tensor,
                head_mentions_indices: torch.LongTensor,
                head_mentions_indices_mask: torch.LongTensor,
                tail_mentions_indices: torch.LongTensor,
                tail_mentions_indices_mask: torch.LongTensor,
                ht_comb_indices: torch.LongTensor,
                ht_comb_mask: torch.LongTensor,
                dis_h_2_t: torch.LongTensor,
                dis_t_2_h: torch.LongTensor,
                is_train: bool = True):
        """
        :param mention_span_embeddings:    (N, num_spans, H)
        :param head_mentions_indices:      (N, R, H)
        :param head_mentions_indices_mask: (N, R, H)
        :param tail_mentions_indices:      (N, R, T)
        :param tail_mentions_indices_mask: (N, R, T)
        :param ht_comb_indices: (N, R, HT, 2)
        :param ht_comb_mask:    (N, R, HT)
        :param dis_h_2_t:       (N, R, HT)
        :param dis_t_2_h:       (N, R, HT)
        :param is_train: True or False
        :return:
        """
        # obtain head_mentions_embeddings, tail_mentions_embeddings
        # (N, R, T, E)
        head_mentions_embeddings = batched_index_select(mention_span_embeddings, head_mentions_indices)
        tail_mentions_embeddings = batched_index_select(mention_span_embeddings, tail_mentions_indices)
        head_mentions_embeddings = head_mentions_embeddings*head_mentions_indices_mask.unsqueeze(-1).float()
        tail_mentions_embeddings = tail_mentions_embeddings*tail_mentions_indices_mask.unsqueeze(-1).float()
        # del head_mentions_indices_mask
        N, R, H, E = head_mentions_embeddings.size()
        T = tail_mentions_embeddings.size(2)
        NR = N*R
        NRHET = (N, R, H, E, T)
        HT = ht_comb_mask.size(2)
        head_mentions_embeddings = head_mentions_embeddings.view(N*R, H, E)
        tail_mentions_embeddings = tail_mentions_embeddings.view(N*R, T, E)

        # BiDAF head_attend_tail/tail_attend_head
        # weights_to_return = {
        #     "hAt_weights": None,  # mutual attend: head attend tail (N, R, H, T)
        #     "tAh_weights": None,  # mutual attend: tail attend head (N, R, T, H)
        #     "weights_on_mp": None  # Mention pairs self-attend:     (N, R, HT, HT)
        # }
        if self.mutual_attender == MutualAttender.ML_MMA:
            # (NR, H, E), (NR, T, E)
            attended_head, attended_tail = self.attention(
                head_mentions_embeddings,
                tail_mentions_embeddings, dis_h_2_t,
                head_mentions_indices_mask,
                tail_mentions_indices_mask, NRHET)
            # weights_to_return["hAt_weights"] = hAt_weights.view(N, R, H, T)
            # weights_to_return["tAh_weights"] = tAh_weights.view(N, R, T, H)
        elif self.mutual_attender == MutualAttender.NONE:
            attended_head = head_mentions_embeddings
            attended_tail = tail_mentions_embeddings
        else:
            raise Exception(f'mutual_attender ERROR in {self.__class__.__name__} forward.')
        del head_mentions_embeddings, tail_mentions_embeddings

        head_indices = ht_comb_indices[:, :, :, 0].view(NR, HT)
        tail_indices = ht_comb_indices[:, :, :, 1].view(NR, HT)
        del ht_comb_indices

        # repr(NR, H, E) indices(NR, HT) --> (NR, HT, E)
        attended_head = batched_index_select(attended_head, head_indices)
        # repr(NR, T, E) indices(NR, HT) --> (NR, HT, E)
        attended_tail = batched_index_select(attended_tail, tail_indices)

        if self.use_distance:
            h2t_dis_embedd = self.dis_embed(dis_h_2_t.view(NR, HT))
            t2h_dis_embedd = self.dis_embed(dis_t_2_h.view(NR, HT))

        if self.use_distance:
            # (N*R, HT, E)
            if self.overlap:
                rel_features = torch.cat([attended_head, attended_tail,
                                          attended_head * attended_tail,
                                          h2t_dis_embedd,
                                          t2h_dis_embedd], dim=-1)
            else:
                rel_features = torch.cat([attended_head, attended_tail,
                                          h2t_dis_embedd,
                                          t2h_dis_embedd], dim=-1)
        else:
            # (N*R, HT, E)
            if self.overlap:
                rel_features = torch.cat([attended_head, attended_tail,
                                          attended_head * attended_tail], dim=-1)
            else:
                rel_features = torch.cat([attended_head, attended_tail], dim=-1)
        # (N*R, HT, E)
        rel_features = rel_features*ht_comb_mask.view(NR, HT).unsqueeze(-1).float()

        # Fusion mention pairs
        if self.integration_attender == IntegrationAttender.NONE:
            pass
        elif self.integration_attender == IntegrationAttender.ML_MA:
            rel_features = rel_features.permute(1, 0, 2)
            # weights (N, L(tgt), S(src))  (N, R, HT, HT)
            rel_features = self.transformer_encoder(
                rel_features, src_key_padding_mask=(1 - ht_comb_mask.view(NR, HT)).to(dtype=torch.bool))
            # weights_to_return["weights_on_mp"] = attn_weights.view(N, R, HT, HT)
            rel_features = rel_features.permute(1, 0, 2)
        else:
            raise Exception(f'integration_attender ERROR in {self.__class__.__name__} forward.')

        # LogSumExp()
        # (N, R, HT, C)
        logits = self.final_proj(rel_features).view(N, R, HT, -1)
        del rel_features
        logits = logits - (1.0-ht_comb_mask.unsqueeze(-1).float())*1e10
        logits = torch.logsumexp(logits, dim=2)

        return logits


class RelationExtraction(nn.Module):
    def __init__(self, config):
        super(RelationExtraction, self).__init__()
        self.config = config
        # ====================== Embedding Layer =======================
        self.word_embedding = Embedding(
            pretrained_model_name_or_path=config.model_name,
            transformer_type=config.transformer_type,
            freeze_embed=config.freeze_embed,
            embedd_dim=config.embedd_dim,  # down project to size `embedd_dim`
            word_dropout=config.word_dropout,  # then dropout
            max_length=config.max_length if config.use_entity_cluster else None,
            coref_size=config.coref_size if config.use_entity_cluster else None,
            num_entity_type=config.entity_type_num if config.use_entity_type else None,
            entity_type_size=config.entity_type_size if config.use_entity_type else None
        )
        # ====================== Encoder Layer =======================
        hidden_size = config.hidden_size
        self.bilstm_encoder = BiLSTMEncoder(
            input_size=self.word_embedding.output_dim,
            num_units=hidden_size,
            num_layers=config.num_bilstm_layers,
            dropout=1.0-config.lstm_keep_prob,
        )
        # self.repr_drop = nn.Dropout(1.0-config.lstm_keep_prob)
        self.repr_down_proj = nn.Linear(self.bilstm_encoder.output_dim, hidden_size)
        # ================= Mention Extractor Layer ===================
        self.mention_extractor = MentionExtractor(
            in_features=hidden_size, mention_span_pooling=config.entity_span_pooling)
        # ================ Attender & Aggregator ====================
        self.attender_aggregator = AttenderAggregator(
            in_features=hidden_size,
            dis_size=config.dis_size,
            relation_num=config.relation_num,
            use_distance=config.use_distance,
            mutual_attender=config.mutual_attender,
            integration_attender=config.integration_attender,
            overlap=config.use_overlap,
            logging=config.logging
        )
        self.output_dim = config.relation_num
        # self.loss_func = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, inputs: Dict[str, Union[torch.Tensor, Dict]], is_train=True):
        """
            Main forward function
        :param inputs: dict
            input_ids:           (N, word-pieces-L)
            attention_mask:      (N, word-pieces-L)
            entity_cluster_ids:  (N, token-L)
            entity_type_ids:     (N, token-L)
            input_mask:          (N, token-L)

            entity_span_indices: (N, num_mentions, 2)

            for_relation_repr: dict
                head_mentions_indices:      (N, R, H)
                head_mentions_indices_mask: (N, R, H)
                tail_mentions_indices:      (N, R, T)
                tail_mentions_indices_mask: (N, R, T)
                ht_comb_indices:            (N, R, HT, 2)
                ht_comb_mask:               (N, R, HT)
            dis_h_2_t: (N, R, HT)
            dis_t_2_h: (N, R, HT)
        :param is_train: True or False
        :return: logits: (N, R, relation_num)
        """

        # B: batch_size, number of documents
        # L: sequence length (after padding over the batch)
        # E: embed_dim(+coref_dim+entity_dim)
        # [B, L, E]
        seq_repr = self.word_embedding(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            t2p_map=inputs['t2p_map'],
            t2p_map_mask=inputs['t2p_map_mask'],
            word_mask=inputs['word_mask'],
            entity_cluster_ids=inputs['entity_cluster_ids'] if self.config.use_entity_cluster else None,
            entity_type_ids=inputs['entity_type_ids'] if self.config.use_entity_type else None,
            indices=inputs['indexes']
        )

        # [B, L, BiLSTM-output_dim]
        seq_repr = self.bilstm_encoder(seq_repr, input_lengths=inputs['word_mask'].sum(dim=1))
        # seq_repr = self.repr_drop(seq_repr)
        # [B, L, H]
        seq_repr = torch.relu(self.repr_down_proj(seq_repr))

        # num_mentions: number of mentions per document (after padding over the batch)
        # [B, num_mentions, H]
        mention_repr = self.mention_extractor(seq_repr, inputs['entity_span_indices'])

        # R: number of entity pairs per document (after padding over the batch)
        # [B, R, relation_num]
        logits = self.attender_aggregator(
            mention_span_embeddings=mention_repr,
            dis_h_2_t=inputs['dis_h_2_t'],
            dis_t_2_h=inputs['dis_t_2_h'],
            is_train=is_train,
            **inputs['for_relation_repr']
        )

        return logits

    # def loss(self,
    #          logits: torch.Tensor,
    #          relation_multi_label: torch.Tensor,
    #          relation_mask: torch.Tensor) -> torch.Tensor:
    #     # [B, R, relation_num]
    #     loss = self.loss_func(logits, relation_multi_label) * relation_mask.unsqueeze(2)
    #     loss = torch.sum(loss) / (self.config.relation_num * torch.sum(relation_mask))
    #     return loss
