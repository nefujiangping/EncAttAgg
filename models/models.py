from typing import List, Tuple

import torch
from torch import nn
from allennlp.nn import util

from trainer.GCNN.CDR_GCNN_Trainer import CDRGCNNTrainer_Softmax
from models.transformer import TransformerEncoder, TransformerEncoderLayer
from models.modules import MultiLayerMMA, SpanExtractor, TwoLayerLinear, EncoderLSTM, OurLinear
from misc import constant as CST


def print_nan(tensor, code):
    if torch.isnan(tensor).sum().item() > 0:
        print(code)
        return True
    return False


class RelationExtraction(nn.Module):
    def __init__(self, config):
        super(RelationExtraction, self).__init__()
        self.config = config

        # word embedding
        if not config.use_bert_embedding:
            assert config.embedd_dim == config.data_word_vec.shape[1], f"config.embedd_dim != config.data_word_vec.shape[1]: " \
                                                                       f"{config.embedd_dim} != {config.data_word_vec.shape[1]}"
            print(f'Use GloVe embeddings. embedd_size: {config.data_word_vec.shape[1]}')
            word_vec_size = config.data_word_vec.shape[0]
            self.word_emb = nn.Embedding(word_vec_size, config.data_word_vec.shape[1])
            self.word_emb.weight.data.copy_(torch.from_numpy(config.data_word_vec))
            self.word_emb.weight.requires_grad = not config.freeze_embedd
        else:
            print('Use Bert embeddings.')
            if config.embedd_dim != config.bert_embedd_dim:  # do down-projection
                print(f"Down-project input embedding to {config.embedd_dim}.")
                self.embed_project = nn.Linear(in_features=config.bert_embedd_dim, out_features=config.embedd_dim, bias=False)
            else:
                print("Don't project input embedding.")

        self.drop_word = config.drop_word['use_drop']
        if self.drop_word:
            print(f"Use word drop: dropout {config.drop_word['drop']}")
            self.word_drop = nn.Dropout(config.drop_word['drop'])

        self.use_entity_type = True
        self.use_coreference = True

        hidden_size = config.hidden_size
        input_size = config.embedd_dim
        # entity type embedding & co-reference cluster embedding
        if self.use_entity_type:
            input_size += config.entity_type_size
            self.ner_emb = nn.Embedding(config.entity_type_num, config.entity_type_size, padding_idx=0)
            # self.ner_emb = nn.Embedding(7, config.entity_type_size, padding_idx=0)
        if self.use_coreference:
            input_size += config.coref_size
            self.entity_embed = nn.Embedding(config.max_length, config.coref_size, padding_idx=0)

        # Encoder: stack of BiLSTM layers
        bidirectional = True
        concat_nlayer_outputs = config.cat_nlayer
        self.encoder = EncoderLSTM(input_size, hidden_size,
                                   nlayers=config.nlayer,
                                   concat=concat_nlayer_outputs,
                                   bidir=bidirectional,
                                   dropout=1-config.lstm_keep_prob,
                                   return_last=False)
        in_features = hidden_size * (config.nlayer if concat_nlayer_outputs else 1)
        in_features = in_features * (2 if bidirectional else 1)

        # down project encoder outputs to `hidden_size`
        self.linear_re = nn.Linear(in_features, hidden_size)
        config.logging(f" In {self.__class__.__name__} Constructor\n"
                       f" encoder output size: {in_features}, project to: {hidden_size} (hidden_size)")

        # Attender layer
        self.which_model = config.which_model
        if self.which_model == CST.which_model['BiLSTM-M']:
            self.relation_embedding = \
                BiLSTM_M(in_features=hidden_size,
                         entity_span_pooling=config.entity_span_pooling,
                         coref_pooling=config.coref_pooling,
                         use_distance=config.use_distance,
                         dis_size=config.dis_size,
                         use_bilinear=config.use_bilinear,
                         relation_num=config.relation_num,
                         hidden_dim=512)
        elif self.which_model == CST.which_model['BRAN-M']:
            self.bran = \
                BRAN_M(in_features=hidden_size,
                       dis_size=config.dis_size,
                       relation_num=config.relation_num,
                       entity_span_pooling=config.entity_span_pooling,
                       bi_affine_dropout=config.bi_affine_dropout,
                       use_distance=config.use_distance,
                       logging=config.logging,
                       dataset=config.dataset)
        elif self.which_model in [CST.which_model['EncAgg'], CST.which_model['EncAttAgg']]:
            if self.which_model == CST.which_model['EncAgg']:
                assert config.mutual_attender['attender'] == CST.mutual_attender['NONE'] \
                       and config.integration_attender['attender'] == CST.integration_attender['NONE'], \
                       f"Now it is running `EncAgg` model, make sure that both attenders are set to `NONE`"
            self.enc_att_agg = \
                EncAttAgg(in_features=hidden_size,
                          dis_size=config.dis_size,
                          relation_num=config.relation_num,
                          entity_span_pooling=config.entity_span_pooling,
                          use_distance=config.use_distance,
                          integration_attender=config.integration_attender,
                          logging=config.logging,
                          mutual_attender=config.mutual_attender,
                          dataset=config.dataset,
                          overlap=config.use_overlap)
        else:
            raise Exception(f"which_model ERROR in {self.__class__.__name__} init")

    def forward(self, context_idxs, pos, context_ner, context_char_idxs, input_mask,
                for_relation_repr, relation_mask, dis_h_2_t, dis_t_2_h, bert_feature=None,
                relation_label=None, is_train=True):
        if bert_feature is not None:
            if self.config.embedd_dim != self.config.bert_embedd_dim:
                sent = self.embed_project(bert_feature)
            else:
                sent = bert_feature
        else:
            sent = self.word_emb(context_idxs)
        if self.drop_word:
            sent = self.word_drop(sent)

        if self.use_coreference:
            sent = torch.cat([sent, self.entity_embed(pos)], dim=-1)
        if self.use_entity_type:
            sent = torch.cat([sent, self.ner_emb(context_ner)], dim=-1)

        context_output = self.encoder(sent, input_mask.sum(dim=1))
        context_output = torch.relu(self.linear_re(context_output))

        if self.which_model == CST.which_model['BiLSTM-M']:
            predict_logits, weights_to_return = \
                self.relation_embedding(sentence_repr=context_output,
                                        relation_mask=relation_mask,
                                        dis_h_2_t=dis_h_2_t,
                                        dis_t_2_h=dis_t_2_h,
                                        **for_relation_repr)
        elif self.which_model == CST.which_model['BRAN-M']:
            predict_logits, weights_to_return = \
                self.bran(sentence_repr=context_output,
                          relation_mask=relation_mask,
                          dis_h_2_t=dis_h_2_t,
                          dis_t_2_h=dis_t_2_h,
                          relation_label=relation_label,
                          is_train=is_train,
                          **for_relation_repr)
        elif self.which_model in [CST.which_model['EncAgg'], CST.which_model['EncAttAgg']]:
            predict_logits, weights_to_return = \
                self.enc_att_agg(sentence_repr=context_output,
                                 relation_mask=relation_mask,
                                 dis_h_2_t=dis_h_2_t,
                                 dis_t_2_h=dis_t_2_h,
                                 relation_label=relation_label,
                                 is_train=is_train,
                                 **for_relation_repr)
        else:
            raise Exception(f"which_model ERROR in {self.__class__.__name__} forward")

        if is_train:
            return predict_logits
        else:
            return predict_logits, weights_to_return


class EncAttAgg(nn.Module):
    def __init__(self, in_features: int, dis_size: int, relation_num: int,
                 entity_span_pooling: str = f"{CST.pooling_style['mean']}",
                 use_distance: bool = True,
                 integration_attender: dict = None,
                 logging=None,
                 mutual_attender: dict = None,
                 dataset: str = CST.datasets['DocRED'],
                 overlap=True):
        super(EncAttAgg, self).__init__()
        self.use_distance = use_distance
        self.num_relation = relation_num
        self.in_features = in_features
        self.dis_size = dis_size
        self.dataset = dataset
        self.overlap = overlap
        self.entity_span_pooling = entity_span_pooling
        self._entity_spans_embeddings = SpanExtractor(in_features, entity_span_pooling)

        rel_features_dim = in_features*3
        if not self.overlap:
            rel_features_dim = in_features*2

        if self.use_distance:
            self.dis_embed = self.dis_embed = nn.Embedding(20, dis_size, padding_idx=10)
            rel_features_dim += 2*self.dis_size

        # params of mutual attender
        self.mutual_attender = mutual_attender['attender']  # 'ML-MMA', 'NONE'
        self.mutual_attend_num_layers = mutual_attender['num_layers']
        self.mutual_attend_nhead = mutual_attender['nhead']
        self.mutual_attend_dim_ff = 4*in_features
        self.mutual_drop = mutual_attender['drop']
        self.share_single_mutual_attender = mutual_attender['shared']

        # params of integration attender
        self.integration_attender = integration_attender['attender']  # 'ML-MA', 'NONE'
        self.integration_num_layers = integration_attender['num_layers']
        self.integration_nhead = integration_attender['nhead']
        self.integration_dim_ff = 4 * rel_features_dim
        self.integration_drop = integration_attender['drop']

        # mutual_attender
        if self.mutual_attender == CST.mutual_attender['ML-MMA']:
            self.attention = MultiLayerMMA(num_layers=self.mutual_attend_num_layers, in_features=in_features,
                                           num_heads=self.mutual_attend_nhead,
                                           dim_feedforward=self.mutual_attend_dim_ff,
                                           dropout=self.mutual_drop, activation="relu",
                                           share_single_mutual_attender=self.share_single_mutual_attender,
                                           return_repeat=False)
        elif self.mutual_attender == CST.mutual_attender['NONE']:
            pass
        else:
            raise Exception(f'mutual_attender ERROR in {self.__class__.__name__} init')

        # integration_attender
        if self.integration_attender == CST.integration_attender['ML-MA']:
            encoder_layer = TransformerEncoderLayer(d_model=rel_features_dim, nhead=self.integration_nhead,
                                                    dim_feedforward=self.integration_dim_ff,
                                                    dropout=self.integration_drop, activation='relu')
            encoder_norm = nn.LayerNorm(rel_features_dim)
            self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=self.integration_num_layers,
                                                          norm=encoder_norm)
        elif self.integration_attender == CST.integration_attender['NONE']:
            pass
        else:
            raise Exception('integration_attender ERROR in {self.__class__.__name__} init')

        # project mention representation to scores
        hidden_dim = 512
        self.final_proj = TwoLayerLinear(rel_features_dim, hidden_dim=hidden_dim, activation=torch.relu, out_dim=self.num_relation)

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
                sentence_repr: torch.Tensor,
                entity_span_indices: torch.LongTensor,
                head_mentions_indices: torch.LongTensor,
                head_mentions_indices_mask: torch.LongTensor,
                tail_mentions_indices: torch.LongTensor,
                tail_mentions_indices_mask: torch.LongTensor,
                ht_comb_indices: torch.LongTensor,
                ht_comb_mask: torch.LongTensor,
                dis_h_2_t: torch.LongTensor,
                dis_t_2_h: torch.LongTensor,
                relation_mask: torch.LongTensor,
                relation_label: torch.LongTensor,
                is_train: bool = True):
        """

        :param sentence_repr: (N, L, E)
        :param entity_span_indices: (N, M, HT, 2)
        :param head_mentions_indices:      (N, R, H)
        :param head_mentions_indices_mask: (N, R, H)
        :param tail_mentions_indices:      (N, R, T)
        :param tail_mentions_indices_mask: (N, R, T)
        :param ht_comb_indices: (N, R, HT, 2)
        :param ht_comb_mask:    (N, R, HT)
        :param dis_h_2_t:       (N, R, HT)
        :param dis_t_2_h:       (N, R, HT)
        :param relation_mask:  (N, R)
        :param relation_label: (N, R)
        :param is_train: True or False
        :return:
        """

        # Shape: (batch_size, num_spans, embedding_dim)
        span_embeddings = self._entity_spans_embeddings(sentence_repr, entity_span_indices)
        del entity_span_indices

        # obtain head_mentions_embeddings, tail_mentions_embeddings
        # (N, R, T, E)
        head_mentions_embeddings = util.batched_index_select(span_embeddings, head_mentions_indices)
        tail_mentions_embeddings = util.batched_index_select(span_embeddings, tail_mentions_indices)
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
        weights_to_return = {
            "hAt_weights": None,  # mutual attend: head attend tail (N, R, H, T)
            "tAh_weights": None,  # mutual attend: tail attend head (N, R, T, H)
            "weights_on_mp": None  # Mention pairs self-attend:     (N, R, HT, HT)
        }
        if self.mutual_attender == CST.mutual_attender['ML-MMA']:
            # (NR, H, E), (NR, T, E)
            attended_head, hAt_weights, attended_tail, tAh_weights = self.attention(head_mentions_embeddings,
                                                                                    tail_mentions_embeddings, dis_h_2_t,
                                                                                    head_mentions_indices_mask,
                                                                                    tail_mentions_indices_mask, NRHET,
                                                                                    True)
            weights_to_return["hAt_weights"] = hAt_weights.view(N, R, H, T)
            weights_to_return["tAh_weights"] = tAh_weights.view(N, R, T, H)
        elif self.mutual_attender == CST.mutual_attender['NONE']:
            attended_head = head_mentions_embeddings
            attended_tail = tail_mentions_embeddings
        else:
            raise Exception(f'mutual_attender ERROR in {self.__class__.__name__} forward.')
        del head_mentions_embeddings, tail_mentions_embeddings

        head_indices = ht_comb_indices[:, :, :, 0].view(NR, HT)
        tail_indices = ht_comb_indices[:, :, :, 1].view(NR, HT)
        del ht_comb_indices

        # repr(NR, H, E) indices(NR, HT) --> (NR, HT, E)
        attended_head = util.batched_index_select(attended_head, head_indices)
        # repr(NR, T, E) indices(NR, HT) --> (NR, HT, E)
        attended_tail = util.batched_index_select(attended_tail, tail_indices)

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
        if self.integration_attender == CST.integration_attender['NONE']:
            pass
        elif self.integration_attender == CST.integration_attender['ML-MA']:
            rel_features = rel_features.permute(1, 0, 2)
            # weights (N, L(tgt), S(src))  (N, R, HT, HT)
            rel_features, attn_weights = self.transformer_encoder(rel_features,
                                                                  src_key_padding_mask=(
                                                                              1 - ht_comb_mask.view(NR, HT)).to(
                                                                      dtype=torch.bool))
            weights_to_return["weights_on_mp"] = attn_weights.view(N, R, HT, HT)
            rel_features = rel_features.permute(1, 0, 2)
        else:
            raise Exception(f'integration_attender ERROR in {self.__class__.__name__} forward.')

        # LogSumExp()
        # (N, R, HT, C)
        logits = self.final_proj(rel_features).view(N, R, HT, -1)
        del rel_features
        logits = logits - (1-ht_comb_mask.unsqueeze(-1).float())*1e10
        logits = torch.logsumexp(logits, dim=2)

        return logits, weights_to_return


class BRAN_M(nn.Module):
    def __init__(self, in_features: int, dis_size: int, relation_num: int,
                 entity_span_pooling: str = f"{CST.pooling_style['mean']}",
                 bi_affine_dropout: float = 0.,
                 use_distance: bool = True,
                 logging=None,
                 dataset: str = CST.datasets['DocRED']):
        super(BRAN_M, self).__init__()
        self.use_distance = use_distance
        self.num_relation = relation_num
        self.in_features = in_features
        self.dis_size = dis_size
        self.dataset = dataset
        self.entity_span_pooling = entity_span_pooling
        self._entity_spans_embeddings = SpanExtractor(in_features, entity_span_pooling)

        self.mlp2head = TwoLayerLinear(input_dim=in_features,
                                       hidden_dim=in_features,
                                       activation=torch.relu,
                                       out_dim=in_features,
                                       bias=(False, False))
        self.mlp2tail = TwoLayerLinear(input_dim=in_features,
                                       hidden_dim=in_features,
                                       activation=torch.relu,
                                       out_dim=in_features,
                                       bias=(False, False))

        self.head_drop = None
        if bi_affine_dropout > 0.:
            self.head_drop = nn.Dropout(bi_affine_dropout)
            self.tail_drop = nn.Dropout(bi_affine_dropout)

        in_for_bili = in_features
        if self.use_distance:
            in_for_bili += self.dis_size
            self.dis_embed = self.dis_embed = nn.Embedding(20, dis_size, padding_idx=10)
        self.bili = nn.Bilinear(in_for_bili, in_for_bili, self.num_relation)

        self.print_config(logging)

    def print_config(self, logging):
        log_str = f"\nClass {self.__class__.__name__} constructor.\n" \
            f"drop: {self.head_drop}\n" \
            f"use_distance: {self.use_distance}\n" \
            f"dis_size: {self.dis_size}\n"

        if logging is not None:
            logging(log_str)
        else:
            print(log_str)

    def forward(self,
                sentence_repr: torch.Tensor,
                entity_span_indices: torch.LongTensor,
                head_mentions_indices: torch.LongTensor,
                head_mentions_indices_mask: torch.LongTensor,
                tail_mentions_indices: torch.LongTensor,
                tail_mentions_indices_mask: torch.LongTensor,
                ht_comb_indices: torch.LongTensor,
                ht_comb_mask: torch.LongTensor,
                dis_h_2_t: torch.LongTensor,
                dis_t_2_h: torch.LongTensor,
                relation_mask: torch.LongTensor,
                relation_label: torch.LongTensor,
                is_train: bool = True):
        """

        :param sentence_repr: (N, L, E)
        :param entity_span_indices: (N, M, HT, 2)
        :param head_mentions_indices:      (N, R, H)
        :param head_mentions_indices_mask: (N, R, H)
        :param tail_mentions_indices:      (N, R, T)
        :param tail_mentions_indices_mask: (N, R, T)
        :param ht_comb_indices: (N, R, HT, 2)
        :param ht_comb_mask:    (N, R, HT)
        :param dis_h_2_t:       (N, R, HT)
        :param dis_t_2_h:       (N, R, HT)
        :param relation_mask:  (N, R)
        :param relation_label: (N, R)
        :param is_train: True or False
        :return:
        """

        # Shape: (batch_size, num_spans, embedding_dim)
        span_embeddings = self._entity_spans_embeddings(sentence_repr, entity_span_indices)
        del entity_span_indices

        # obtain head_mentions_embeddings, tail_mentions_embeddings
        # (N, R, T, E)
        head_mentions_embeddings = \
            util.batched_index_select(self.mlp2head(span_embeddings),
                                      head_mentions_indices)
        tail_mentions_embeddings = \
            util.batched_index_select(self.mlp2tail(span_embeddings),
                                      tail_mentions_indices)
        if self.head_drop is not None:
            head_mentions_embeddings = self.head_drop(head_mentions_embeddings)
            tail_mentions_embeddings = self.tail_drop(tail_mentions_embeddings)

        head_mentions_embeddings = head_mentions_embeddings*head_mentions_indices_mask.unsqueeze(-1).float()
        tail_mentions_embeddings = tail_mentions_embeddings*tail_mentions_indices_mask.unsqueeze(-1).float()
        # del head_mentions_indices_mask
        N, R, H, E = head_mentions_embeddings.size()
        T = tail_mentions_embeddings.size(2)
        NR = N*R
        HT = ht_comb_mask.size(2)

        head_mentions_embeddings = head_mentions_embeddings.view(N*R, H, E)
        tail_mentions_embeddings = tail_mentions_embeddings.view(N*R, T, E)

        head_indices = ht_comb_indices[:, :, :, 0].view(NR, HT)
        tail_indices = ht_comb_indices[:, :, :, 1].view(NR, HT)
        del ht_comb_indices
        # repr(NR, H, E) indices(NR, HT) --> (NR, HT, E)
        head_mentions_embeddings = util.batched_index_select(head_mentions_embeddings, head_indices)
        # repr(NR, T, E) indices(NR, HT) --> (NR, HT, E)
        tail_mentions_embeddings = util.batched_index_select(tail_mentions_embeddings, tail_indices)

        if self.use_distance:
            # Shape: (N, R*HT, E+dis_size)
            head_mentions_embeddings = torch.cat([head_mentions_embeddings, self.dis_embed(dis_h_2_t.view(N * R, HT))], dim=2)
            tail_mentions_embeddings = torch.cat([tail_mentions_embeddings, self.dis_embed(dis_t_2_h.view(N * R, HT))], dim=2)

        # (N, R*HT, num_relation)
        logits = self.bili(head_mentions_embeddings, tail_mentions_embeddings)
        del head_mentions_embeddings, tail_mentions_embeddings
        logits = logits.view(N, R, HT, self.num_relation)

        # Mask out padding
        logits = logits - (1 - ht_comb_mask.unsqueeze(-1).float()) * 1e10
        # (N, R, num_relation)
        logits = torch.logsumexp(logits, dim=2)

        return logits, None


class BiLSTM_M(nn.Module):

    def __init__(self, in_features, entity_span_pooling, coref_pooling,
                 dis_size,
                 use_distance=True,
                 use_bilinear=True,
                 relation_num=97,
                 hidden_dim=384):
        """
        :param in_features: last dim of input sequence_tensor
            output shape (..., input_dim) if True and coref_pooling=='mean-max',
            otherwise (..., input_dim*2)
        """
        super(BiLSTM_M, self).__init__()
        self.use_distance = use_distance
        self.dis_size = dis_size
        self.use_bilinear = use_bilinear
        self.relation_num = relation_num

        self.entity_span_pooling = entity_span_pooling
        self.coref_pooling = coref_pooling
        self._entity_spans_embeddings = SpanExtractor(in_features, entity_span_pooling)

        if self.coref_pooling == CST.pooling_style['attn']:
            self._global_attention = TwoLayerLinear(input_dim=in_features)

        final_in_features = in_features
        if self.use_distance:
            self.dis_embed = nn.Embedding(20, self.dis_size, padding_idx=10)
            final_in_features += self.dis_size

        self.hidden_dim = hidden_dim

        if self.use_bilinear:
            self.bili = torch.nn.Bilinear(final_in_features, final_in_features, self.relation_num)
        else:
            self.final_projection = TwoLayerLinear(input_dim=final_in_features*2+in_features, hidden_dim=hidden_dim,
                                                   activation=torch.relu, out_dim=self.relation_num, bias=(False, True))

        print(f'entity_span_pooling pooling: =={entity_span_pooling}==')
        print(f'co-references pooling:       =={coref_pooling}==')

    def forward(self,
                sentence_repr: torch.Tensor,
                entity_span_indices: torch.LongTensor,
                entity_span_indices_mask: torch.LongTensor,
                vertex_indices: torch.LongTensor,
                vertex_indices_mask: torch.LongTensor,
                head_tail_indices: torch.LongTensor,
                relation_mask: torch.LongTensor,
                dis_h_2_t,
                dis_t_2_h):
        """
        :param dis_t_2_h: (N, R, HT)
        :param dis_h_2_t: (N, R, HT)
        :param sentence_repr: (batch_size, seq_len, embed_dim)
        :param entity_span_indices: (batch_size, num_spans, 2)
        :param entity_span_indices_mask: (batch_size, num_spans)
        :param vertex_indices: (batch_size, num_vertexes, num_max_corefs_per_vertex)
        :param vertex_indices_mask: (batch_size, num_vertexes, num_max_corefs_per_vertex)
        :param head_tail_indices: (batch_size, num_max_rels_per_example, 2)
        :param relation_mask: (batch_size, num_max_rels_per_example)

        :return: (batch_size, num_max_rels_per_example, embed_dim) if do_coref_down_project=True and coref_pooling == 'mean-max'
                 (batch_size, num_max_rels_per_example, embed_dim*2)
        """
        # Shape: (batch_size, num_spans, embedding_dim)
        span_embeddings = self._entity_spans_embeddings(sentence_repr, entity_span_indices)

        flat_vertex_indices = util.flatten_and_batch_shift_indices(vertex_indices, span_embeddings.size(1))
        # Shape:               (batch_size, num_vertexes, num_max_corefs_per_vertex, embedding_dim)
        vertex_embeddings = util.batched_index_select(span_embeddings, vertex_indices, flat_vertex_indices)
        # vertex_indices_mask: (batch_size, num_vertexes, num_max_corefs_per_vertex)

        if self.coref_pooling == CST.pooling_style['max']:
            vertex_embeddings = vertex_embeddings - (1-vertex_indices_mask.unsqueeze(-1).float())*1e32
            vertex_embeddings = torch.max(vertex_embeddings, dim=2)[0]
        elif self.coref_pooling == CST.pooling_style['mean']:  # mean pooling of entity spans in vertex
            vertex_embeddings = vertex_embeddings * vertex_indices_mask.unsqueeze(-1).float()
            # Shape: (batch_size, num_vertexes)
            num_corefs = vertex_indices_mask.sum(dim=2)
            num_corefs = num_corefs + (num_corefs < 1).long()  # +1 for padding entity
            # Shape: (batch_size, num_vertexes, embedding_dim)
            vertex_embeddings = torch.sum(vertex_embeddings, dim=2) / num_corefs.unsqueeze(-1).float()
        elif self.coref_pooling == CST.pooling_style['attn']:  # attention pooling
            # Shape (batch_size, num_spans, 1)
            global_attention_logits = self._global_attention(span_embeddings)
            # Shape: (batch_size, num_vertexes, num_max_corefs_per_vertex)
            span_attention_logits = util.batched_index_select(global_attention_logits, vertex_indices, flat_vertex_indices).squeeze(-1)
            # Shape: (batch_size, num_vertexes, num_max_corefs_per_vertex)
            span_attention_weights = util.masked_softmax(span_attention_logits, vertex_indices_mask, memory_efficient=True)
            # Shape: (batch_size, num_vertexes, embedding_dim)
            vertex_embeddings = util.weighted_sum(vertex_embeddings, span_attention_weights)
        else:
            raise Exception(f"co-reference style param error, must be in {'; '.join(CST.pooling_style.values())}")

        # or Shape: (batch_size, num_max_rels_per_example, 2, embedding_dim*2)
        head_tail_embeddings = util.batched_index_select(vertex_embeddings, head_tail_indices)
        head_tail_embeddings * relation_mask.unsqueeze(-1).unsqueeze(-1).float()
        # Shape: (batch_size, num_max_rels_per_example, embedding_dim or embedding_dim*2)
        head_repr = head_tail_embeddings[:, :, 0, :]
        tail_repr = head_tail_embeddings[:, :, 1, :]

        if self.use_bilinear:
            if self.use_distance:
                head_repr = torch.cat([head_repr, self.dis_embed(dis_h_2_t)], dim=-1)
                tail_repr = torch.cat([tail_repr, self.dis_embed(dis_t_2_h)], dim=-1)
            predict_re = self.bili(head_repr, tail_repr)
        else:
            if self.use_distance:
                predict_re = self.final_projection(torch.cat([head_repr, self.dis_embed(dis_h_2_t),
                                                              tail_repr, self.dis_embed(dis_t_2_h),
                                                              head_repr * tail_repr], dim=-1))
            else:
                predict_re = self.final_projection(torch.cat([head_repr,
                                                              tail_repr,
                                                              head_repr * tail_repr], dim=-1))

        return predict_re, None


class GCNN(nn.Module):
    def __init__(self, config):
        super(GCNN, self).__init__()
        # Word embeddings
        if not config.use_bert_embedding:
            num_embeddings = config.data_word_vec.shape[0]
            wv_dim = config.data_word_vec.shape[1]
            config.logging(f'Use GloVe embeddings. vec_size: {wv_dim}; num_embeddings: {num_embeddings}')
            self.word_emb = nn.Embedding(num_embeddings, wv_dim)
            self.word_emb.weight.data.copy_(torch.from_numpy(config.data_word_vec))
            self.word_emb.weight.requires_grad = not config.freeze_embedd
        else:
            wv_dim = config.bert_embedd_dim
            config.logging(f'Use BERT embeddings. vec_size: {wv_dim}.')
        # Embedding Projection
        self.emb_proj = None
        if wv_dim != config.embedd_dim:
            self.emb_proj = nn.Linear(wv_dim, config.embedd_dim, bias=False)
            config.logging(f"Project embeddings to {config.embedd_dim} dim.")
        # Relative position embedding
        self.position_emb = nn.Embedding(config.num_position_embeddings, config.position_dim,
                                         padding_idx=config.pad_position_idx)
        self.inp_dropout = None
        if config.inp_dropout > 0.:
            self.inp_dropout = nn.Dropout(config.inp_dropout)
        # GCN encoder
        in_features = config.embedd_dim + 2*config.position_dim
        self.gcn_encoder = GCNN_Encoder(in_features=in_features,
                                        gcn_dim=config.gcn_dim,
                                        num_unRare_edge_types=config.num_unRare_edge_types,
                                        num_all_edge_types=config.num_edge_types,
                                        num_blocks=config.num_blocks,
                                        use_gate=config.gcn_use_gate,
                                        dropout=config.gcn_dropout,
                                        residual=config.gcn_residual,
                                        activation=torch.relu,
                                        _parallel_forward=config.parallel_forward,
                                        logging=config.logging)
        # bi-affine pairwise scoring
        self.bi_affine = BiAffineForward(in_features=in_features,
                                         relation_num=config.relation_num,
                                         entity_span_pooling=config.entity_span_pooling,
                                         ff_dim=config.bi_affine_ff_dim,
                                         ff_dropout=config.bi_affine_dropout)

    def forward(self, context_idxs, input_lengths, dist_e1, dist_e2,
                adj_matrices, for_relation_repr, bert_feature=None):
        document = self.word_emb(context_idxs) if bert_feature is None else bert_feature
        if self.emb_proj is not None:
            document = self.emb_proj(document)
        document = torch.cat([document, self.position_emb(dist_e1), self.position_emb(dist_e2)], dim=-1)
        if self.inp_dropout is not None:
            document = self.inp_dropout(document)

        document = self.gcn_encoder(document, adj_matrices)  # forward or forward_broadcast
        logits = self.bi_affine(sentence_repr=document, **for_relation_repr)
        return logits


class GCNN_Encoder(nn.Module):
    def __init__(self,
                 in_features: int = 140,
                 gcn_dim: int = 140,
                 num_unRare_edge_types: int = 4,
                 num_all_edge_types: int = 10,
                 num_blocks: int = 2,
                 use_gate: int = True,
                 dropout: float = 0.05,
                 residual: bool = True,
                 activation=torch.relu,
                 _parallel_forward: bool = True,
                 logging=None):
        super(GCNN_Encoder, self).__init__()
        self.in_features = in_features
        self.gcn_dim = gcn_dim
        self.num_unRare_edge_types = num_unRare_edge_types
        self.num_all_edge_types = num_all_edge_types
        self.num_blocks = num_blocks
        self.gate = use_gate
        self.residual = residual
        self.drop_p = dropout
        self.use_drop = False
        if self.drop_p > 0.:
            self.use_drop = True
            self.dropout = nn.Dropout(dropout)
        self.activation = activation
        Linear = OurLinear

        if self.residual:
            assert self.gcn_dim == self.in_features, f"Now residual=True, but gcn_dim != in_features."

        self._parallel = _parallel_forward
        if not self._parallel:  # broadcast_forward()
            self.in_proj = nn.ModuleList()  # in_arc, The first two are self-node edge and adjacent word
            self.out_proj = nn.ModuleList()  # out_arc, The first two are self-node edge and adjacent word
            if self.gate:
                self.in_gate_proj = nn.ModuleList()
                self.out_gate_proj = nn.ModuleList()
                self.sigmoid = nn.Sigmoid()
            for layer_idx in range(self.num_blocks):
                _gcn_in_dim = self.in_features if layer_idx == 0 else self.gcn_dim
                _gcn_out_dim = self.gcn_dim
                this_layer_in_proj = nn.ModuleList()  # in_arc
                this_layer_out_proj = nn.ModuleList()  # out_arc
                if self.gate:
                    this_layer_in_gate = nn.ModuleList()
                    this_layer_out_gate = nn.ModuleList()
                for et_idx in range(self.num_all_edge_types):
                    this_layer_in_proj.append(
                        Linear(_gcn_in_dim, _gcn_out_dim) if et_idx <= self.num_unRare_edge_types
                        else this_layer_in_proj[-1]
                    )
                    this_layer_out_proj.append(
                        Linear(_gcn_in_dim, _gcn_out_dim) if et_idx <= self.num_unRare_edge_types
                        else this_layer_out_proj[-1]
                    )
                    if self.gate:
                        this_layer_in_gate.append(
                            Linear(_gcn_in_dim, 1) if et_idx <= self.num_unRare_edge_types
                            else this_layer_in_gate[-1]
                        )
                        this_layer_out_gate.append(
                            Linear(_gcn_in_dim, 1) if et_idx <= self.num_unRare_edge_types
                            else this_layer_out_gate[-1]
                        )
                self.in_proj.append(this_layer_in_proj)
                self.out_proj.append(this_layer_out_proj)
                if self.gate:
                    self.in_gate_proj.append(this_layer_in_gate)
                    self.out_gate_proj.append(this_layer_out_gate)

        else:  # parallel_forward()
            self.W_in = nn.ModuleList()
            self.W_out = nn.ModuleList()
            if self.gate:
                self.W_in_gate = nn.ModuleList()
                self.W_out_gate = nn.ModuleList()
                self.sigmoid = nn.Sigmoid()
            if self.num_all_edge_types - self.num_unRare_edge_types > 0:
                self.rare_W_in = nn.ModuleList()
                self.rare_W_out = nn.ModuleList()
                if self.gate:
                    self.rare_W_in_gate = nn.ModuleList()
                    self.rare_W_out_gate = nn.ModuleList()
            for layer_idx in range(self.num_blocks):
                _gcn_in_dim = self.in_features if layer_idx == 0 else self.gcn_dim
                self.W_in.append(nn.Linear(_gcn_in_dim, self.gcn_dim*self.num_unRare_edge_types))
                self.W_out.append(nn.Linear(_gcn_in_dim, self.gcn_dim*self.num_unRare_edge_types))
                if self.gate:
                    self.W_in_gate.append(nn.Linear(_gcn_in_dim, 1*self.num_unRare_edge_types))
                    self.W_out_gate.append(nn.Linear(_gcn_in_dim, 1*self.num_unRare_edge_types))

                if self.num_all_edge_types - self.num_unRare_edge_types > 0:
                    self.rare_W_in.append(nn.Linear(_gcn_in_dim, self.gcn_dim))
                    self.rare_W_out.append(nn.Linear(_gcn_in_dim, self.gcn_dim))
                    if self.gate:
                        self.rare_W_in_gate.append(nn.Linear(_gcn_in_dim, 1))
                        self.rare_W_out_gate.append(nn.Linear(_gcn_in_dim, 1))

        self.print_config(logging)

    def print_config(self, logging=None):
        attrs = ['in_features', 'gcn_dim', 'num_unRare_edge_types', 'num_all_edge_types',
                 'num_blocks', 'gate', 'use_drop', 'drop_p', 'residual']
        config_lines = list()
        for attr in attrs:
            value = self.__getattribute__(attr)
            config_lines.append("{:<23}: {}\n".format(attr, value))
        log_str = "".join(config_lines)
        if logging is not None:
            logging(log_str)
        else:
            print(log_str)

    def forward1(self,
                 seq_repr: torch.Tensor,
                 adj: torch.LongTensor) -> List[torch.Tensor]:
        """
            Encode the document using stack of GCN layers.
        :param seq_repr: (B, L, E)
        :param adj: Adjacency matrix. (B, num_edge_types, L, L)
             the first `num_unRare_edge_types` (num_edge_types>num_unRare_edge_types) labels have separate parameters,
             and the rest use the same parameters.
             L: num_nodes, here we consider each token as a node, i.e., num_nodes = seq_len
        :return: (B, L, gcn_dim), employ `num_blocks` GCN layers on the seq_repr, and output the encoded repr.
        """
        # gcn_out = seq_repr
        out = [seq_repr]
        B, L, E = seq_repr.size()
        D = self.gcn_dim
        for layer_idx in range(self.num_blocks):  # each GCN layer
            gcn_inp = out[-1]
            act_sum = torch.zeros((B, L, D), dtype=seq_repr.dtype,
                                  layout=seq_repr.layout, device=seq_repr.device)
            for et_idx in range(self.num_all_edge_types):  # each edge type
                in_arc = self.in_proj[layer_idx][et_idx](gcn_inp)
                # (B, L, L) * (B, L, D) --> (B, L, D)
                in_arc = torch.matmul(adj[:, et_idx, :, :].to(torch.float), in_arc)
                if self.use_drop:
                    in_arc = self.dropout(in_arc)
                if self.gate:
                    in_arc_gate = self.in_gate_proj[layer_idx][et_idx](gcn_inp)
                    # (B, L, L) * (B, L, D) --> (B, L, D)
                    in_arc_gate = torch.matmul(adj[:, et_idx, :, :].to(torch.float), in_arc_gate)
                    in_arc_gate = self.sigmoid(in_arc_gate)
                    in_arc = in_arc * in_arc_gate

                out_arc = self.out_proj[layer_idx][et_idx](gcn_inp)
                out_arc = torch.matmul(adj[:, et_idx, :, :].permute(0, 2, 1).to(torch.float), out_arc)
                if self.use_drop:
                    out_arc = self.dropout(out_arc)
                if self.gate:
                    out_arc_gate = self.out_gate_proj[layer_idx][et_idx](gcn_inp)
                    out_arc_gate = torch.matmul(adj[:, et_idx, :, :].permute(0, 2, 1).to(torch.float), out_arc_gate)
                    out_arc_gate = self.sigmoid(out_arc_gate)
                    out_arc = out_arc * out_arc_gate
                act_sum = act_sum + in_arc + out_arc
            if self.residual:
                act_sum = act_sum + gcn_inp
            gcn_out = self.activation(act_sum)

            out.append(gcn_out)

        return out[-1]

    def forward(self,
                seq_repr: torch.Tensor,
                adj: torch.LongTensor) -> List[torch.Tensor]:
        if self._parallel:
            return self.parallel_forward(seq_repr, adj)
        else:
            return self.broadcast_forward(seq_repr, adj)

    def broadcast_forward(self,
                          seq_repr: torch.Tensor,
                          adj: torch.LongTensor) -> List[torch.Tensor]:
        """
            Encode the document using stack of GCN layers.
        :param seq_repr: (B, L, E)
        :param adj: Adjacency matrix. (B, num_edge_types, L, L)
             the first `num_unRare_edge_types` (num_edge_types>num_unRare_edge_types) labels have separate parameters,
             and the rest use the same parameters.
             L: num_nodes, here we consider each token as a node, i.e., num_nodes = seq_len
        :return: (B, L, gcn_dim), employ `num_blocks` GCN layers on the seq_repr, and output the encoded repr.
        """
        out = [seq_repr]
        # gcn_out = seq_repr
        B, L, E = seq_repr.size()
        D = self.gcn_dim
        for layer_idx in range(self.num_blocks):  # each GCN layer
            gcn_inp = out[-1]
            act_sum = None
            for et_idx in range(self.num_unRare_edge_types):  # each edge type
                in_arc = self.in_proj[layer_idx][et_idx](gcn_inp)
                # (B, L, L) * (B, L, D) --> (B, L, D)
                in_arc = torch.matmul(adj[:, et_idx, :, :].to(torch.float), in_arc)
                if self.use_drop:
                    in_arc = self.dropout(in_arc)
                if self.gate:
                    in_arc_gate = self.in_gate_proj[layer_idx][et_idx](gcn_inp)
                    # (B, L, L) * (B, L, D) --> (B, L, D)
                    in_arc_gate = torch.matmul(adj[:, et_idx, :, :].to(torch.float), in_arc_gate)
                    in_arc_gate = self.sigmoid(in_arc_gate)
                    in_arc = in_arc * in_arc_gate

                out_arc = self.out_proj[layer_idx][et_idx](gcn_inp)
                out_arc = torch.matmul(adj[:, et_idx, :, :].transpose(1, 2).to(torch.float), out_arc)
                if self.use_drop:
                    out_arc = self.dropout(out_arc)
                if self.gate:
                    out_arc_gate = self.out_gate_proj[layer_idx][et_idx](gcn_inp)
                    out_arc_gate = torch.matmul(adj[:, et_idx, :, :].permute(0, 2, 1).to(torch.float), out_arc_gate)
                    out_arc_gate = self.sigmoid(out_arc_gate)
                    out_arc = out_arc * out_arc_gate
                if act_sum is None:
                    act_sum = in_arc + out_arc
                else:
                    act_sum = act_sum + in_arc + out_arc

            Rest = self.num_all_edge_types-self.num_unRare_edge_types
            if Rest > 0:
                et_idx = self.num_unRare_edge_types
                # (B, L, D)
                in_arc = self.in_proj[layer_idx][et_idx](gcn_inp)
                # (B, Rest, L, L) * (B, 1, L, D) --> (B, Rest, L, D)
                in_arc = torch.matmul(adj[:, et_idx:, :, :].to(torch.float), in_arc.unsqueeze(1))
                assert in_arc.size() == (B, Rest, L, D)
                if self.use_drop:
                    in_arc = self.dropout(in_arc)
                if self.gate:
                    in_arc_gate = self.in_gate_proj[layer_idx][et_idx](gcn_inp)
                    # (B, Rest, L, L) * (B, 1, L, 1) --> (B, Rest, L, D)
                    in_arc_gate = torch.matmul(adj[:, et_idx:, :, :].to(torch.float), in_arc_gate.unsqueeze(1))
                    in_arc_gate = self.sigmoid(in_arc_gate)
                    in_arc = in_arc * in_arc_gate
                # (B, Rest, L, D) --> (B, L, D)
                in_arc = in_arc.sum(dim=1)

                out_arc = self.out_proj[layer_idx][et_idx](gcn_inp)
                out_arc = torch.matmul(adj[:, et_idx:, :, :].permute(0, 1, 3, 2).to(torch.float), out_arc.unsqueeze(1))
                assert out_arc.size() == (B, Rest, L, D)
                if self.use_drop:
                    out_arc = self.dropout(out_arc)
                if self.gate:
                    out_arc_gate = self.out_gate_proj[layer_idx][et_idx](gcn_inp)
                    out_arc_gate = torch.matmul(adj[:, et_idx:, :, :].permute(0, 1, 3, 2).to(torch.float), out_arc_gate.unsqueeze(1))
                    assert out_arc_gate.size() == (B, Rest, L, 1)
                    out_arc_gate = self.sigmoid(out_arc_gate)
                    out_arc = out_arc * out_arc_gate
                assert out_arc.size() == (B, Rest, L, D)
                out_arc = out_arc.sum(dim=1)
                act_sum = act_sum + in_arc + out_arc

            if self.residual:
                act_sum = act_sum + gcn_inp
            gcn_out = self.activation(act_sum)
            out.append(gcn_out)

        return out[-1]

    def forward2(self,
                 seq_repr: torch.Tensor,
                 adj: torch.LongTensor) -> torch.Tensor:
        B, L, E = seq_repr.size()
        D = self.gcn_dim
        out_exs = seq_repr
        for layer_idx in range(self.num_blocks):
            gcn_in = out_exs
            out_exs = []
            for ex_idx in range(B):  # for each example
                out_tokens = []
                for tok_idx in range(L):  # each token in the document
                    token_addition = torch.zeros((D), dtype=seq_repr.dtype,
                                                 layout=seq_repr.layout, device=seq_repr.device)
                    for edge_type_idx in range(self.num_all_edge_types):
                        for j in range(L):
                            if adj[ex_idx, edge_type_idx, tok_idx, j].data == 1:
                                addition = self.in_proj[layer_idx][edge_type_idx](gcn_in[ex_idx, j])
                                if self.gate:
                                    in_arc_gate = self.in_gate_proj[layer_idx][edge_type_idx](gcn_in[ex_idx, j])
                                    in_arc_gate = self.sigmoid(in_arc_gate)
                                    addition = addition * in_arc_gate
                                token_addition = token_addition + addition
                            if adj[ex_idx, edge_type_idx, j, tok_idx].data == 1:
                                addition = self.out_proj[layer_idx][edge_type_idx](gcn_in[ex_idx, j])
                                if self.gate:
                                    out_arc_gate = self.out_gate_proj[layer_idx][edge_type_idx](gcn_in[ex_idx, j])
                                    out_arc_gate = self.sigmoid(out_arc_gate)
                                    addition = addition * out_arc_gate
                                token_addition = token_addition + addition
                    out_tokens.append(token_addition)
                out_tokens = torch.stack(out_tokens, dim=0)
                # print(out_tokens)
                out_exs.append(out_tokens)
            out_exs = torch.stack(out_exs, dim=0)
            if self.activation is not None:
                out_exs = self.activation(out_exs)
            # print(out_exs)
            if out_exs.size() != (B, L, D):
                print(f"layer_idx: {layer_idx}")
                print(f"out_exs.size() != (B, L, D): {out_exs.size()} != {(B, L, D)}")

        return out_exs

    def parallel_forward(self, seq_repr, adj):
        # This implementation is much faster.
        # The first edge types use a single Linear (equivalent to concatenation of several Linear s).
        # The first two `un_rare_propagate()` call is equivalent to
        # the loop `for et_idx in range(self.num_unRare_edge_types)` in `forward_broadcast()`.
        out = [seq_repr]
        B, L, E = seq_repr.size()
        U = self.num_unRare_edge_types
        Rest = self.num_all_edge_types-self.num_unRare_edge_types
        D = self.gcn_dim
        for layer_idx in range(self.num_blocks):
            gcn_in = out[-1]
            # high frequency edge types use separate weight matrices for each edge type
            # in arc
            _adj_ = adj[:, :U, :, :].to(torch.float)
            in_arc = self.un_rare_propagate(gcn_in=gcn_in,
                                            proj=self.W_in[layer_idx],
                                            gate_proj=self.W_in_gate[layer_idx] if self.gate else None,
                                            adj=_adj_,
                                            BULD=(B, U, L, D))
            # out arc, transpose the adjacent matrices
            _adj_ = adj[:, :U, :, :].permute(0, 1, 3, 2).to(torch.float)
            out_arc = self.un_rare_propagate(gcn_in=gcn_in,
                                             proj=self.W_out[layer_idx],
                                             gate_proj=self.W_out_gate[layer_idx] if self.gate else None,
                                             adj=_adj_,
                                             BULD=(B, U, L, D))
            gcn_out = in_arc + out_arc

            # The rest rare (lower frequency) edge types share a same weight matrix
            if Rest > 0:
                _adj_ = adj[:, U:, :, :].to(torch.float)
                in_arc = self.rest_propagate(gcn_in=gcn_in,
                                             proj=self.rare_W_in[layer_idx],
                                             gate_proj=self.rare_W_in_gate[layer_idx] if self.gate else None,
                                             adj=_adj_,
                                             BRLD=(B, Rest, L, D))
                _adj_ = adj[:, U:, :, :].permute(0, 1, 3, 2).to(torch.float)
                out_arc = self.rest_propagate(gcn_in=gcn_in,
                                              proj=self.rare_W_out[layer_idx],
                                              gate_proj=self.rare_W_out_gate[layer_idx] if self.gate else None,
                                              adj=_adj_,
                                              BRLD=(B, Rest, L, D))
                gcn_out = gcn_out + in_arc + out_arc

            if self.residual:
                gcn_out = gcn_out + gcn_in
            gcn_out = self.activation(gcn_out)
            out.append(gcn_out)

        return out[-1]

    def un_rare_propagate(self,        # type: GCNN_Encoder
                          gcn_in,      # type: torch.Tensor
                          proj,        # type: nn.Module
                          gate_proj,   # type: nn.Module
                          adj,         # type: torch.Tensor
                          BULD         # type: Tuple
                          ):
        # type: (...) -> torch.Tensor
        B, U, L, D = BULD
        # (B, L, D*U)
        out_arc = proj(gcn_in)
        assert out_arc.size() == (B, L, D*U)
        # (B, U, L, D)
        out_arc = out_arc.view(B, L, U, D).contiguous().permute(0, 2, 1, 3)
        # (B, U, L, D) <= (B, U, L, L) x (B, U, L, D)
        out_arc = torch.matmul(adj, out_arc)
        assert out_arc.size() == (B, U, L, D)
        if self.use_drop:
            out_arc = self.dropout(out_arc)
        if self.gate:
            # (B, L, U)
            out_arc_gate = gate_proj(gcn_in)
            # (B, U, L, 1)
            out_arc_gate = out_arc_gate.view(B, L, U, 1).contiguous().permute(0, 2, 1, 3)
            # (B, U, L, 1) <= (B, U, L, L) x (B, U, L, 1)
            out_arc_gate = torch.matmul(adj, out_arc_gate)
            assert out_arc_gate.size() == (B, U, L, 1)
            out_arc_gate = self.sigmoid(out_arc_gate)
            # (B, U, L, D) <= (B, U, L, D)*(B, U, L, 1)
            out_arc = out_arc * out_arc_gate
        # summation over different edge types
        # (B, L, D) <= (B, U, L, D)
        out_arc = out_arc.sum(dim=1)

        return out_arc

    def rest_propagate(self,        # type: GCNN_Encoder
                       gcn_in,      # type: torch.Tensor
                       proj,        # type: nn.Module
                       gate_proj,   # type: nn.Module
                       adj,         # type: torch.Tensor
                       BRLD         # type: Tuple
                       ):
        # type: (...) -> torch.Tensor
        B, Rest, L, D = BRLD
        in_arc = proj(gcn_in)
        # (B, Rest, L, L) * (B, 1, L, D) --> (B, Rest, L, D)
        in_arc = torch.matmul(adj, in_arc.unsqueeze(1))
        assert in_arc.size() == (B, Rest, L, D)
        if self.use_drop:
            in_arc = self.dropout(in_arc)
        if self.gate:
            in_arc_gate = gate_proj(gcn_in)
            # (B, Rest, L, 1) <= (B, Rest, L, L) x (B, 1, L, 1)
            in_arc_gate = torch.matmul(adj, in_arc_gate.unsqueeze(1))
            assert in_arc_gate.size() == (B, Rest, L, 1)
            in_arc_gate = self.sigmoid(in_arc_gate)
            # (B, Rest, L, D) <= (B, Rest, L, D) * (B, Rest, L, 1)
            in_arc = in_arc * in_arc_gate
        # (B, L, D) <= (B, Rest, L, D)
        in_arc = in_arc.sum(dim=1)
        return in_arc


class BiAffineForward(nn.Module):
    def __init__(self,
                 in_features: int, relation_num: int,
                 entity_span_pooling: str = f"{CST.pooling_style['mean']}",
                 ff_dim: int = 140,
                 ff_dropout: float = 0.05,
                 logging=None):
        super(BiAffineForward, self).__init__()
        self.num_relation = relation_num
        self.in_features = in_features
        self.ff_dim = ff_dim
        self.entity_spans_embeddings = SpanExtractor(in_features, entity_span_pooling)

        self.mlp2head = TwoLayerLinear(input_dim=in_features,
                                       hidden_dim=ff_dim,
                                       activation=torch.relu,
                                       out_dim=ff_dim,
                                       bias=(False, False))
        self.mlp2tail = TwoLayerLinear(input_dim=in_features,
                                       hidden_dim=ff_dim,
                                       activation=torch.relu,
                                       out_dim=ff_dim,
                                       bias=(False, False))
        self.head_drop = None
        self.tail_drop = None
        if ff_dropout > 0.:
            self.head_drop = nn.Dropout(ff_dropout)
            self.tail_drop = nn.Dropout(ff_dropout)
        self.bili = nn.Bilinear(ff_dim, ff_dim, self.num_relation)

    def forward(self,
                sentence_repr: torch.Tensor,
                entity_span_indices: torch.LongTensor,
                head_mentions_indices: torch.LongTensor,
                head_mentions_indices_mask: torch.LongTensor,
                tail_mentions_indices: torch.LongTensor,
                tail_mentions_indices_mask: torch.LongTensor,
                ht_comb_indices: torch.LongTensor,
                ht_comb_mask: torch.LongTensor):
        """

        :param sentence_repr:              (N, L, D), N=batch_size, L=document length, D=feature dimension
        :param entity_span_indices:        (N, M, 2), M=number of mentions of the target entity pair (pad to maximum)
        :param head_mentions_indices:      (N, H)   , H=number of head mentions (pad to maximum of this batch)
        :param head_mentions_indices_mask: (N, H)   , H=number of tail mentions (pad to maximum of this batch)
        :param tail_mentions_indices:      (N, T)
        :param tail_mentions_indices_mask: (N, T)
        :param ht_comb_indices:            (N, HT, 2)
        :param ht_comb_mask:               (N, HT)
        :return:
        """
        span_embeddings = self.entity_spans_embeddings(sentence_repr, entity_span_indices)
        N, H = head_mentions_indices.size()
        _, T = tail_mentions_indices.size()
        _, HT = ht_comb_mask.size()
        # two separate projection, then select head/tail mentions
        # (N, H, D)
        head_mentions_embeddings = \
            util.batched_index_select(self.mlp2head(span_embeddings),
                                      head_mentions_indices)
        # (N, T, D)
        tail_mentions_embeddings = \
            util.batched_index_select(self.mlp2tail(span_embeddings),
                                      tail_mentions_indices)

        # Dropout
        if self.head_drop is not None:
            head_mentions_embeddings = self.head_drop(head_mentions_embeddings)
            tail_mentions_embeddings = self.tail_drop(tail_mentions_embeddings)
        # mask out padding
        head_mentions_embeddings = head_mentions_embeddings*head_mentions_indices_mask.unsqueeze(-1).float()
        tail_mentions_embeddings = tail_mentions_embeddings*tail_mentions_indices_mask.unsqueeze(-1).float()

        # repr(N, H, D) indices(N, HT) --> (N, HT, D)
        head_mentions_embeddings = util.batched_index_select(head_mentions_embeddings, ht_comb_indices[:, :, 0])
        # repr(N, T, D) indices(N, HT) --> (N, HT, D)
        tail_mentions_embeddings = util.batched_index_select(tail_mentions_embeddings, ht_comb_indices[:, :, 1])

        # (N, HT, num_relation)
        logits = self.bili(head_mentions_embeddings, tail_mentions_embeddings)
        # Mask out padding
        logits = logits - (1 - ht_comb_mask.unsqueeze(-1).float()) * 1e10
        # (N, R, num_relation)
        logits = torch.logsumexp(logits, dim=1)

        return logits


def main():
    import numpy as np

    randomseed = 0
    torch.manual_seed(randomseed)
    torch.cuda.manual_seed(randomseed)
    torch.cuda.manual_seed_all(randomseed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(randomseed)
    B, L, E = 2, 3, 4
    D = 6
    num_unRare_edge_types = 4
    num_all_edge_types = 4
    num_blocks = 1
    use_gate = False
    dropout = 0.
    seq_repr = torch.randn(B, L, E).to(torch.float).cuda()
    adj = torch.zeros((B, num_all_edge_types, L, L), dtype=torch.long).cuda()
    # self-node
    for ex_idx in range(B):
        adj[ex_idx, 0].copy_(torch.from_numpy(np.eye(L)))
    # adjacent word
    for ex_idx in range(B):
        adj[ex_idx, 1, :-1, 1:].copy_(torch.from_numpy(np.eye(L-1)))
    for ex_idx in range(B):
        for j in range(2, num_all_edge_types):
            adj[ex_idx, j].copy_(torch.from_numpy(np.random.randint(0, 2, (L, L))))
    # print(seq_repr)
    # print(adj)
    gcn_encoder = GCNN_Encoder(in_features=E,
                               gcn_dim=D,
                               num_unRare_edge_types=num_unRare_edge_types,
                               num_all_edge_types=num_all_edge_types,
                               num_blocks=num_blocks,
                               use_gate=use_gate,
                               dropout=dropout,
                               residual=False,
                               activation=torch.relu).cuda()
    out1 = gcn_encoder.broadcast_forward(seq_repr, adj)
    print(out1)
    out2 = gcn_encoder.parallel_forward(seq_repr, adj)
    print(out2)

















