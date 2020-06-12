from typing import Iterator, Dict, Tuple
from trainer.DocREDBaseTrainer import BaseTrainer
import numpy as np
import random
from collections import defaultdict

from trainer.DocREDBaseTrainer import max_num_mentions_per_example, max_num_entities_per_example, \
    max_num_mentions_per_entity, IGNORE_INDEX
import h5py


class SimpleTrainer(BaseTrainer):

    def __init__(self, params) -> None:
        super().__init__(params)

    def __assemble_mentions_vertexes(self, batch_example_idx, vertex_set, vertex_indices, vertex_indices_mask,
                                     entity_span_indices, entity_span_indices_mask) -> Tuple[int, int, Dict]:
        span_pos_to_idx = {}
        num_mention = 0
        max_num_coref = 1
        for vertex_idx, entity in enumerate(vertex_set):
            max_num_coref = max(max_num_coref, len(entity))
            for _e_idx, span in enumerate(entity):
                vertex_indices[batch_example_idx, vertex_idx, _e_idx] = num_mention
                vertex_indices_mask[batch_example_idx, vertex_idx, _e_idx] = 1
                span_unique_pos = f"{span['pos'][0]}_{span['pos'][1]}"
                span_pos_to_idx[span_unique_pos] = num_mention
                entity_span_indices[batch_example_idx, num_mention, 0] = span['pos'][0]
                entity_span_indices[batch_example_idx, num_mention, 1] = span['pos'][1] - 1  # to include end index
                entity_span_indices_mask[batch_example_idx, num_mention] = 1
                num_mention += 1
        return max_num_coref, num_mention, span_pos_to_idx

    def get_train_batch(self) -> Iterator[Dict]:
        random.shuffle(self.train_order)
        # text_idx (0 ~ 194784)
        context_idxs = np.zeros((self.batch_size, self.max_length), dtype=np.int64)
        # bert_embeddings
        if self.use_bert_embedding:
            bert_embeddings = np.zeros((self.batch_size, self.max_length, self.bert_embedd_dim), dtype=np.float32)
        # entity position: entity mentions referring to the same entity
        # are assigned an identical idx (1 ~ num_max_entity)
        context_pos = np.zeros((self.batch_size, self.max_length), dtype=np.int16)
        # h_mapping = torch.Tensor(self.batch_size, self.h_t_limit, self.max_length).cuda()
        # t_mapping = torch.Tensor(self.batch_size, self.h_t_limit, self.max_length).cuda()
        # each entity pair may have more than one relation label
        # (`None`, or one real relation label, or several real relation labels)
        relation_multi_label = np.zeros((self.batch_size, self.train_h_t_limit, self.relation_num), dtype=np.float32)
        # 1 for entity pair having relation (None or real relation), 0 for padding
        relation_mask = np.zeros((self.batch_size, self.train_h_t_limit), dtype=np.float32)

        # token idx 1,2,...,text_len (1 ~ 512)
        # pos_idx = torch.LongTensor(self.batch_size, self.max_length).cuda()
        # NER type idx (0 ~ 6)
        context_ner = np.zeros((self.batch_size, self.max_length), dtype=np.int16)
        # char idx (0 ~ 263)
        # context_char_idxs = torch.LongTensor(self.batch_size, self.max_length, self.char_limit).cuda()
        # randomly assign one rel label to each entity pair (0 for `None` relation)
        relation_label = np.zeros((self.batch_size, self.train_h_t_limit), dtype=np.int16)
        # relative distance of h_entity and t_entity
        ht_pair_pos = np.zeros((self.batch_size, self.train_h_t_limit), dtype=np.int16)

        # entity_span_indices
        # last dim `0` for start, `1` for end (inclusive)
        entity_span_indices = np.zeros((self.batch_size, max_num_mentions_per_example, 2), dtype=np.int16)
        entity_span_indices_mask = np.zeros((self.batch_size, max_num_mentions_per_example), dtype=np.int16)

        # vertex_indices
        vertex_indices = np.zeros((self.batch_size, max_num_entities_per_example, max_num_mentions_per_entity), dtype=np.int16)
        vertex_indices_mask = np.zeros((self.batch_size, max_num_entities_per_example, max_num_mentions_per_entity), dtype=np.int16)

        # head_tail_indices [:, :, 0] for head, [:, :, 1] for tail
        head_tail_indices = np.zeros((self.batch_size, self.train_h_t_limit, 2), dtype=np.int16)

        num_pos_entity_pairs, num_neg_entity_pairs, num_entity_pairs = 0, 0, 0
        with h5py.File(self.train_bert_feature, 'r') as fin:
            for batch_idx in range(self.train_batches):
                start_id = batch_idx * self.batch_size
                cur_bsz = min(self.batch_size, self.train_len - start_id)
                cur_batch = list(self.train_order[start_id: start_id + cur_bsz])
                cur_batch.sort(key=lambda x: np.sum(self.data_train_word[x] > 0), reverse=True)

                # zero_() some tensors
                for _tensor in [entity_span_indices, entity_span_indices_mask,
                                vertex_indices, vertex_indices_mask, head_tail_indices,
                                relation_multi_label, relation_mask, ht_pair_pos]:  # head_tail_indices_mask]:
                    _tensor.fill(0)
                relation_label.fill(IGNORE_INDEX)

                max_h_t_cnt, batch_max_num_entities, batch_max_num_vertexes, batch_max_num_coref = 1, 1, 1, 1
                batch_lens, indexes = [], []
                for batch_example_idx, index in enumerate(cur_batch):
                    context_idxs[batch_example_idx] = self.data_train_word[index, :]
                    context_pos[batch_example_idx]= self.data_train_pos[index, :]
                    # context_char_idxs[batch_example_idx].copy_(torch.from_numpy(self.data_train_char[index, :]))
                    context_ner[batch_example_idx] = self.data_train_ner[index, :]

                    if self.use_bert_embedding:
                        seq_embedd = np.array(fin[str(index)])
                        batch_lens.append(seq_embedd.shape[0])
                        bert_embeddings[batch_example_idx] = self.padding(seq_embedd, self.max_length, self.bert_embedd_dim)

                    # for example_rel_idx in range(self.max_length):
                    #     if self.data_train_word[index, example_rel_idx] == 0:
                    #         break
                    #     pos_idx[batch_example_idx, example_rel_idx] = example_rel_idx + 1

                    ins = self.train_file[index]

                    num_coref, num_mention, span_pos_to_idx = \
                        self.__assemble_mentions_vertexes(batch_example_idx, ins['vertexSet'],
                                                          vertex_indices, vertex_indices_mask,
                                                          entity_span_indices, entity_span_indices_mask)
                    batch_max_num_vertexes = max(batch_max_num_vertexes, len(ins['vertexSet']))
                    batch_max_num_entities = max(batch_max_num_entities, num_mention)

                    labels = ins['labels']
                    idx2label = defaultdict(list)
                    for label in labels:
                        idx2label[(label['h'], label['t'])].append(label['r'])
                    train_tripe = list(idx2label.keys())
                    rel_cnt = 0
                    positive_is_enough = False
                    # assemble real relations
                    for ex_rel_idx, (h_idx, t_idx) in enumerate(train_tripe):
                        example_rel_idx = rel_cnt
                        hlist = ins['vertexSet'][h_idx]
                        tlist = ins['vertexSet'][t_idx]
                        head_tail_indices[batch_example_idx, example_rel_idx, 0] = h_idx
                        head_tail_indices[batch_example_idx, example_rel_idx, 1] = t_idx

                        ht_pair_pos[batch_example_idx, example_rel_idx] = self.get_head_tail_relative_pos(hlist[0], tlist[0])

                        label = idx2label[(h_idx, t_idx)]
                        for r in label:
                            relation_multi_label[batch_example_idx, example_rel_idx, r] = 1

                        relation_mask[batch_example_idx, example_rel_idx] = 1
                        rt = np.random.randint(len(label))
                        relation_label[batch_example_idx, example_rel_idx] = label[rt]
                        # this positive rel example ends
                        rel_cnt += 1
                        if rel_cnt == self.train_h_t_limit:
                            positive_is_enough = True
                            break

                    # assemble NA relations
                    train_limit = len(train_tripe)
                    num_pos_entity_pairs += rel_cnt
                    random.shuffle(ins['na_triple'])
                    num_neg = len(ins['na_triple'])
                    if self.use_neg_sample:
                        num_neg = self.neg_sample_multiplier * (train_limit if train_limit > 0 else 1)
                    train_limit += num_neg
                    train_limit = min(self.train_h_t_limit, train_limit)

                    neg_counter = 0
                    if not positive_is_enough:
                        for (h_idx, t_idx) in ins['na_triple']:
                            example_rel_idx = rel_cnt
                            hlist = ins['vertexSet'][h_idx]
                            tlist = ins['vertexSet'][t_idx]

                            if self.filter_by_entity_type:
                                if not self.entity_type_is_valid(hlist, tlist):
                                    continue

                            head_tail_indices[batch_example_idx, example_rel_idx, 0] = h_idx
                            head_tail_indices[batch_example_idx, example_rel_idx, 1] = t_idx
                            # head_tail_indices_mask[i, j] = 1

                            relation_multi_label[batch_example_idx, example_rel_idx, 0] = 1
                            relation_label[batch_example_idx, example_rel_idx] = 0
                            relation_mask[batch_example_idx, example_rel_idx] = 1
                            ht_pair_pos[batch_example_idx, example_rel_idx] = self.get_head_tail_relative_pos(hlist[0], tlist[0])
                            # this negative rel example ends
                            rel_cnt += 1
                            neg_counter += 1
                            if rel_cnt == train_limit:
                                break

                    # this example ends
                    indexes.append(index)
                    max_h_t_cnt = max(max_h_t_cnt, rel_cnt)
                    num_entity_pairs += rel_cnt
                    num_neg_entity_pairs += neg_counter
                    assert num_pos_entity_pairs + num_neg_entity_pairs == num_entity_pairs

                # this batch ends
                input_mask = (context_idxs[:cur_bsz] > 0).astype(np.bool)
                input_lengths = np.sum(input_mask, axis=1)
                max_c_len = int(np.max(input_lengths))
                if self.use_bert_embedding:
                    assert input_lengths.tolist() == batch_lens, 'len ERROR'

                input_mask = context_idxs[:cur_bsz, :max_c_len] > 0

                yield {
                    'context_idxs': self._to_tensor(context_idxs[:cur_bsz, :max_c_len], 'long'),
                    'bert_feature': self._to_tensor(bert_embeddings[:cur_bsz, :max_c_len, :]) if self.use_bert_embedding else None,
                    'context_pos': self._to_tensor(context_pos[:cur_bsz, :max_c_len], 'long'),
                    'context_ner': self._to_tensor(context_ner[:cur_bsz, :max_c_len], 'long'),
                    'input_mask': self._to_tensor(input_mask, 'long'),
                    'for_relation_repr': {
                        'entity_span_indices': self._to_tensor(entity_span_indices[:cur_bsz, :batch_max_num_entities, :], 'long'),
                        'entity_span_indices_mask': self._to_tensor(entity_span_indices_mask[:cur_bsz, :batch_max_num_entities], 'long'),
                        'vertex_indices': self._to_tensor(vertex_indices[:cur_bsz, :batch_max_num_vertexes, :batch_max_num_coref], 'long'),
                        'vertex_indices_mask': self._to_tensor(vertex_indices_mask[:cur_bsz, :batch_max_num_vertexes, :batch_max_num_coref], 'long'),
                        'head_tail_indices': self._to_tensor(head_tail_indices[:cur_bsz, :max_h_t_cnt, :], 'long'),
                    },
                    'relation_label': self._to_tensor(relation_label[:cur_bsz, :max_h_t_cnt], 'long'),
                    'ht_pair_pos': self._to_tensor(ht_pair_pos[:cur_bsz, :max_h_t_cnt], 'long'),
                    'relation_multi_label': self._to_tensor(relation_multi_label[:cur_bsz, :max_h_t_cnt]),
                    'relation_mask': self._to_tensor(relation_mask[:cur_bsz, :max_h_t_cnt]),
                    'pos_idx': None,
                    'context_char_idxs': None,
                    'indexes': indexes
                }

        self.num_train_entity_pairs = num_entity_pairs
        self.num_pos_entity_pairs = num_pos_entity_pairs
        self.num_neg_entity_pairs = num_neg_entity_pairs
        self.print_train_num_pos_neg()

    def get_test_batch(self) -> Iterator[Dict]:
        context_idxs = np.zeros((self.test_batch_size, self.max_length), dtype=np.int64)
        if self.use_bert_embedding:
            bert_embeddings = np.zeros((self.test_batch_size, self.max_length, self.bert_embedd_dim), dtype=np.float32)
        context_pos = np.zeros((self.test_batch_size, self.max_length), dtype=np.int16)
        relation_mask = np.zeros((self.test_batch_size, self.test_relation_limit), dtype=np.float32)
        context_ner = np.zeros((self.test_batch_size, self.max_length), dtype=np.int16)
        ht_pair_pos = np.zeros((self.test_batch_size, self.test_relation_limit), dtype=np.int16)

        entity_span_indices = np.zeros((self.test_batch_size, max_num_mentions_per_example, 2), dtype=np.int16)
        entity_span_indices_mask = np.zeros((self.test_batch_size, max_num_mentions_per_example), dtype=np.int16)

        # vertex_indices
        vertex_indices = np.zeros((self.test_batch_size, max_num_entities_per_example, max_num_mentions_per_entity), dtype=np.int16)
        vertex_indices_mask = np.zeros((self.test_batch_size, max_num_entities_per_example, max_num_mentions_per_entity), dtype=np.int16)

        # head_tail_indices [:, :, 0] for head, [:, :, 1] for tail
        head_tail_indices = np.zeros((self.test_batch_size, self.test_relation_limit, 2), dtype=np.int16)

        num_entity_pairs = 0
        with h5py.File(self.test_bert_feature, 'r') as fin:
            for b in range(self.test_batches):
                start_id = b * self.test_batch_size
                cur_bsz = min(self.test_batch_size, self.test_len - start_id)
                cur_batch = list(self.test_order[start_id: start_id + cur_bsz])

                for _tensor in [ht_pair_pos, relation_mask, entity_span_indices, entity_span_indices_mask,
                                vertex_indices, vertex_indices_mask, head_tail_indices]:
                    _tensor.fill(0)

                max_h_t_cnt, batch_max_num_entities, batch_max_num_vertexes, batch_max_num_coref = 1,1,1,1

                cur_batch.sort(key=lambda x: np.sum(self.data_test_word[x] > 0), reverse=True)

                labels, L_vertex, titles, indexes, batch_lens = [], [], [], [], []
                for batch_example_idx, index in enumerate(cur_batch):
                    context_idxs[batch_example_idx] = self.data_test_word[index, :]
                    context_pos[batch_example_idx] = self.data_test_pos[index, :]
                    # context_char_idxs[i].copy_(torch.from_numpy(self.data_test_char[index, :]))
                    context_ner[batch_example_idx] = self.data_test_ner[index, :]

                    if self.use_bert_embedding:
                        seq_embedd = np.array(fin[str(index)])
                        batch_lens.append(seq_embedd.shape[0])
                        bert_embeddings[batch_example_idx] = self.padding(seq_embedd, self.max_length, self.bert_embedd_dim)

                    idx2label = defaultdict(list)
                    ins = self.test_file[index]

                    for label in ins['labels']:
                        idx2label[(label['h'], label['t'])].append(label['r'])

                    num_coref, num_mention, span_pos_to_idx = \
                        self.__assemble_mentions_vertexes(batch_example_idx, ins['vertexSet'],
                                                          vertex_indices, vertex_indices_mask,
                                                          entity_span_indices, entity_span_indices_mask)
                    batch_max_num_vertexes = max(batch_max_num_vertexes, len(ins['vertexSet']))
                    batch_max_num_entities = max(batch_max_num_entities, num_mention)

                    L = len(ins['vertexSet'])
                    titles.append(ins['title'])

                    example_rel_idx = 0
                    for h_idx in range(L):
                        for t_idx in range(L):
                            if h_idx != t_idx:
                                hlist = ins['vertexSet'][h_idx]
                                tlist = ins['vertexSet'][t_idx]
                                if self.filter_by_entity_type:
                                    if not self.entity_type_is_valid(hlist, tlist):
                                        continue
                                head_tail_indices[batch_example_idx, example_rel_idx, 0] = h_idx
                                head_tail_indices[batch_example_idx, example_rel_idx, 1] = t_idx

                                relation_mask[batch_example_idx, example_rel_idx] = 1

                                ht_pair_pos[batch_example_idx, example_rel_idx] = self.get_head_tail_relative_pos(hlist[0], tlist[0])

                                example_rel_idx += 1
                                num_entity_pairs += 1

                    max_h_t_cnt = max(max_h_t_cnt, example_rel_idx)
                    label_set = {}
                    for label in ins['labels']:
                        label_set[(label['h'], label['t'], label['r'])] = label['in' + self.train_prefix]

                    labels.append(label_set)
                    L_vertex.append(L)
                    indexes.append(index)

                input_mask = (context_idxs[:cur_bsz] > 0).astype(np.bool)
                input_lengths = np.sum(input_mask, axis=1)
                max_c_len = int(np.max(input_lengths))
                if self.use_bert_embedding:
                    assert input_lengths.tolist() == batch_lens, 'len ERROR'

                input_mask = context_idxs[:cur_bsz, :max_c_len] > 0

                yield {
                    'context_idxs': self._to_tensor(context_idxs[:cur_bsz, :max_c_len], 'long'),
                    'bert_feature': self._to_tensor(bert_embeddings[:cur_bsz, :max_c_len, :]) if self.use_bert_embedding else None,
                    'context_pos': self._to_tensor(context_pos[:cur_bsz, :max_c_len], 'long'),
                    'context_ner': self._to_tensor(context_ner[:cur_bsz, :max_c_len], 'long'),
                    'input_mask': self._to_tensor(input_mask, 'long'),
                    'for_relation_repr': {
                        'entity_span_indices': self._to_tensor(entity_span_indices[:cur_bsz, :batch_max_num_entities, :], 'long'),
                        'entity_span_indices_mask': self._to_tensor(entity_span_indices_mask[:cur_bsz, :batch_max_num_entities], 'long'),
                        'vertex_indices': self._to_tensor(vertex_indices[:cur_bsz, :batch_max_num_vertexes, :batch_max_num_coref], 'long'),
                        'vertex_indices_mask': self._to_tensor(vertex_indices_mask[:cur_bsz, :batch_max_num_vertexes, :batch_max_num_coref], 'long'),
                        'head_tail_indices': self._to_tensor(head_tail_indices[:cur_bsz, :max_h_t_cnt, :], 'long'),
                    },
                    'relation_mask': self._to_tensor(relation_mask[:cur_bsz, :max_h_t_cnt]),
                    'ht_pair_pos': self._to_tensor(ht_pair_pos[:cur_bsz, :max_h_t_cnt], 'long'),
                    'context_char_idxs': None,
                    'titles': titles,
                    'indexes': indexes,
                    'labels': labels,
                    'L_vertex': L_vertex
                }

        self.num_test_entity_pairs = num_entity_pairs
        self.print_test_num_entity_pairs()

    def get_config_str(self) -> str:
        config_str = super(SimpleTrainer, self).get_config_str()
        return config_str + self.get_format_params_end_line()
