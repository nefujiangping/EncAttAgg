from typing import Tuple, Dict, Any
from collections import defaultdict
from torch.utils.data import Dataset
import random
import numpy as np
import torch
import torch.nn.functional as F
from data_loader.data_utils import *


# max statistic in train, dev and test set of DocRED
max_num_mentions_per_example = 84
max_num_entities_per_example = 42
max_num_mentions_per_entity = 23

max_num_head_mentions_X_tail_mentions = 204
mentions_limit = 8
test_mentions_limit = max_num_mentions_per_entity

keys = [
    "head_mentions_indices", "head_mentions_indices_mask",
    "tail_mentions_indices", "tail_mentions_indices_mask",
    "ht_comb_indices", "ht_comb_mask", "ht_pair_pos"
]


class BertEncDataSet(Dataset):

    def __init__(self, data_dir: str, prefix: str, pad_token_id: int, is_train: bool, config) -> None:
        self.config = config
        self.data = json.load(open(f"{data_dir}/{prefix}.json"))
        self.data_word = np.load(f"{data_dir}/{prefix}_word.npy")
        self.data_pos = np.load(f"{data_dir}/{prefix}_pos.npy")
        self.data_ner = np.load(f"{data_dir}/{prefix}_ner.npy")
        self.input_ids = np.load(f"{data_dir}/{prefix}_input_ids.npy")
        self.token_pieces_map = np.load(f"{data_dir}/{prefix}_pieces_token_map.npy")
        self.token_pieces_map_mask = np.load(f"{data_dir}/{prefix}_pieces_token_map_mask.npy")
        self.offset = 1  # [CLS] is prepended to the sequence
        self.is_train = is_train
        self.pad_token_id = pad_token_id
        self.dis2idx = self.__get_dis2idx()
        self.marker2token = get_marker_to_token(model_name=config.model_name, meta_dir=config.data_dir)

    def __get_dis2idx(self) -> np.ndarray:
        values = np.zeros((self.config.max_length, ), dtype='int64')
        base = 2
        for i in range(1, 10):
            if i == 1:
                values[1] = 1
            else:
                values[base:] = i
                base *= 2
        return values

    def get_head_tail_relative_pos(self, head, tail, sent_map) -> int:
        if sent_map:
            delta_dis = sent_map[head['sent_id']][head["pos"][0]] - sent_map[tail['sent_id']][tail["pos"][0]]
        else:
            delta_dis = head['pos'][0] - tail['pos'][0]
        if delta_dis < 0:
            relative_pos_idx = -int(self.dis2idx[-delta_dis])
        else:
            relative_pos_idx = int(self.dis2idx[delta_dis])
        return relative_pos_idx

    def __assemble_mention(self, entities) -> Tuple[list, dict, int]:
        span_pos_to_idx = {}
        entity_span_indices = []
        num_entities = 0
        for e in entities:
            for m in e:
                s_pos, e_pos = m["pos"][0], m["pos"][1]
                span_unique_pos = f"{s_pos}_{e_pos}"
                span_pos_to_idx[span_unique_pos] = num_entities
                entity_span_indices.append([s_pos, e_pos-1])
                num_entities += 1
        return entity_span_indices, span_pos_to_idx, num_entities

    def __assemble_relation(self, span_pos_to_idx, hlist, tlist, limit, sent_map) -> Tuple[Dict[str, list], list]:
        num_H, num_T = 0, 0
        head_mentions_indices = []
        for hh in hlist[:limit]:
            head_mentions_indices.append(span_pos_to_idx[f"{hh['pos'][0]}_{hh['pos'][1]}"])
            num_H += 1
        head_mentions_indices_mask = [1] * num_H
        tail_mentions_indices = []
        for tt in tlist[:limit]:
            tail_mentions_indices.append(span_pos_to_idx[f"{tt['pos'][0]}_{tt['pos'][1]}"])
            num_T += 1
        tail_mentions_indices_mask = [1] * num_T
        # h1,h2; t1,t2,t3 :
        # ht_comb_indices[..., 0]: h1 h1 h1 h2 h2 h2
        # ht_comb_indices[..., 1]: t1 t2 t3 t1 t2 t3
        kk = 0
        ht_comb_indices = []
        ht_pair_pos = []
        for index_h, hh in enumerate(hlist[:limit]):
            for index_t, tt in enumerate(tlist[:limit]):
                ht_comb_indices.append([index_h, index_t])
                ht_pair_pos.append(self.get_head_tail_relative_pos(hh, tt, sent_map))
                kk += 1
        ht_comb_mask = [1] * kk
        values = {
            "head_mentions_indices": head_mentions_indices,
            "head_mentions_indices_mask": head_mentions_indices_mask,
            "tail_mentions_indices": tail_mentions_indices,
            "tail_mentions_indices_mask": tail_mentions_indices_mask,
            "ht_comb_indices": ht_comb_indices,
            "ht_comb_mask": ht_comb_mask,
            "ht_pair_pos": ht_pair_pos
        }
        nums = [num_H, num_T, kk]
        return values, nums

    def __padding2(self, f, num_rels, max_H, max_T, max_comb) -> Dict[str, Any]:
        hm_ind = np.zeros((num_rels, max_H), dtype=np.int16)
        hm_mask = np.zeros((num_rels, max_H), dtype=np.int16)
        tm_ind = np.zeros((num_rels, max_T), dtype=np.int16)
        tm_mask = np.zeros((num_rels, max_T), dtype=np.int16)

        ht_ind = np.zeros((num_rels, max_comb, 2), dtype=np.int16)
        ht_pair_pos = np.zeros((num_rels, max_comb), dtype=np.int16)
        ht_mask = np.zeros((num_rels, max_comb), dtype=np.int16)

        for i_r in range(num_rels):
            hm_ind[i_r, :len(f['head_mentions_indices'][i_r])] = f['head_mentions_indices'][i_r]
            hm_mask[i_r, :len(f['head_mentions_indices_mask'][i_r])] = f['head_mentions_indices_mask'][i_r]
            tm_ind[i_r, :len(f['tail_mentions_indices'][i_r])] = f['tail_mentions_indices'][i_r]
            tm_mask[i_r, :len(f['tail_mentions_indices_mask'][i_r])] = f['tail_mentions_indices_mask'][i_r]
            ht_pair_pos[i_r, :len(f['ht_pair_pos'][i_r])] = f['ht_pair_pos'][i_r]
            ht_mask[i_r, :len(f['ht_comb_mask'][i_r])] = f['ht_comb_mask'][i_r]
            ht_ind[i_r, :len(f['ht_comb_indices'][i_r]), :] = f['ht_comb_indices'][i_r]

        return {
            'head_mentions_indices': hm_ind,
            'head_mentions_indices_mask': hm_mask,
            'tail_mentions_indices': tm_ind,
            'tail_mentions_indices_mask': tm_mask,
            'ht_comb_indices': ht_ind,
            'ht_comb_mask': ht_mask,
            'ht_pair_pos': ht_pair_pos
        }

    def __func(self, entities, h_idx, t_idx, span_pos_to_idx, H, T, C, for_relation_repr,
               label: list, relations, relation_single_label, single_label,
               shuffle_mentions=True, sent_map=None):
        hlist = entities[h_idx]
        tlist = entities[t_idx]
        if shuffle_mentions:
            random.shuffle(hlist)
            random.shuffle(tlist)
        values, nums = self.__assemble_relation(span_pos_to_idx, hlist, tlist, mentions_limit, sent_map)
        H = max(H, nums[0])
        T = max(T, nums[1])
        C = max(C, nums[2])
        for i in range(len(keys)):
            for_relation_repr[keys[i]].append(values[keys[i]])
        relation_multi_label = [0] * self.config.relation_num
        for r in label:
            relation_multi_label[r] = 1
        relations.append(relation_multi_label)
        relation_single_label.append(single_label)
        return H, T, C

    def __get_train_item(self, index):
        ex = self.data[index]
        entities = ex['vertexSet']
        word_len = (self.data_word[index] > 0).astype(np.int32).sum()
        data_word_ids = self.data_word[index, :word_len]
        input_ids = self.input_ids[index]  # word-pieces ids
        input_ids = input_ids[input_ids != int(self.pad_token_id)]
        t2p_map = self.token_pieces_map[index, :word_len]  # [word_len, 6]
        t2p_map_mask = self.token_pieces_map_mask[index, :word_len]  # [word_len, 6]
        sent_map = None

        entity_span_indices, span_pos_to_idx, num_entities = self.__assemble_mention(entities)

        labels = ex['labels']
        idx2label = defaultdict(list)
        for label in labels:
            idx2label[(label['h'], label['t'])].append(label['r'])
        train_tripe = list(idx2label.keys())

        ex_max_num_H, ex_max_num_T = 1, 1
        ex_max_num_combination = 1

        for_relation_repr = {
            'head_mentions_indices': [],
            'head_mentions_indices_mask': [],
            'tail_mentions_indices': [],
            'tail_mentions_indices_mask': [],
            'ht_comb_indices': [],
            'ht_comb_mask': [],
            'ht_pair_pos': []
        }

        # pos pairs
        num_relations = 0
        positive_is_enough = False
        relations, relation_single_label = [], []
        for (h_idx, t_idx) in train_tripe:
            labels = idx2label[(h_idx, t_idx)]
            single_label = labels[np.random.randint(len(labels))]
            ex_max_num_H, ex_max_num_T, ex_max_num_combination = self.__func(
                entities, h_idx, t_idx, span_pos_to_idx,
                ex_max_num_H, ex_max_num_T, ex_max_num_combination,
                for_relation_repr, labels, relations, relation_single_label, single_label,
                shuffle_mentions=True, sent_map=None
            )
            # this positive rel example ends
            num_relations += 1
            if num_relations == self.config.train_h_t_limit:
                positive_is_enough = True
                break
        pos_counter = num_relations

        # NA pairs
        train_limit = pos_counter
        num_neg = len(ex['na_triple'])
        if self.config.use_neg_sample:
            num_neg = self.config.neg_sample_multiplier * (train_limit if train_limit > 0 else 1)
        train_limit += num_neg
        train_limit = min(self.config.train_h_t_limit, train_limit)

        random.shuffle(ex['na_triple'])
        neg_counter = 0
        if not positive_is_enough:
            for (h_idx, t_idx) in ex['na_triple']:
                labels, single_label = [0], 0
                ex_max_num_H, ex_max_num_T, ex_max_num_combination = self.__func(
                    entities, h_idx, t_idx, span_pos_to_idx,
                    ex_max_num_H, ex_max_num_T, ex_max_num_combination,
                    for_relation_repr, labels, relations, relation_single_label, single_label,
                    shuffle_mentions=True, sent_map=None
                )
                # this negative rel example ends
                num_relations += 1
                neg_counter += 1
                if num_relations == train_limit:
                    break

        assert num_relations == pos_counter + neg_counter, f"rel_instance: {num_relations} != pos:{pos_counter} + neg:{neg_counter}"

        for_relation_repr = self.__padding2(
            for_relation_repr, num_relations, ex_max_num_H, ex_max_num_T, ex_max_num_combination)
        for_relation_repr.update({'entity_span_indices': entity_span_indices})

        return {
            'context_idxs': data_word_ids,
            'context_pos': self.data_pos[index, :word_len],
            'context_ner': self.data_ner[index, :word_len],
            'input_ids': input_ids.tolist(),  # word-pieces ids
            't2p_map': t2p_map,
            't2p_map_mask': t2p_map_mask,
            'for_relation_repr': for_relation_repr,
            'relation_label': relation_single_label,
            'relation_multi_label': relations,
            'index': index,
            'train': 1,
            'pad_id': self.pad_token_id,
            'n': [word_len, num_entities, num_relations, ex_max_num_combination, ex_max_num_H, ex_max_num_T]
        }

    def __get_test_item(self, index):
        ex = self.data[index]
        entities = ex['vertexSet']
        word_len = (self.data_word[index] > 0).astype(np.int32).sum()
        data_word_ids = self.data_word[index, :word_len]
        input_ids = self.input_ids[index]  # word-pieces ids
        input_ids = input_ids[input_ids != int(self.pad_token_id)]
        t2p_map = self.token_pieces_map[index, :word_len]  # [word_len, 6]
        t2p_map_mask = self.token_pieces_map_mask[index, :word_len]  # [word_len, 6]

        entity_span_indices, span_pos_to_idx, num_entities = self.__assemble_mention(entities)

        ex_max_num_H, ex_max_num_T = 1, 1
        ex_max_num_combination = 1

        for_relation_repr = {
            'head_mentions_indices': [],
            'head_mentions_indices_mask': [],
            'tail_mentions_indices': [],
            'tail_mentions_indices_mask': [],
            'ht_comb_indices': [],
            'ht_comb_mask': [],
            'ht_pair_pos': []
        }

        L = len(ex['vertexSet'])
        et_pairs = [(h_idx, t_idx) for h_idx in range(L) for t_idx in range(L) if h_idx != t_idx]
        for (h_idx, t_idx) in et_pairs:
            hlist = entities[h_idx]
            tlist = entities[t_idx]
            values, nums = self.__assemble_relation(span_pos_to_idx, hlist, tlist, mentions_limit, sent_map=None)
            ex_max_num_H = max(ex_max_num_H, nums[0])
            ex_max_num_T = max(ex_max_num_T, nums[1])
            ex_max_num_combination = max(ex_max_num_combination, nums[2])
            for i in range(len(keys)):
                for_relation_repr[keys[i]].append(values[keys[i]])

        for_relation_repr = self.__padding2(
            for_relation_repr, len(et_pairs), ex_max_num_H, ex_max_num_T, ex_max_num_combination)
        for_relation_repr.update({'entity_span_indices': entity_span_indices})

        return {
            'context_idxs': data_word_ids,
            'context_pos': self.data_pos[index, :word_len],
            'context_ner': self.data_ner[index, :word_len],
            'input_ids': input_ids.tolist(),  # word-pieces ids
            't2p_map': t2p_map,
            't2p_map_mask': t2p_map_mask,
            'for_relation_repr': for_relation_repr,
            'index': index,
            'train': 0,  # flag whether this is train data
            'pad_id': self.pad_token_id,  # padding token for collate_fn_bert_enc
            'title': ex['title'],
            'L_vertex': len(entities),
            'labels': {(l['h'], l['t'], l['r']): l['in' + self.config.train_prefix] for l in ex['labels']},
            'n': [word_len, num_entities, len(et_pairs), ex_max_num_combination, ex_max_num_H, ex_max_num_T]
        }

    def __getitem__(self, index) -> dict:
        if self.is_train:
            return self.__get_train_item(index)
        else:
            return self.__get_test_item(index)

    def __len__(self) -> int:
        return len(self.data)


def to_tensor(data, dtype='f') -> torch.Tensor:
    assert dtype in ['f', 'l', 'b']
    if dtype == 'f':
        return torch.tensor(data, dtype=torch.float32)
    elif dtype == 'l':
        return torch.tensor(data, dtype=torch.long)
    elif dtype == 'b':
        return torch.tensor(data, dtype=torch.bool)
    else:
        raise NotImplementedError("support dtypes: [float32, long, bool]")


def collate_fn_bert_enc(data) -> dict:
    is_train = True if data[0]['train'] else False
    pad_id = data[0]['pad_id']
    c_len = max([d['n'][0] for d in data])
    num_ets = max([d['n'][1] for d in data])
    num_rels = max([d['n'][2] for d in data])
    num_comb = max([d['n'][3] for d in data])
    num_H = max([d['n'][4] for d in data])
    num_T = max([d['n'][5] for d in data])
    indexes = [d['index'] for d in data]
    max_len = max([len(f["input_ids"]) for f in data])
    input_ids = [d["input_ids"] + [pad_id] * (max_len - len(d["input_ids"])) for d in data]
    attention_mask = [[1.0] * len(d["input_ids"]) + [0.0] * (max_len - len(d["input_ids"])) for d in data]
    word_mask, context_idxs = [], []
    t2p_map, t2p_map_mask = [], []
    context_pos, context_ner = [], []
    relation_label, relation_multi_label, relation_mask = [], [], []
    m_ind, hm_ind, hm_mask, tm_ind, tm_mask, ht_ind, ht_mask, ht_pair_pos = [], [], [], [], [], [], [], []
    titles, L_vertex, labels, for_attn_weights = [], [], [], []
    # padding
    for d in data:
        p_c_len = c_len - d['n'][0]
        p_num_rels = num_rels-d['n'][2]
        context_idxs.append(F.pad(to_tensor(d['context_idxs'], 'l'), [0, p_c_len]))
        word_mask.append(to_tensor([1]*d['n'][0]+[0]*p_c_len, 'l'))
        context_pos.append(F.pad(to_tensor(d['context_pos'], 'l'), [0, p_c_len]))
        context_ner.append(F.pad(to_tensor(d['context_ner'], 'l'), [0, p_c_len]))
        # list of elem: [p_c_len, pieces_per_token] (after padding)
        t2p_map.append(F.pad(to_tensor(d['t2p_map'], 'l'), [0, 0, 0, p_c_len]))
        t2p_map_mask.append(F.pad(to_tensor(d['t2p_map_mask'], 'l'), [0, 0, 0, p_c_len]))
        f = d['for_relation_repr']
        m_ind.append(F.pad(to_tensor(f['entity_span_indices'], 'l'), [0, 0, 0, num_ets - d['n'][1]]))
        pad = [0, num_H - d['n'][4], 0, p_num_rels]
        hm_ind.append(F.pad(to_tensor(f['head_mentions_indices'], 'l'), pad=pad))
        hm_mask.append(F.pad(to_tensor(f['head_mentions_indices_mask'], 'l'), pad=pad))
        pad = [0, num_T - d['n'][5], 0, p_num_rels]
        tm_ind.append(F.pad(to_tensor(f['tail_mentions_indices'], 'l'), pad=pad))
        tm_mask.append(F.pad(to_tensor(f['tail_mentions_indices_mask'], 'l'), pad=pad))
        pad = [0, 0, 0, num_comb - d['n'][3], 0, p_num_rels]
        ht_ind.append(F.pad(to_tensor(f['ht_comb_indices'], 'l'), pad=pad))
        ht_mask.append(F.pad(to_tensor(f['ht_comb_mask'], 'l'), pad=pad[2:]))
        ht_pair_pos.append(F.pad(to_tensor(f['ht_pair_pos'], 'l'), [0, num_comb-d['n'][3], 0, p_num_rels]))
        if is_train:
            relation_label.append(F.pad(to_tensor(d['relation_label'], 'l'), [0, p_num_rels]))
            relation_multi_label.append(F.pad(to_tensor(d['relation_multi_label'], 'f'), [0, 0, 0, p_num_rels]))
            relation_mask.append(to_tensor([1]*d['n'][2] + [0]*p_num_rels, 'f'))
        else:
            titles.append(d['title'])
            L_vertex.append(d['L_vertex'])
            labels.append(d['labels'])

    ht_pair_pos = torch.stack(ht_pair_pos, dim=0)

    if is_train:
        relation_label = torch.stack(relation_label, dim=0)
        relation_multi_label = torch.stack(relation_multi_label, dim=0)
        relation_mask = torch.stack(relation_mask, dim=0)

    # add dummy mask for padding rel to prevent from `/ 0` or `nan`
    head_mentions_indices_mask = torch.stack(hm_mask, dim=0)
    tail_mentions_indices_mask = torch.stack(tm_mask, dim=0)
    ht_comb_mask = torch.stack(ht_mask, dim=0)
    head_mentions_indices_mask[:, :, 0] = 1
    tail_mentions_indices_mask[:, :, 0] = 1
    ht_comb_mask[:, :, 0] = 1

    # [B, p_c_len, pieces_per_token]
    t2p_map_mask = torch.stack(t2p_map_mask, dim=0)
    t2p_map_mask[:, :, 0] = 1

    return {
        'indexes': indexes,
        'input_ids': torch.tensor(input_ids, dtype=torch.long),
        'attention_mask': torch.tensor(attention_mask, dtype=torch.float),
        'context_idxs': torch.stack(context_idxs, dim=0),
        't2p_map': torch.stack(t2p_map, dim=0),
        't2p_map_mask': t2p_map_mask,
        'entity_cluster_ids': torch.stack(context_pos, dim=0),
        'entity_type_ids': torch.stack(context_ner, dim=0),
        'word_mask': torch.stack(word_mask, dim=0),
        'entity_span_indices': torch.stack(m_ind, dim=0),
        'for_relation_repr': {
            'head_mentions_indices': torch.stack(hm_ind, dim=0),
            'head_mentions_indices_mask': head_mentions_indices_mask,
            'tail_mentions_indices': torch.stack(tm_ind, dim=0),
            'tail_mentions_indices_mask': tail_mentions_indices_mask,
            'ht_comb_indices': torch.stack(ht_ind, dim=0),
            'ht_comb_mask': ht_comb_mask
        },
        'dis_h_2_t': ht_pair_pos + 10,
        'dis_t_2_h': -ht_pair_pos + 10,
        'relation_label': relation_label,
        'relation_multi_label': relation_multi_label,
        'relation_mask': relation_mask,
        'context_char_idxs': None,
        'pos_idx': None,
        # For test
        'titles': titles,
        'L_vertex': L_vertex,
        'labels': labels,
        'for_attn_weights': for_attn_weights
    }
