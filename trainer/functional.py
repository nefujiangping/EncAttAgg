# coding: utf-8
from typing import Dict, List, Iterator
import torch
import random
import numpy as np
import os
import yaml
import json
from misc.metrics import Accuracy
from misc.constant import dirs_to_backup


class Functional(object):

    def __init__(self, params: Dict = None):
        self.__set_seed(params['randomseed'])
        self.params = params
        self.use_gpu = True if params['device'] == 'gpu' else False
        self.device = torch.device("cuda" if torch.cuda.is_available() and self.use_gpu else "cpu")

        self.checkpoint_dir = params['checkpoint_dir']
        self.summary_dir = params['summary_dir']
        self.log_dir = params['log_dir']
        self.data_path = params['data_path']
        self.bert_embedding_dir = params['bert_embedding_dir']
        self.backup = not params['not_backup'] if 'not_backup' in params else None

        self.dataset = params['dataset']
        self.train_prefix = params['train_prefix']
        self.test_prefix = params['test_prefix']

        self.use_bert_embedding = params['use_bert_embedding']
        self.bert_embedd_dim = 768
        if self.use_bert_embedding:
            print('Use bert embedding...')
            self.bert_embedd_dim = params['bert_embedd_dim']
        self.embedd_dim = params['embedd_dim']
        self.freeze_embedd = params['freeze_embedd']
        self.drop_word = params['drop_word'] if 'drop_word' in params else None
        self.dis_size = params['dis_size'] if 'dis_size' in params else None
        self.coref_size = params['coref_size'] if 'coref_size' in params else None
        self.entity_type_size = params['entity_type_size'] if 'entity_type_size' in params else None
        self.nlayer = params['nlayer'] if 'nlayer' in params else None
        self.cat_nlayer = params['cat_nlayer'] if 'cat_nlayer' in params else None
        self.hidden_size = params['hidden_size'] if 'hidden_size' in params else None
        self.lstm_keep_prob = params['lstm_keep_prob'] if 'lstm_keep_prob' in params else None  # for lstm
        self.entity_span_pooling = params['entity_span_pooling'] if 'entity_span_pooling' in params else None
        self.coref_pooling = params['coref_pooling'] if 'coref_pooling' in params else None  # For Baseline BiLSTM-M, default `mean`
        self.which_model = params['which_model'] if 'which_model' in params else None

        self.mutual_attender = params['mutual_attender'] if 'mutual_attender' in params else None
        self.integration_attender = params['integration_attender'] if 'integration_attender' in params else None
        self.use_bilinear = params['use_bilinear'] if 'use_bilinear' in params else None
        self.use_distance = params['use_distance'] if 'use_distance' in params else None
        self.use_overlap = params['use_overlap'] if 'use_overlap' in params else None
        self.bi_affine_dropout = params['bi_affine_dropout'] if 'bi_affine_dropout' in params else None

        self.max_epoch = params['max_epoch'] if 'max_epoch' in params else None
        self.init_lr = params['init_lr'] if 'init_lr' in params else None
        self.batch_size = params['batch_size'] if 'batch_size' in params else None
        self.train_h_t_limit = params['train_rel_limit_per_example'] if 'train_rel_limit_per_example' in params else None
        self.test_batch_size = params['test_batch_size'] if params['test_batch_size'] else self.batch_size
        self.exp_id = params['exp_id']
        self.test_relation_limit = params['test_rel_limit_per_example'] if 'test_rel_limit_per_example' in params else None
        self.use_lr_scheduler = params['use_lr_scheduler'] if 'use_lr_scheduler' in params else None
        self.use_neg_sample = params['use_neg_sample'] if 'use_neg_sample' in params else None
        self.neg_sample_multiplier = params['neg_sample_multiplier'] if 'neg_sample_multiplier' in params else None

        self.debug_test = params['debug_test']
        self.write_weights = params['write_weights']
        self.epoch_start_to_eval = params['epoch_start_to_eval']

        self.pretrain_model = None
        self.model_loaded_from_epoch = None
        self.coref_only = False
        self.use_sample_weight = False
        self.accumulation_steps = 1
        # if self.use_neg_sample:
        #     self.train_h_t_limit = 70 * (self.neg_sample_multiplier + 1)  # 70: max num_rels of train (and dev)

        self.max_length = params['max_length']
        self.relation_num = params['relation_num']
        self.entity_type_num = params['entity_type_num']
        self.pos_num = 2 * self.max_length
        self.entity_num = self.max_length
        self.dis2idx = self._get_dis2idx()
        self.period = 50

        for _dir in [self.log_dir, self.checkpoint_dir, self.summary_dir]:
            if not os.path.exists(_dir):
                os.mkdir(_dir)

        self.logging(self.exp_id)
        self.id2word = json.load(open(os.path.join(self.data_path, 'word2id.json')))
        self.id2word = {idx: word for word, idx in self.id2word.items()}

        self.input_theta_of_best_epoch = -1
        self.acc_NA = Accuracy()
        self.acc_not_NA = Accuracy()
        self.acc_total = Accuracy()
        self.best_scores = {
            'main_metric': {
                'ign_f1': 0.0,
                'epoch': 0,
                'auc': 0.0
            },
            'auc': 0.0,
            'ign_coref_f1': 0.0,
            'ign_non_coref_f1': 0.0
        }

        self.num_train_entity_pairs = 0
        self.num_pos_entity_pairs = None
        self.num_neg_entity_pairs = None
        self.num_test_entity_pairs = None

    def print_train_num_pos_neg(self):
        self.logging(f"num_train_entity_pairs: {self.num_train_entity_pairs}. "
                     f"use_neg_sample: {self.use_neg_sample}; #P:#N=1:{self.neg_sample_multiplier}"
                     f" = {self.num_pos_entity_pairs}:{self.num_neg_entity_pairs}")

    def print_test_num_entity_pairs(self):
        self.logging(f"num_test_entity_pairs: {self.num_test_entity_pairs}. ")

    def __set_seed(self, randomseed):
        torch.manual_seed(randomseed)
        torch.cuda.manual_seed(randomseed)
        torch.cuda.manual_seed_all(randomseed)
        random.seed(randomseed)
        np.random.seed(randomseed)
        torch.backends.cudnn.deterministic = True

    def logging(self, s, print_=True, log_=True):
        if print_:
            print(s)
        if log_:
            with open(os.path.join(os.path.join("log", self.exp_id)), 'a+', encoding='utf8') as f_log:
                f_log.write(s + '\n')

    def set_on_cpu(self):
        self.use_gpu = False
        self.device = torch.device("cpu")

    def backup_codes(self):
        from shutil import copy as sh_copy, copytree as sh_cptree
        import os

        def ignore(src, names):
            return ['__pycache__']
        # backup dirs
        backup_dir = f"./{self.summary_dir}/{self.exp_id}"
        for _dir in dirs_to_backup:
            sh_cptree(_dir, f"./{backup_dir}/{_dir}", ignore=ignore)
        # backup ./*.py
        for _file in os.listdir('./'):
            if _file.endswith('.py'):
                sh_copy(f"./{_file}", f"./{backup_dir}/{_file}")
        # backup params
        _file = os.path.basename(self.params['param_file'])
        yaml.dump(self.params, open(f"{backup_dir}/{self.exp_id}___{_file}", 'w'))

    def visualize_data(self, prefix, data, n=1, max_len=30):
        def to_np(from_data, dtype='int32'):
            return from_data.cpu().numpy().astype(dtype).tolist()
        for i in range(n):
            try:
                # print(f"{prefix}_index: {data['indexes'][i]}")
                self.logging(f"{prefix}_index: {data['indexes'][i]}", log_=(i == 0))
                if 'context_idxs' in data:
                    context_idxs = to_np(data['context_idxs'][i])
                    tokens_str = ' '.join([self.id2word[idx] for idx in context_idxs[:max_len]])
                    self.logging(f"tokens: {tokens_str}", log_=(i == 0))
                if 'context_pos' in data:
                    context_pos = to_np(data['context_pos'][i][:max_len])
                    print(f"context_pos (coref): {context_pos}")
                if 'for_relation_repr' in data:
                    for_relation_repr = data['for_relation_repr']
                    if 'entity_span_indices' in for_relation_repr:
                        entity_span_indices = to_np(for_relation_repr['entity_span_indices'][i][:5])
                        print(f"entity_span_indices: {entity_span_indices}")
                if 'ht_pair_pos' in data:
                    print(f"ht_pair_pos size : {data['ht_pair_pos'].size()}")
                    print('one sample (part):')
                    to_print = to_np(data['ht_pair_pos'][i][:5])
                    if isinstance(to_print[0], list):
                        for item in to_print:
                            print(item)
                    else:
                        print(to_print)
            except IndexError as indexE:
                break

    def padding(self, X, max_len, dim):
        return np.concatenate([
            X, np.zeros((max_len-X.shape[0], dim))
        ])

    def _get_dis2idx(self):
        values = np.zeros((self.max_length), dtype='int64')
        values[1] = 1
        values[2:] = 2
        values[4:] = 3
        values[8:] = 4
        values[16:] = 5
        values[32:] = 6
        values[64:] = 7
        values[128:] = 8
        values[256:] = 9
        return values

    def get_head_tail_relative_pos(self, head, tail) -> int:
        delta_dis = head['pos'][0] - tail['pos'][0]
        if delta_dis < 0:
            relative_pos_idx = -int(self.dis2idx[-delta_dis])
        else:
            relative_pos_idx = int(self.dis2idx[delta_dis])
        return relative_pos_idx

    def set_max_epoch(self, max_epoch):
        self.max_epoch = max_epoch

    def _compute_acc(self, relation_label, output) -> None:
        for i in range(output.shape[0]):
            for j in range(output.shape[1]):
                label = relation_label[i][j]
                if label < 0:
                    break
                if label == 0:
                    self.acc_NA.add(output[i][j] == label)
                else:
                    self.acc_not_NA.add(output[i][j] == label)
                self.acc_total.add(output[i][j] == label)

    def _to_tensor(self, data, dtype='float'):
        assert dtype in ['float', 'long', 'bool']
        if dtype == 'float':
            return torch.Tensor(data).to(self.device)
        elif dtype == 'long':
            return torch.LongTensor(data).to(self.device)
        elif dtype == 'bool':
            return torch.BoolTensor(data).to(self.device)

    def format_params(self, params) -> List[str]:
        config_lines = list()
        for attr in params:
            try:
                value = self.__getattribute__(attr)
                if value is not None:
                    config_lines.append("{:<23}: {}\n".format(attr, value))
            except AttributeError:
                continue
        return config_lines

    def get_format_params_end_line(self):
        return f"{'='*48}\n\n"

    def get_train_batch(self) -> Iterator[Dict]:
        raise NotImplementedError

    def get_test_batch(self) -> Iterator[Dict]:
        raise NotImplementedError

    def check_get_batches(self):
        print("start get_train_batches ...")
        for data in self.get_train_batch():
            pass
        print("get_train_batches ends.")
        print("start get_test_batches ...")
        for data in self.get_test_batch():
            pass
        print("get_test_batches ends.")

    def get_config_str(self) -> str:
        params = ['dataset', 'train_prefix', 'test_prefix', 'max_length', 'use_bert_embedding',
                  'bert_embedd_dim', 'embedd_dim', 'freeze_embedd', 'drop_word', 'nlayer', 'cat_nlayer',
                  'hidden_size', 'which_model', 'mutual_attender', 'integration_attender', 'bi_affine_dropout',
                  'use_distance', 'use_overlap', 'use_bilinear', 'batch_size', 'init_lr', 'use_lr_scheduler',
                  'train_h_t_limit', 'test_batch_size', 'test_relation_limit',
                  'max_epoch', 'epoch_start_to_eval', 'use_neg_sample', 'neg_sample_multiplier']
        # Make sure that params have the same name with the property.
        # E.g., 'dataset' --> self.dataset
        # `format_params` will call self.__getattribute__(param)
        config_lines = list()
        config_lines.append(f"{'='*18} Parameters {'='*18}\n")
        config_lines.extend(self.format_params(params))
        return "".join(config_lines)


