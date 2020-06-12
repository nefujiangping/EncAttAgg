# coding: utf-8
from typing import List, Dict, Tuple
import copy

import h5py
import torch
import torch.nn as nn
import torch.optim as optim
import os
import time
import json
import random
import scipy.sparse as sp
from torch.optim.lr_scheduler import ReduceLROnPlateau, ExponentialLR
from torch.nn.utils import clip_grad_value_

from trainer.functional import Functional
from misc.metrics import p_r_f
from misc.util import TensorboardWriter
from misc.metrics import coref_vs_non_coref_performance, diff_dist_performance
from typing import Iterator, Dict
import numpy as np

# max statistic in train, dev and test set
max_num_mentions_per_example = 69
max_num_entities_per_example = 23
max_num_mentions_per_entity = 29

max_num_head_mentions_X_tail_mentions = 288
mentions_limit = 10  # 10
test_mentions_limit = max_num_mentions_per_entity

IGNORE_INDEX = -100


class CDRGCNNTrainer_Softmax(Functional):

    def __init__(self, params) -> None:
        super(CDRGCNNTrainer_Softmax, self).__init__(params)
        self.lowercase = params['lowercase']
        self.train_hypernym_filter = params['train_hypernym_filter']
        self.test_hypernym_filter = params['test_hypernym_filter']
        self.train_on_trainanddev = params['train_on_trainanddev']
        self.avg_params = params['avg_params']
        self.averaged_params = {}
        self.bin_size = self.test_bin_size = 10
        self.grad_clip_value = params['grad_clip_value']
        self.lr_decay = params['lr_decay']

        self.num_position_embeddings = self.max_length*2+1
        self.pad_position_idx = self.num_position_embeddings-1
        self.position_dim = params['position_dim']  # in_dim = 2*pos_dim + embedd_dim
        self.inp_dropout = params['inp_dropout']
        self.parallel_forward = params['parallel_forward']
        self.gcn_dim = params['gcn_dim']
        self.num_edge_types = params['num_edge_types']
        self.num_unRare_edge_types = params['num_unRare_edge_types']
        self.num_blocks = params['num_blocks']
        self.gcn_use_gate = params['gcn_use_gate']
        self.gcn_dropout = params['gcn_dropout']
        self.gcn_residual = params['gcn_residual']
        self.bi_affine_ff_dim = params['bi_affine_ff_dim']
        self.bi_affine_dropout = params['bi_affine_dropout']

    # This function is copied from EoG's `Trainer`
    # https://github.com/fenchri/edge-oriented-graph/blob/ad066448ee7027e766498a653023176343b0137e/src/nnet/trainer.py#L260
    def parameter_averaging(self, model, epoch=None, reset=False, num_recent=10):
        for p_name, p_value in model.named_parameters():
            if p_name not in self.averaged_params:
                self.averaged_params[p_name] = []

            if reset:
                p_new = copy.deepcopy(self.averaged_params[p_name][-1])  # use last epoch param

            elif epoch:
                p_new = np.mean(self.averaged_params[p_name][:epoch], axis=0)  # estimate average until this epoch

            else:
                self.averaged_params[p_name].append(p_value.data.to('cpu').numpy())
                # estimate average over recent `num_recent` epoch
                p_new = np.mean(self.averaged_params[p_name][-num_recent:], axis=0)

            p_value.data = torch.from_numpy(p_new).to(self.device)

    def _do_test(self, ori_model, model, epoch, lr_scheduler, summary_writer, optimizer, train_states):
        self.logging('-' * 90)
        eval_start_time = time.time()
        f1s, precision, recall, f1 = self.test(model)
        tlt = time.localtime()
        time_str = f"{tlt[0]}/{tlt[1]}/{tlt[2]} {tlt[3]}:{tlt[4]}"
        self.logging('| epoch {:3d} | eval_time: {:5.2f}s'.format(epoch, time.time() - eval_start_time))
        self.logging("{:<17} | epoch {:<2d} | "
                     "Ign F1: {:3.2f}, Precision: {:3.2f}; Recall: {:3.2f}"
                     .format(time_str, epoch, f1*100, precision*100, recall*100))

        # logging for tensorboard
        summary_writer.log_val_f1(f1, epoch)
        summary_writer.log_val_precision(precision, epoch)
        summary_writer.log_val_recall(recall, epoch)
        summary_writer.log_val_noncoref_f1(f1s['ign_non-coref'], epoch)
        summary_writer.log_val_coref_f1(f1s['ign_coref'], epoch)
        summary_writer.log_val_short_f1(f1s['ign_dist:1-25'], epoch)
        summary_writer.log_val_long_f1(f1s['ign_dist:26+'], epoch)

        if lr_scheduler:
            lr_scheduler.step(f1)

        # states to save
        to_save = {
            'model': ori_model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'train_states': train_states,  # global step, total_loss
            'best_scores': self.best_scores
        }
        if lr_scheduler:
            to_save.update({'lr_scheduler': lr_scheduler.state_dict()})

        # save `ign_f1` (main_metric) checkpoint
        if f1 > self.best_scores['main_metric']['ign_f1']:
            self.best_scores['main_metric']['ign_f1'] = f1
            self.best_scores['main_metric']['epoch'] = epoch
            # if not self.train_on_trainanddev:
            #     json.dump({"epoch": epoch, "prf": f"P: {precision}, R: {recall}, F1: {f1}."
            #               f" train on pure train, test on dev"},
            #               open(self.best_epoch_theta_dir + f"/{self.model_name}.json", 'w'))

            log_str = f'=== {time_str}  save model at epoch {epoch} '
            summary_writer.log_text("{:<17} | epoch {:<2d} | "
                                    "Ign F1: {:3.2f}, Precision: {:3.2f}; Recall: {:3.2f}"
                                    .format(time_str, epoch, f1*100, precision*100, recall*100), epoch)
            self.logging('='*50)
            self.logging(log_str + '='*(50-len(log_str)))
            self.logging('='*50)
            path = os.path.join(self.checkpoint_dir, self.exp_id)
            torch.save(to_save, path)

        self.logging('-' * 90)

    def _do_train_log(self, total_loss, epoch, start_time, global_step):
        cur_loss = total_loss / self.period
        elapsed = time.time() - start_time
        predict_mins_per_epoch = self.num_train_entity_pairs / self.period / self.batch_size * elapsed / 60.
        self.logging('| epoch {:2d} | step {:4d} |  min/{:3d}b {:2.1f} | train loss {:5.5f} '
                     '| not NA acc: {:4.2f} | NA acc: {:4.2f} | min/epoch {:2.1f}'
                     .format(epoch, global_step, self.period, elapsed/60., cur_loss,
                             self.acc_not_NA.get(), self.acc_NA.get(), predict_mins_per_epoch))

    def _compute_acc(self, relation_label, output) -> None:
        for i in range(output.shape[0]):
            label = relation_label[i]
            if label < 0:
                break
            if label == 0:
                self.acc_NA.add(output[i] == label)
            else:
                self.acc_not_NA.add(output[i] == label)
            self.acc_total.add(output[i] == label)

    def train(self, model_pattern):

        summary_writer = TensorboardWriter(self.exp_id)
        tlt = time.localtime()
        time_str = f"{tlt[0]}/{tlt[1]}/{tlt[2]} {tlt[3]}:{tlt[4]}"
        summary_writer.log_text(f"{time_str}\n{self.params['cmd']}", 0)

        ori_model = model_pattern(config=self)
        state_dict = None
        start_epoch = -1
        train_states = {'global_step': 0, 'total_loss': 0.0}
        if self.pretrain_model:
            state_dict = torch.load(self.pretrain_model)
            ori_model.load_state_dict(state_dict['model'])
            self.best_scores.update(state_dict['best_scores'])
            start_epoch = self.best_scores['main_metric']['epoch']
            train_states.update(state_dict['train_states'])
            assert start_epoch == self.model_loaded_from_epoch, 'Given `model_loaded_from_epoch` is not equal to `epoch in state_dict`'
            # Note that If continuing to train model from checkpoint using the same `randomseed`,
            # the model will use the same sampled examples (pos/neg samples of each document) as from the epoch 0
            self.logging('load model from %s, epoch %d' % (self.pretrain_model, self.model_loaded_from_epoch))

        if self.backup:
            self.backup_codes()

        if self.use_gpu:
            ori_model.cuda()
            model = nn.DataParallel(ori_model)
        else:
            model = ori_model
        parameters = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = optim.Adam(parameters, lr=self.init_lr)
        if state_dict:
            optimizer.load_state_dict(state_dict['optimizer'])

        cross_entropy_loss = nn.CrossEntropyLoss(reduction='none', ignore_index=IGNORE_INDEX)

        prev_lr = self.init_lr
        lr_scheduler = None
        if self.use_lr_scheduler:
            lr_scheduler = ReduceLROnPlateau(optimizer,
                                             mode='max', patience=3,
                                             factor=0.5, min_lr=5e-6, verbose=True)
        if self.lr_decay < 1.:
            lr_decay = ExponentialLR(optimizer, gamma=self.params['lr_decay'])

        if state_dict and 'lr_scheduler' in state_dict:
            if self.use_lr_scheduler:
                lr_scheduler.load_state_dict(state_dict['lr_scheduler'])
            else:
                print('pretrain model has `lr_scheduler` states, but now `use_lr_scheduler` is OFF.')

        start_time = time.time()
        tlt = time.localtime()
        log_str = f'=== {tlt[0]}/{tlt[1]}/{tlt[2]} {tlt[3]}:{tlt[4]}  start to train...'
        self.logging(log_str)
        model.train()
        optimizer.zero_grad()
        visualize_flag = True
        try:
            for epoch in range(start_epoch+1, self.max_epoch):
                summary_writer.log_train_lr(optimizer.param_groups[0]['lr'], epoch)
                epoch_start_time = time.time()
                self.acc_NA.clear()
                self.acc_not_NA.clear()
                self.acc_total.clear()
                forward_time = 0
                for data in self.get_train_batch():
                    if self.debug_test:
                        break
                    if visualize_flag:
                        self.visualize_data(self.train_prefix, data)
                    context_idxs = data['context_idxs']
                    dist_e1 = data['dist_e1']
                    dist_e2 = data['dist_e2']
                    input_lengths = data['input_lengths']
                    adj_matrices = data['adj_matrices']
                    for_relation_repr = data['for_relation_repr']
                    bert_feature = data['bert_feature']

                    relation_label = data['relation_label']
                    relation_mask = data['relation_mask']
                    before_forward = time.time()
                    predict_re = model(context_idxs, input_lengths, dist_e1, dist_e2,
                                       adj_matrices, for_relation_repr, bert_feature)
                    forward_time += (time.time()-before_forward)
                    loss = torch.sum(cross_entropy_loss(predict_re, relation_label) * relation_mask) / torch.sum(relation_mask)
                    # (N, R)
                    output = torch.argmax(predict_re, dim=-1)
                    output = output.data.cpu().numpy()

                    if self.accumulation_steps > 1:
                        if visualize_flag:
                            print('Use accumulation gradients...')
                        loss = loss / self.accumulation_steps
                        loss.backward()
                        if ((train_states['global_step']+1) % self.accumulation_steps) == 0:
                            optimizer.step()
                            optimizer.zero_grad()
                        train_states['total_loss'] += loss.item() * self.accumulation_steps
                    else:
                        loss.backward()
                        clip_grad_value_(parameters, self.params['grad_clip_value'])
                        optimizer.step()
                        optimizer.zero_grad()
                        train_states['total_loss'] += loss.item()

                    relation_label = relation_label.data.cpu().numpy()
                    self._compute_acc(relation_label, output)

                    train_states['global_step'] += 1

                    if train_states['global_step'] % self.period == 0:
                        summary_writer.log_train_loss(train_states['total_loss'] / self.period, train_states['global_step'])
                        summary_writer.log_train_not_na_acc(self.acc_not_NA.get(), train_states['global_step'])
                        self._do_train_log(train_states['total_loss'], epoch, start_time, train_states['global_step'])
                        train_states['total_loss'] = 0
                        start_time = time.time()
                    # this batch ends
                    visualize_flag = False

                # this epoch ends
                self.logging(f"Time usage of this epoch: {(time.time()-epoch_start_time)/60} mins")
                self.logging(f"Time usage of this epoch (forward): {forward_time / 60} mins.")
                if self.lr_decay < 1.:
                    lr_decay.step()
                if self.avg_params['use_avg_params']:
                    self.parameter_averaging(ori_model, num_recent=self.avg_params['num_recent_epochs'])
                if (epoch + 1) > self.epoch_start_to_eval:
                    model.eval()
                    self._do_test(ori_model, model, epoch, lr_scheduler, summary_writer, optimizer, train_states)
                    new_lr = optimizer.param_groups[0]['lr']
                    if new_lr < prev_lr:
                        self.logging('epoch {:3d}: reducing learning rate to {}.'.format(epoch, new_lr))
                    prev_lr = optimizer.param_groups[0]['lr']
                    model.train()
                if self.avg_params['use_avg_params']:
                    self.parameter_averaging(ori_model, reset=True)

        except KeyboardInterrupt:
            summary_writer.close()
            self.logging("Best epoch = %d | f1 %.4f | auc = %f" % (self.best_scores['main_metric']['epoch'],
                                                                   self.best_scores['main_metric']['ign_f1'],
                                                                   self.best_scores['main_metric']['auc']))
        print("Finish training")
        summary_writer.close()
        self.logging("Best epoch = %d | f1 %.4f | auc = %f" % (self.best_scores['main_metric']['epoch'],
                                                               self.best_scores['main_metric']['ign_f1'],
                                                               self.best_scores['main_metric']['auc']))

    def test(self, model):
        eval_start_time = time.time()
        test_result = []
        data_idx = 0
        for data in self.get_test_batch():
            with torch.no_grad():
                context_idxs = data['context_idxs']
                dist_e1 = data['dist_e1']
                dist_e2 = data['dist_e2']
                input_lengths = data['input_lengths']
                adj_matrices = data['adj_matrices']
                for_relation_repr = data['for_relation_repr']
                bert_feature = data['bert_feature']
                # relation_mask = data['relation_mask']
                # [(doc_index, ins['title'], h_idx, t_idx, 0)]
                entity_pairs = data['entity_pairs']
                test_batch_size = context_idxs.size(0)

                predict_re = model(context_idxs, input_lengths, dist_e1, dist_e2,
                                   adj_matrices, for_relation_repr, bert_feature)
                predict_re = torch.nn.functional.softmax(predict_re, dim=-1)
            # (N, 2) with value probability
            predict_re = predict_re.data.cpu().numpy()

            for ex_idx in range(test_batch_size):
                doc_index, KB_ID, h_idx, t_idx, _ = entity_pairs[ex_idx]
                r = np.argmax(predict_re[ex_idx])
                if r == 1:
                    test_result.append({
                        "index": doc_index,
                        "KB_ID": KB_ID,
                        "h_idx": h_idx,
                        "t_idx": t_idx,
                        "r_idx": 1,  # to be compatible with DocRED
                        "intrain": False  # to be compatible with DocRED
                    })

            data_idx += 1
            if data_idx % self.period == 0:
                print('| step {:3d} | time: {:5.2f}'.format(data_idx // self.period, (time.time() - eval_start_time)))
                eval_start_time = time.time()

        out_file_name = f"{self.exp_id}_{self.test_prefix}.json"
        json.dump(test_result, open(out_file_name, "w"), ensure_ascii=False, indent=2)

        # These `f1s` is not the final performances, because of the mergence of our pre-processing
        # Here we show these metrics for better observations of the training process
        f1s = coref_vs_non_coref_performance(self.test_file, test_result)
        diff_dist_f1s = diff_dist_performance(self.test_file, test_result)
        f1s.update(diff_dist_f1s)

        if self.is_test:
            precision, recall, f1 = p_r_f(out_file_name, 'test')
        else:
            precision, recall, f1 = p_r_f(out_file_name, 'dev')

        # print(f"P: {precision}, R: {recall}, F1: {f1}")
        return f1s, precision, recall, f1

    def testall(self, model_pattern):
        model = model_pattern(config=self)
        checkpoint_path = os.path.join(self.checkpoint_dir, self.exp_id)
        checkpoint = torch.load(checkpoint_path)
        print(f"load checkpoint from {checkpoint_path}")
        if 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
            # this is only for `ign_f1` checkpoint, because the epoch is always corresponding to the best `ign_f1`
            # if 'best_scores' in checkpoint:
            #     print(f"load model from epoch {checkpoint['best_scores']['main_metric']['epoch']}")
        else:
            model.load_state_dict(checkpoint)
        model.cuda()
        model.eval()
        f1s, precision, recall, f1 = self.test(model)
        if not self.is_test and f1s:  # dev
            for key, value in f1s.items():
                self.logging('{:<15} | {:.2f} '.format(key, value*100))

    def load_train_data(self) -> None:
        print("Reading training data...")
        prefix = self.train_prefix

        print('train', prefix)
        if not self.lowercase:
            self.data_train_word = np.load(os.path.join(self.data_path, prefix + '_word_nolower.npy'))
        else:
            self.data_train_word = np.load(os.path.join(self.data_path, prefix + '_word.npy'))
        self.data_train_pos = np.load(os.path.join(self.data_path, prefix + '_pos.npy'))
        self.data_train_ner = np.load(os.path.join(self.data_path, prefix + '_ner.npy'))
        # self.data_train_char = np.load(os.path.join(self.data_path, prefix + '_char.npy'))
        self.train_file = json.load(open(os.path.join(self.data_path, prefix + '.json')))
        self.train_sparse_adj_matrices = json.load(open(os.path.join(self.data_path, prefix + '_adj_matrix.json')))
        # if self.use_bert:
        self.train_bert_feature = f'{self.bert_embedding_dir}/{self.bert_embedd_dim}_dev_train.h5'

        print("Finish reading")
        assert self.data_train_word.shape[0] == 1000
        assert prefix == 'dev_train'

        if self.train_on_trainanddev:
            self.train_len = ins_num = self.data_train_word.shape[0]

            self.train_order = list(range(ins_num))
            self.train_bins = ins_num // self.bin_size
            if ins_num % self.bin_size != 0:
                self.train_bins += 1
            self.num_train_entity_pairs = 10000
        else:
            self.train_len = 500
            self.train_order = list(range(self.train_len))
            self.train_bins = self.train_len // self.bin_size
            if self.train_len % self.bin_size != 0:
                self.train_bins += 1
            self.num_train_entity_pairs = 5000

        print(f'train index: [{min(self.train_order)}, {max(self.train_order)}]')
        self.train_doc_lens = {}
        for doc in self.train_file:
            self.train_doc_lens[doc['title']] = sum([len(sent) for sent in doc['sents']])

    def load_test_data(self) -> None:
        prefix = self.test_prefix
        if self.train_on_trainanddev:
            assert prefix == 'dev_test'
        else:
            assert prefix == 'dev_train'
        print("Reading testing data...")
        self.data_word_vec = np.load(os.path.join(self.data_path, 'vec.npy'))
        # self.data_char_vec = np.load(os.path.join(self.data_path, 'char_vec.npy'))
        self.rel2id = json.load(open(os.path.join(self.data_path, 'rel2id.json')))
        self.negs_to_remove = json.load(open(os.path.join(self.data_path, 'filter_out_negs.json')))
        self.edgeType2idx = json.load(open(os.path.join(self.data_path, 'edgeType2idx.json')))
        self.idx2edgeType = {v: k for k, v in self.edgeType2idx.items()}
        self.test_sparse_adj_matrices = json.load(open(os.path.join(self.data_path, prefix + '_adj_matrix.json')))
        self.id2rel = {v: k for k, v in self.rel2id.items()}

        print(prefix)
        self.is_test = ('dev_test' == prefix)
        if not self.lowercase:
            self.data_test_word = np.load(os.path.join(self.data_path, prefix + '_word_nolower.npy'))
        else:
            self.data_test_word = np.load(os.path.join(self.data_path, prefix + '_word.npy'))
        self.data_test_pos = np.load(os.path.join(self.data_path, prefix + '_pos.npy'))
        self.data_test_ner = np.load(os.path.join(self.data_path, prefix + '_ner.npy'))
        # self.data_test_char = np.load(os.path.join(self.data_path, prefix + '_char.npy'))
        self.test_file = json.load(open(os.path.join(self.data_path, prefix + '.json')))
        # if self.use_bert:
        self.test_bert_feature = f'{self.bert_embedding_dir}/{self.bert_embedd_dim}_{prefix}.h5'
        self.test_len = 500
        print("Finish reading")

        if prefix == 'dev_train':  # dev is in train_file
            self.test_doc_indices = list(range(1000))[500:]
        elif prefix == 'dev_test':
            self.test_doc_indices = list(range(self.test_len))
        print(f'test index: [{min(self.test_doc_indices)}, {max(self.test_doc_indices)}]')

        self.test_doc_lens = {}
        for doc in self.test_file:
            self.test_doc_lens[doc['title']] = sum([len(sent) for sent in doc['sents']])

    def assemble_mentions(self, entity_span_indices, example_idx, vertexSet):
        span_pos_to_idx = {}
        num_entities = 0
        for vertex_idx, entity in enumerate(vertexSet):
            for _e_idx, span in enumerate(entity):
                span_unique_pos = f"{span['pos'][0]}_{span['pos'][1]}"
                span_pos_to_idx[span_unique_pos] = num_entities
                entity_span_indices[example_idx, num_entities, 0] = span['pos'][0]
                entity_span_indices[example_idx, num_entities, 1] = span['pos'][1] - 1  # to include end index
                num_entities += 1
        return span_pos_to_idx, num_entities

    def assemble_relation(self, head_mentions_indices, head_mentions_indices_mask,
                          tail_mentions_indices, tail_mentions_indices_mask,
                          ht_comb_indices, ht_comb_mask,
                          ii, span_pos_to_idx, hlist, tlist, limit=max_num_mentions_per_entity):
        num_H, num_T = 0, 0
        for hh in hlist[:limit]:
            head_mentions_indices[ii, num_H] = span_pos_to_idx[f"{hh['pos'][0]}_{hh['pos'][1]}"]
            head_mentions_indices_mask[ii,  num_H] = 1
            num_H += 1
        for tt in tlist[:limit]:
            tail_mentions_indices[ii, num_T] = span_pos_to_idx[f"{tt['pos'][0]}_{tt['pos'][1]}"]
            tail_mentions_indices_mask[ii, num_T] = 1
            num_T += 1
        # h1,h2; t1,t2,t3 :
        # ht_comb_indices[..., 0]: h1 h1 h1 h2 h2 h2
        # ht_comb_indices[..., 1]: t1 t2 t3 t1 t2 t3
        kk = 0
        for index_h, hh in enumerate(hlist[:limit]):
            for index_t, tt in enumerate(tlist[:limit]):
                ht_comb_indices[ii, kk, 0] = index_h
                ht_comb_indices[ii, kk, 1] = index_t
                # ht_pair_pos[ii,  kk] = \
                #     self.get_head_tail_relative_pos(hh, tt)
                ht_comb_mask[ii,  kk] = 1
                kk += 1
        return num_H, num_T, kk

    def assemble_adj_matrices(self, adj_matrices, ex_idx, doc_len, __matrices):
        for et_idx in range(self.num_edge_types):
            et = self.idx2edgeType[et_idx]
            if et in __matrices:
                matrix = __matrices[et]
                adj_matrices[ex_idx, et_idx] = \
                    sp.coo_matrix((matrix['data'], (matrix['ind']['row'], matrix['ind']['col'])),
                                  shape=(doc_len, doc_len), dtype=np.int8).toarray()
            else:  # fill 0
                adj_matrices[ex_idx, et_idx] = \
                    sp.coo_matrix((doc_len, doc_len), dtype=np.int8).toarray()

    def get_rel_dist(self, men_indices, doc_len) -> np.ndarray:
        tok_idx = np.linspace(0, doc_len-1, num=doc_len, dtype=np.int32)
        rel_dist = np.expand_dims(tok_idx, axis=1) - np.expand_dims(men_indices, axis=0)
        indices = np.argmin(np.abs(rel_dist), axis=1)
        return rel_dist[tok_idx, indices]

    def get_triplets(self, ins, doc_index) -> List[Tuple[int, str, int, int, int]]:
        triplets = []
        for pos_rel in ins['labels']:
            triplets.append((doc_index, ins['title'], pos_rel['h'], pos_rel['t'], pos_rel['r']))
        for h_idx, t_idx in ins['na_triple']:
            if self.train_hypernym_filter:
                h_KBID = ins['vertexSet'][h_idx][0]['KB_ID']
                t_KBID = ins['vertexSet'][t_idx][0]['KB_ID']
                neg = [str(ins['title']), h_KBID, t_KBID]
                if neg in self.negs_to_remove:
                    continue
            triplets.append((doc_index, ins['title'], h_idx, t_idx, 0))
        return triplets

    def get_test_entity_pairs(self, ins, doc_index):
        entity_pairs = []
        L = len(ins['vertexSet'])
        for h_idx in range(L):
            for t_idx in range(L):
                hlist = ins['vertexSet'][h_idx]
                tlist = ins['vertexSet'][t_idx]
                h_type = hlist[0]['type']
                t_type = tlist[0]['type']
                assert h_type in ['Chemical', 'Disease']
                assert t_type in ['Chemical', 'Disease']
                if h_idx != t_idx and (h_type == 'Chemical' and t_type == 'Disease'):
                    if self.test_hypernym_filter:
                        neg = [str(ins['title']), hlist[0]['KB_ID'], tlist[0]['KB_ID']]
                        if neg in self.negs_to_remove:
                            continue
                    entity_pairs.append((doc_index, ins['title'], h_idx, t_idx, 0))
        return entity_pairs

    def get_entity_indices(self, ins) -> Dict[int, np.ndarray]:
        entity_indices = {}
        for en_idx, entity in enumerate(ins['vertexSet']):
            _indices = []
            for m in entity:
                _indices.extend(list(range(m['pos'][0], m['pos'][1])))
            entity_indices[en_idx] = np.array(_indices)
        return entity_indices

    def get_train_batch(self) -> Iterator[Dict]:
        random.shuffle(self.train_order)
        context_idxs = np.zeros((self.batch_size, self.max_length), dtype=np.int64)
        if self.use_bert_embedding:
            bert_embeddings = np.zeros((self.batch_size, self.max_length, self.bert_embedd_dim), dtype=np.float32)
        input_lengths = np.zeros(self.batch_size, dtype=np.int16)
        dist_e1 = np.zeros((self.batch_size, self.max_length), dtype=np.int16)
        dist_e2 = np.zeros((self.batch_size, self.max_length), dtype=np.int16)
        adj_matrices = np.zeros((self.batch_size, self.num_edge_types, self.max_length, self.max_length), dtype=np.int8)
        relation_label = np.zeros(self.batch_size, dtype=np.int8)
        relation_mask = np.zeros(self.batch_size, dtype=np.int8)
        entity_span_indices = np.zeros((self.batch_size, max_num_mentions_per_entity*2, 2), dtype=np.int16)
        head_mentions_indices = np.zeros((self.batch_size, mentions_limit), dtype=np.int16)
        head_mentions_indices_mask = np.zeros((self.batch_size, mentions_limit), dtype=np.int8)
        tail_mentions_indices = np.zeros((self.batch_size, mentions_limit), dtype=np.int16)
        tail_mentions_indices_mask = np.zeros((self.batch_size, mentions_limit), dtype=np.int8)
        ht_comb_indices = np.zeros((self.batch_size, mentions_limit * mentions_limit, 2), dtype=np.int16)
        ht_comb_mask = np.zeros((self.batch_size, mentions_limit * mentions_limit), dtype=np.int8)

        nums = {
            'ex_idx': 0,
            'batch_max_num_entities': 1,
            'batch_max_num_H': 1,
            'batch_max_num_T': 1,
            'batch_max_num_combination': 1,
            'indexes': []
        }

        def reset():
            dist_e1.fill(self.pad_position_idx)
            dist_e2.fill(self.pad_position_idx)
            relation_label.fill(IGNORE_INDEX)
            for _tensor in [context_idxs, entity_span_indices, head_mentions_indices, tail_mentions_indices,
                            tail_mentions_indices_mask, head_mentions_indices_mask, ht_comb_mask,
                            ht_comb_indices, relation_mask, input_lengths, adj_matrices]:
                _tensor.fill(0)
            for k in nums.keys():
                nums[k] = 1
            nums['ex_idx'] = 0
            nums['indexes'] = []

        def make_input(_cur_bsz, _max_c_len):
            return {
                'context_idxs': self._to_tensor(context_idxs[:_cur_bsz, :_max_c_len], 'long'),
                'bert_feature': self._to_tensor(bert_embeddings[:_cur_bsz, :_max_c_len, :]) if self.use_bert_embedding else None,
                'input_lengths': self._to_tensor(input_lengths[:_cur_bsz], 'long'),
                'dist_e1': self._to_tensor(dist_e1[:_cur_bsz, :_max_c_len], 'long'),
                'dist_e2': self._to_tensor(dist_e2[:_cur_bsz, :_max_c_len], 'long'),
                'adj_matrices': self._to_tensor(adj_matrices[:_cur_bsz, :, :_max_c_len, :_max_c_len], 'long'),
                'for_relation_repr': {
                    'entity_span_indices':
                        self._to_tensor(entity_span_indices[:_cur_bsz, :nums['batch_max_num_entities'], :], 'long'),
                    'head_mentions_indices':
                        self._to_tensor(head_mentions_indices[:_cur_bsz, :nums['batch_max_num_H']], 'long'),
                    'head_mentions_indices_mask':
                        self._to_tensor(head_mentions_indices_mask[:_cur_bsz, :nums['batch_max_num_H']], 'long'),
                    'tail_mentions_indices':
                        self._to_tensor(tail_mentions_indices[:_cur_bsz, :nums['batch_max_num_T']], 'long'),
                    'tail_mentions_indices_mask':
                        self._to_tensor(tail_mentions_indices_mask[:_cur_bsz, :nums['batch_max_num_T']], 'long'),
                    'ht_comb_indices':
                        self._to_tensor(ht_comb_indices[:_cur_bsz, :nums['batch_max_num_combination'], :], 'long'),
                    'ht_comb_mask':
                        self._to_tensor(ht_comb_mask[:_cur_bsz, :nums['batch_max_num_combination']], 'long'),
                },
                'relation_label': self._to_tensor(relation_label[:_cur_bsz], 'long'),
                'relation_mask': self._to_tensor(relation_mask[:_cur_bsz]),
                'indexes': copy.deepcopy(nums['indexes'])
            }

        reset()
        num_entity_pairs = 0

        with h5py.File(self.train_bert_feature, 'r') as fin:
            for _bin in range(self.train_bins):
                start_id = _bin * self.bin_size
                cur_bin_sz = min(self.bin_size, self.train_len - start_id)
                cur_bin = list(self.train_order[start_id: start_id + cur_bin_sz])
                cur_bin.sort(key=lambda x: np.sum(self.data_train_word[x] > 0), reverse=True)

                for doc_index in cur_bin:  # each document
                    ins = self.train_file[doc_index]
                    vertexSet = ins['vertexSet']
                    doc_key = ins['title']
                    if self.use_bert_embedding:
                        ex_embedding = self.padding(np.array(fin[str(doc_index)]), self.max_length, self.bert_embedd_dim)
                    sparse_matrices = self.train_sparse_adj_matrices[doc_key]
                    doc_len = self.train_doc_lens[ins['title']]
                    doc_len2 = np.sum(self.data_train_word[doc_index, :] > 0)
                    assert doc_len == doc_len2, f"length error: {doc_len} != {doc_len2}"
                    triplets = self.get_triplets(ins, doc_index)
                    entity_indices = self.get_entity_indices(ins)

                    random.shuffle(triplets)
                    for triplet in triplets:
                        nums['indexes'].append(doc_index)
                        h_idx, t_idx = triplet[2], triplet[3]
                        relation_label[nums['ex_idx']] = triplet[4]
                        relation_mask[nums['ex_idx']] = 1
                        context_idxs[nums['ex_idx']] = self.data_train_word[doc_index, :]
                        if self.use_bert_embedding:
                            bert_embeddings[nums['ex_idx']] = ex_embedding[:, :]
                        input_lengths[nums['ex_idx']] = doc_len
                        dist_e1[nums['ex_idx'], :doc_len] = self.get_rel_dist(entity_indices[h_idx], doc_len) + self.max_length
                        dist_e2[nums['ex_idx'], :doc_len] = self.get_rel_dist(entity_indices[t_idx], doc_len) + self.max_length
                        self.assemble_adj_matrices(adj_matrices, nums['ex_idx'], self.max_length, sparse_matrices)
                        span_pos_to_idx, num_entities = \
                            self.assemble_mentions(entity_span_indices, nums['ex_idx'], [vertexSet[h_idx], vertexSet[t_idx]])
                        nums['batch_max_num_entities'] = max(nums['batch_max_num_entities'], num_entities)

                        hlist = vertexSet[h_idx]
                        tlist = vertexSet[t_idx]
                        # real head-span end-span pairs
                        random.shuffle(hlist)
                        random.shuffle(tlist)
                        num_H, num_T, num_relation_comb = \
                            self.assemble_relation(head_mentions_indices, head_mentions_indices_mask,
                                                   tail_mentions_indices, tail_mentions_indices_mask,
                                                   ht_comb_indices, ht_comb_mask,
                                                   nums['ex_idx'], span_pos_to_idx, hlist, tlist, mentions_limit)
                        nums['batch_max_num_H'] = max(nums['batch_max_num_H'], num_H)
                        nums['batch_max_num_T'] = max(nums['batch_max_num_T'], num_T)
                        nums['batch_max_num_combination'] = max(nums['batch_max_num_combination'], num_relation_comb)

                        head_mentions_indices_mask[:, 0] = 1
                        tail_mentions_indices_mask[:, 0] = 1
                        ht_comb_mask[:, 0] = 1

                        nums['ex_idx'] += 1
                        num_entity_pairs += 1
                        # reach the batch_size or end
                        if nums['ex_idx'] == self.batch_size or (
                                _bin+1 == self.train_bins
                                and doc_index == cur_bin[-1]
                                and triplet == triplets[-1]
                        ):
                            cur_bsz = nums['ex_idx']
                            max_c_len = int(np.max(input_lengths))
                            yield make_input(cur_bsz, max_c_len)
                            reset()

        self.num_train_entity_pairs = num_entity_pairs
        self.print_train_num_pos_neg()

    def get_test_batch(self) -> Iterator[Dict]:
        context_idxs = np.zeros((self.test_batch_size, self.max_length), dtype=np.int64)
        if self.use_bert_embedding:
            bert_embeddings = np.zeros((self.test_batch_size, self.max_length, self.bert_embedd_dim), dtype=np.float32)
        input_lengths = np.zeros(self.test_batch_size, dtype=np.int16)
        dist_e1 = np.zeros((self.test_batch_size, self.max_length), dtype=np.int16)
        dist_e2 = np.zeros((self.test_batch_size, self.max_length), dtype=np.int16)
        adj_matrices = np.zeros((self.test_batch_size, self.num_edge_types, self.max_length, self.max_length), dtype=np.int8)
        relation_label = np.zeros(self.test_batch_size, dtype=np.int8)
        relation_mask = np.zeros(self.test_batch_size, dtype=np.int8)
        limit = max_num_mentions_per_entity
        entity_span_indices = np.zeros((self.test_batch_size, limit*2, 2), dtype=np.int16)
        head_mentions_indices = np.zeros((self.test_batch_size, limit), dtype=np.int16)
        head_mentions_indices_mask = np.zeros((self.test_batch_size, limit), dtype=np.int8)
        tail_mentions_indices = np.zeros((self.test_batch_size, limit), dtype=np.int16)
        tail_mentions_indices_mask = np.zeros((self.test_batch_size, limit), dtype=np.int8)
        ht_comb_indices = np.zeros((self.test_batch_size, limit * limit, 2), dtype=np.int16)
        ht_comb_mask = np.zeros((self.test_batch_size, limit * limit), dtype=np.int8)

        nums = {
            'ex_idx': 0,
            'batch_max_num_entities': 1,
            'batch_max_num_H': 1,
            'batch_max_num_T': 1,
            'batch_max_num_combination': 1,
            'indexes': [],
            'entity_pairs': []
        }

        def reset():
            dist_e1.fill(self.pad_position_idx)
            dist_e2.fill(self.pad_position_idx)
            relation_label.fill(IGNORE_INDEX)
            for _tensor in [context_idxs, entity_span_indices, head_mentions_indices, tail_mentions_indices,
                            tail_mentions_indices_mask, head_mentions_indices_mask, ht_comb_mask,
                            ht_comb_indices, relation_mask, input_lengths, adj_matrices]:
                _tensor.fill(0)
            for k in nums.keys():
                nums[k] = 1
            nums['ex_idx'] = 0
            nums['indexes'] = []
            nums['entity_pairs'] = []

        def make_input(_cur_bsz, _max_c_len):
            return {
                'context_idxs': self._to_tensor(context_idxs[:_cur_bsz, :_max_c_len], 'long'),
                'bert_feature': self._to_tensor(bert_embeddings[:_cur_bsz, :_max_c_len, :]) if self.use_bert_embedding else None,
                'input_lengths': self._to_tensor(input_lengths[:_cur_bsz], 'long'),
                'dist_e1': self._to_tensor(dist_e1[:_cur_bsz, :_max_c_len], 'long'),
                'dist_e2': self._to_tensor(dist_e2[:_cur_bsz, :_max_c_len], 'long'),
                'adj_matrices': self._to_tensor(adj_matrices[:_cur_bsz, :, :_max_c_len, :_max_c_len], 'long'),
                'for_relation_repr': {
                    'entity_span_indices':
                        self._to_tensor(entity_span_indices[:_cur_bsz, :nums['batch_max_num_entities'], :], 'long'),
                    'head_mentions_indices':
                        self._to_tensor(head_mentions_indices[:_cur_bsz, :nums['batch_max_num_H']], 'long'),
                    'head_mentions_indices_mask':
                        self._to_tensor(head_mentions_indices_mask[:_cur_bsz, :nums['batch_max_num_H']], 'long'),
                    'tail_mentions_indices':
                        self._to_tensor(tail_mentions_indices[:_cur_bsz, :nums['batch_max_num_T']], 'long'),
                    'tail_mentions_indices_mask':
                        self._to_tensor(tail_mentions_indices_mask[:_cur_bsz, :nums['batch_max_num_T']], 'long'),
                    'ht_comb_indices':
                        self._to_tensor(ht_comb_indices[:_cur_bsz, :nums['batch_max_num_combination'], :], 'long'),
                    'ht_comb_mask':
                        self._to_tensor(ht_comb_mask[:_cur_bsz, :nums['batch_max_num_combination']], 'long'),
                },
                'relation_label': self._to_tensor(relation_label[:_cur_bsz], 'long'),
                'relation_mask': self._to_tensor(relation_mask[:_cur_bsz]),
                'indexes': copy.deepcopy(nums['indexes']),
                'entity_pairs': nums['entity_pairs']
            }

        reset()

        num_entity_pairs = 0
        with h5py.File(self.test_bert_feature, 'r') as fin:
            for doc_index in self.test_doc_indices:  # each document
                ins = self.test_file[doc_index]
                vertexSet = ins['vertexSet']
                doc_key = ins['title']
                if self.use_bert_embedding:
                    ex_embedding = self.padding(np.array(fin[str(doc_index)]), self.max_length, self.bert_embedd_dim)
                sparse_matrices = self.test_sparse_adj_matrices[doc_key]
                doc_len = self.test_doc_lens[ins['title']]
                doc_len2 = np.sum(self.data_test_word[doc_index, :] > 0)
                assert doc_len == doc_len2, f"length error: {doc_len} != {doc_len2}"
                entity_pairs = self.get_test_entity_pairs(ins, doc_index)
                entity_indices = self.get_entity_indices(ins)

                for ep in entity_pairs:
                    nums['entity_pairs'].append(copy.deepcopy(ep))
                    nums['indexes'].append(doc_index)
                    h_idx, t_idx = ep[2], ep[3]
                    relation_label[nums['ex_idx']] = ep[4]  # dummy label 0
                    relation_mask[nums['ex_idx']] = 1
                    context_idxs[nums['ex_idx']] = self.data_test_word[doc_index, :]
                    if self.use_bert_embedding:
                        bert_embeddings[nums['ex_idx']] = ex_embedding[:, :]
                    input_lengths[nums['ex_idx']] = doc_len
                    dist_e1[nums['ex_idx'], :doc_len] = self.get_rel_dist(entity_indices[h_idx], doc_len) + self.max_length
                    dist_e2[nums['ex_idx'], :doc_len] = self.get_rel_dist(entity_indices[t_idx], doc_len) + self.max_length
                    self.assemble_adj_matrices(adj_matrices, nums['ex_idx'], self.max_length, sparse_matrices)
                    span_pos_to_idx, num_entities = \
                        self.assemble_mentions(entity_span_indices, nums['ex_idx'], [vertexSet[h_idx], vertexSet[t_idx]])
                    nums['batch_max_num_entities'] = max(nums['batch_max_num_entities'], num_entities)

                    num_H, num_T, num_relation_comb = \
                        self.assemble_relation(head_mentions_indices, head_mentions_indices_mask,
                                               tail_mentions_indices, tail_mentions_indices_mask,
                                               ht_comb_indices, ht_comb_mask,
                                               nums['ex_idx'], span_pos_to_idx,
                                               vertexSet[h_idx], vertexSet[t_idx], mentions_limit)
                    nums['batch_max_num_H'] = max(nums['batch_max_num_H'], num_H)
                    nums['batch_max_num_T'] = max(nums['batch_max_num_T'], num_T)
                    nums['batch_max_num_combination'] = max(nums['batch_max_num_combination'], num_relation_comb)

                    head_mentions_indices_mask[:, 0] = 1
                    tail_mentions_indices_mask[:, 0] = 1
                    ht_comb_mask[:, 0] = 1

                    nums['ex_idx'] += 1
                    num_entity_pairs += 1
                    # reach the batch_size or end
                    if nums['ex_idx'] == self.test_batch_size or (
                            doc_index == self.test_doc_indices[-1]
                            and ep == entity_pairs[-1]
                    ):
                        cur_bsz = nums['ex_idx']
                        max_c_len = int(np.max(input_lengths))
                        yield make_input(cur_bsz, max_c_len)
                        reset()

        self.num_test_entity_pairs = num_entity_pairs
        self.print_test_num_entity_pairs()

    def get_config_str(self) -> str:
        config_str = super(CDRGCNNTrainer_Softmax, self).get_config_str()
        # Make sure that params have the same name with the property.
        # E.g., 'lowercase' --> self.lowercase
        params = ['lowercase', 'train_hypernym_filter', 'test_hypernym_filter', 'avg_params',
                  'position_dim', 'inp_dropout', 'bi_affine_ff_dim', 'bi_affine_dropout',
                  'grad_clip_value', 'lr_decay', 'parallel_forward']
        config_lines = self.format_params(params)
        config_str += "".join(config_lines)
        return config_str + self.get_format_params_end_line()


