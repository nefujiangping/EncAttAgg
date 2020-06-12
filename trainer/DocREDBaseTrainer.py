# coding: utf-8
from typing import Iterator, Dict
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import os
import time
import json
import sklearn.metrics

from misc.util import TensorboardWriter
from misc.metrics import coref_vs_non_coref_performance, diff_dist_performance
from trainer.functional import Functional

# max statistic in train, dev and test set of DocRED
max_num_mentions_per_example = 84
max_num_entities_per_example = 42
max_num_mentions_per_entity = 23

max_num_head_mentions_X_tail_mentions = 204
mentions_limit = 8
test_mentions_limit = max_num_mentions_per_entity

IGNORE_INDEX = -100


class BaseTrainer(Functional):

    def __init__(self, params) -> None:
        super(BaseTrainer, self).__init__(params)
        self.filter_by_entity_type = params['filter_by_entity_type']

    def load_train_data(self) -> None:
        print("Reading training data...")
        prefix = self.train_prefix

        print('train', prefix)
        self.data_train_word = np.load(os.path.join(self.data_path, prefix + '_word.npy'))
        self.data_train_pos = np.load(os.path.join(self.data_path, prefix + '_pos.npy'))
        self.data_train_ner = np.load(os.path.join(self.data_path, prefix + '_ner.npy'))
        self.data_train_char = np.load(os.path.join(self.data_path, prefix + '_char.npy'))
        self.train_file = json.load(open(os.path.join(self.data_path, prefix + '.json')))
        self.train_bert_feature = f'{self.bert_embedding_dir}/{self.bert_embedd_dim}_dev_train.h5'

        print("Finish reading")

        self.train_len = ins_num = self.data_train_word.shape[0]
        assert (self.train_len == len(self.train_file))

        self.train_order = list(range(ins_num))
        self.train_batches = ins_num // self.batch_size
        if ins_num % self.batch_size != 0:
            self.train_batches += 1

    def load_test_data(self) -> None:
        print("Reading testing data...")
        self.data_word_vec = np.load(os.path.join(self.data_path, 'vec.npy'))
        self.data_char_vec = np.load(os.path.join(self.data_path, 'char_vec.npy'))
        self.rel2id = json.load(open(os.path.join(self.data_path, 'rel2id.json')))
        self.id2rel = {v: k for k, v in self.rel2id.items()}
        self.relPxx2name = json.load(open(os.path.join(self.data_path, 'PXX_to_idx_name.json')))
        self.relPxx2name = {pxx: idx_name[1] for pxx, idx_name in self.relPxx2name.items()}

        prefix = self.test_prefix
        print(prefix)
        self.is_test = ('dev_test' == prefix)
        self.data_test_word = np.load(os.path.join(self.data_path, prefix + '_word.npy'))
        self.data_test_pos = np.load(os.path.join(self.data_path, prefix + '_pos.npy'))
        self.data_test_ner = np.load(os.path.join(self.data_path, prefix + '_ner.npy'))
        self.data_test_char = np.load(os.path.join(self.data_path, prefix + '_char.npy'))
        self.test_file = json.load(open(os.path.join(self.data_path, prefix + '.json')))
        self.test_bert_feature = f'{self.bert_embedding_dir}/{self.bert_embedd_dim}_{prefix}.h5'

        self.test_len = self.data_test_word.shape[0]
        assert (self.test_len == len(self.test_file))

        print("Finish reading")

        self.test_batches = self.test_len // self.test_batch_size
        if self.test_len % self.test_batch_size != 0:
            self.test_batches += 1

        self.test_order = list(range(self.test_len))
        if not self.write_weights:
            self.test_order.sort(key=lambda x: np.sum(self.data_test_word[x] > 0), reverse=True)
        combs = json.load(open(f'{self.data_path}/DocRED_h_t_entity_type_combs_33.json'))
        self.DocRED_h_t_entity_type_combs = {(h_type, t_type) for (h_type, t_type) in combs}

    def entity_type_is_valid(self, v_h, v_t):
        return len({(h_m['type'], t_m['type']) for h_m in v_h for t_m in v_t} & self.DocRED_h_t_entity_type_combs) > 0

    def train(self, model_pattern):
        assert not self.is_test, 'Now is training.'
        summary_writer = TensorboardWriter(self.exp_id, self.summary_dir)
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
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=self.init_lr)
        if state_dict:
            optimizer.load_state_dict(state_dict['optimizer'])

        BCE = nn.BCEWithLogitsLoss(reduction='none')

        prev_lr = self.init_lr
        lr_scheduler = None
        if self.use_lr_scheduler:
            lr_scheduler = ReduceLROnPlateau(optimizer,
                                             mode='max', patience=3,
                                             factor=0.5, min_lr=5e-6, verbose=True)
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
                    bert_feature = data['bert_feature']
                    context_pos = data['context_pos']
                    for_relation_repr = data['for_relation_repr']
                    relation_label = data['relation_label']
                    input_mask = data['input_mask']
                    relation_multi_label = data['relation_multi_label']
                    relation_mask = data['relation_mask']
                    context_ner = data['context_ner']
                    context_char_idxs = data['context_char_idxs']
                    ht_pair_pos = data['ht_pair_pos']
                    relation_weight = data['relation_weight'] if 'relation_weight' in data else None

                    dis_h_2_t = ht_pair_pos + 10
                    dis_t_2_h = -ht_pair_pos + 10
                    before_forward = time.time()
                    predict_re = model(context_idxs, context_pos, context_ner, context_char_idxs, input_mask,
                                       for_relation_repr, relation_mask, dis_h_2_t, dis_t_2_h, bert_feature,
                                       relation_label, True)
                    forward_time += (time.time() - before_forward)
                    if relation_weight is None:
                        loss = torch.sum(BCE(predict_re, relation_multi_label) * relation_mask.unsqueeze(2)) / (
                                self.relation_num * torch.sum(relation_mask))
                    else:
                        loss = torch.sum(BCE(predict_re, relation_multi_label) * relation_weight.unsqueeze(2)) / (
                                self.relation_num * torch.sum(relation_mask))
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
                # do test
                if (epoch + 1) > self.epoch_start_to_eval:
                    model.eval()
                    self._do_test(ori_model, model, epoch, lr_scheduler, summary_writer, optimizer, train_states)
                    new_lr = optimizer.param_groups[0]['lr']
                    if new_lr < prev_lr:
                        self.logging('epoch {:3d}: reducing learning rate to {}.'.format(epoch, new_lr))
                    prev_lr = optimizer.param_groups[0]['lr']
                    model.train()
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

    def _do_train_log(self, total_loss, epoch, start_time, global_step):
        cur_loss = total_loss / self.period
        elapsed = time.time() - start_time
        predict_mins_per_epoch = int((self.train_batches / self.period) * elapsed / 60.)
        self.logging('| epoch {:2d} | step {:4d} |  min/{:<3d}b {:2.1f} | train loss {:5.5f} '
                     '| not NA acc: {:4.2f} | NA acc: {:4.2f} | min/epoch {:3d}'.format(epoch, global_step, self.period, elapsed/60., cur_loss, self.acc_not_NA.get(), self.acc_NA.get(), predict_mins_per_epoch))

    def _do_test(self, ori_model, model, epoch, lr_scheduler, summary_writer, optimizer, train_states):
        self.logging('-' * 90)
        eval_start_time = time.time()
        f1s, all_f1, all_auc, input_theta, prec, recall, f1, auc = self.test(model)
        self.logging('| epoch {:3d} | eval_time: {:5.2f}s'.format(epoch, time.time() - eval_start_time))

        # logging for tensorboard
        summary_writer.log_inference_theta(input_theta, epoch)
        summary_writer.log_val_all_f1(all_f1, epoch)
        summary_writer.log_val_all_auc(all_auc, epoch)
        summary_writer.log_val_f1(f1, epoch)
        summary_writer.log_val_precision(prec, epoch)
        summary_writer.log_val_recall(recall, epoch)
        summary_writer.log_val_auc(auc, epoch)
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

        if f1 > self.best_scores['main_metric']['ign_f1']:
            self.best_scores['main_metric']['ign_f1'] = f1
            self.best_scores['main_metric']['auc'] = auc
            self.best_scores['main_metric']['epoch'] = epoch
            self.input_theta_of_best_epoch = input_theta
            tlt = time.localtime()
            time_str = f"{tlt[0]}/{tlt[1]}/{tlt[2]} {tlt[3]}:{tlt[4]}"
            log_str = f'=== {time_str}  save model at epoch {epoch} '
            summary_writer.log_text("{:<17} | epoch {:<2d} | "
                                    "Ign F1: {:3.2f}, Ign AUC: {:3.2f}, Prec: {:3.2f}, "
                                    "Recall: {:3.2f}; coref_f1: {:3.2f}, non-coref_f1: {:3.2f} "
                                    "| F1: {:3.2f}, AUC: {:3.2f} "
                                    .format(time_str, epoch, f1*100, auc*100, prec*100, recall*100,
                                            f1s['ign_coref']*100, f1s['ign_non-coref']*100,
                                            all_f1*100, all_auc*100), epoch)
            self.logging('='*50)
            self.logging(log_str + '='*(50-len(log_str)))
            self.logging('='*50)
            path = os.path.join(self.checkpoint_dir, self.exp_id)
            torch.save(to_save, path)
        self.logging('-' * 90)

    def test(self, model, output=False, input_theta=-1):
        if output:
            assert input_theta > 0, 'Give input_theta for test output.'
        test_result = []

        def fill_test_result(_ex_idx, _title, _index, _L, _label, _preds,
                             _hAt, _tAh, _on_mp, _for_attn_weights):
            rel_idx = 0
            pred_triplets_for_attn = []
            for h_idx in range(_L):
                for t_idx in range(_L):
                    if h_idx != t_idx:
                        if self.filter_by_entity_type:
                            hlist = self.test_file[_index]['vertexSet'][h_idx]
                            tlist = self.test_file[_index]['vertexSet'][t_idx]
                            if not self.entity_type_is_valid(hlist, tlist):
                                continue
                        for r in range(1, self.relation_num):
                            in_train = False  # For Ign F1/AUC
                            if (h_idx, t_idx, r) in _label:
                                if _label[(h_idx, t_idx, r)]:
                                    in_train = True
                            right = (h_idx, t_idx, r) in _label
                            _prob = float(_preds[_ex_idx, rel_idx, r])
                            test_result.append(
                                (
                                    right, _prob, in_train, _title, self.id2rel[r],
                                    _index, h_idx, t_idx, r
                                )
                            )
                            if self.write_weights:
                                if _prob > input_theta:
                                    pred_triplets_for_attn.append({
                                        "index": _index,
                                        "r_idx": r,
                                        "rel": self.relPxx2name[self.id2rel[r]],
                                        "h_t_idx": (h_idx, t_idx),
                                        "hAt_weights": _hAt[_ex_idx][rel_idx] if _hAt is not None else None,
                                        "tAh_weights": _tAh[_ex_idx][rel_idx] if _tAh is not None else None,
                                        "weights_on_mp": _on_mp[_ex_idx][rel_idx] if _on_mp is not None else None,
                                        "head_mentions_indices": _for_attn_weights['head_mentions_indices'][_ex_idx][rel_idx],
                                        "head_mentions_indices_mask": _for_attn_weights['head_mentions_indices_mask'][_ex_idx][rel_idx],
                                        "tail_mentions_indices": _for_attn_weights['tail_mentions_indices'][_ex_idx][rel_idx],
                                        "tail_mentions_indices_mask": _for_attn_weights['tail_mentions_indices_mask'][_ex_idx][rel_idx],
                                        "ht_comb_indices": _for_attn_weights['ht_comb_indices'][_ex_idx][rel_idx],
                                        "ht_comb_mask": _for_attn_weights['ht_comb_mask'][_ex_idx][rel_idx],
                                    })

                        rel_idx += 1
            return pred_triplets_for_attn

        if self.is_test:
            assert input_theta > 0., "predict test set now, please give `input_theta`."
            output_file = f"{self.exp_id}_{self.test_prefix}.json"
            assert not os.path.exists(output_file), f"{output_file} exists."

        eval_start_time = time.time()
        total_recall, batch_idx = 0, 0
        attn_weights = []
        for data in self.get_test_batch():
            with torch.no_grad():
                context_idxs = data['context_idxs']
                bert_feature = data['bert_feature']
                context_pos = data['context_pos']
                for_relation_repr = data['for_relation_repr']
                labels = data['labels']
                L_vertex = data['L_vertex']
                input_mask = data['input_mask']
                context_ner = data['context_ner']
                context_char_idxs = data['context_char_idxs']
                relation_mask = data['relation_mask']
                ht_pair_pos = data['ht_pair_pos']

                titles = data['titles']
                indexes = data['indexes']
                for_attn_weights = data['for_attn_weights'] if 'for_attn_weights' in data else None

                dis_h_2_t = ht_pair_pos + 10
                dis_t_2_h = -ht_pair_pos + 10

                predict_re, weights_to_return = \
                    model(context_idxs, context_pos, context_ner, context_char_idxs, input_mask,
                          for_relation_repr, relation_mask, dis_h_2_t, dis_t_2_h,
                          bert_feature, None, False)
                predict_re = torch.sigmoid(predict_re)

            predict_re = predict_re.data.cpu().numpy()
            hAt_weights, tAh_weights, weights_on_mp = None, None, None
            if self.write_weights and weights_to_return is not None:
                if weights_to_return['hAt_weights'] is not None:
                    hAt_weights = weights_to_return['hAt_weights'].data.cpu().numpy().tolist()
                if weights_to_return['tAh_weights'] is not None:
                    tAh_weights = weights_to_return['tAh_weights'].data.cpu().numpy().tolist()
                if weights_to_return['weights_on_mp'] is not None:
                    weights_on_mp = weights_to_return['weights_on_mp'].data.cpu().numpy().tolist()
            for ex_idx in range(len(labels)):
                label = labels[ex_idx]
                index = indexes[ex_idx]
                total_recall += len(label)
                pred_triplets_for_attn = \
                    fill_test_result(ex_idx, titles[ex_idx], index, L_vertex[ex_idx], label, predict_re,
                                     hAt_weights, tAh_weights, weights_on_mp, for_attn_weights)
                if self.write_weights:
                    attn_weights.append({
                        "index": index,
                        "entity_span_indices": for_attn_weights['entity_span_indices'][ex_idx],
                        "idx_to_span_pos": for_attn_weights['idx_to_span_pos'][ex_idx],
                        "pred_triplets": pred_triplets_for_attn
                    })
            batch_idx += 1

            if batch_idx % self.period == 0:
                print('| step {:3d} | time: {:5.2f}'.format(batch_idx // self.period, (time.time() - eval_start_time)))
                eval_start_time = time.time()

        print('total_recall', total_recall)
        if total_recall == 0:
            total_recall = 1  # for test

        if output:  # output decode results/attention weights
            _out_file_path = f"{self.exp_id}_{self.test_prefix}.json"
            self._save_test_output(test_result, input_theta, _out_file_path)
            ret = [None]*8
            if not self.is_test:
                decode_results = json.load(open(_out_file_path))
                f1s = coref_vs_non_coref_performance(self.test_file, decode_results)
                diff_dist_f1s = diff_dist_performance(self.test_file, decode_results)
                del decode_results
                f1s.update(diff_dist_f1s)
                ret[0] = f1s
            if self.write_weights:
                _out_file_path = f"{self.exp_id}_{self.test_prefix}_attn_weights.json"
                json.dump(attn_weights, open(_out_file_path, 'w'))
            return ret

        # sort by the probability, for computing AUC
        test_result.sort(key=lambda x: x[1], reverse=True)

        w, pr_x, pr_y = self._compute_f1(test_result, total_recall, input_theta)
        f1_arr = (2 * pr_x * pr_y / (pr_x + pr_y + 1e-20))
        all_f1 = f1_arr.max()
        f1_pos = f1_arr.argmax()
        theta = test_result[f1_pos][1]

        if input_theta == -1:
            w = f1_pos
            input_theta = theta

        f1_w = f1_arr[w]
        del f1_arr

        all_auc = sklearn.metrics.auc(x=pr_x, y=pr_y)
        del pr_x, pr_y

        if not self.is_test:
            self.logging('ALL  : Theta {:3.4f} | F1 {:3.2f} | AUC {:3.2f}'.format(theta, all_f1*100, all_auc*100))
        else:
            self.logging('ma_f1 {:3.2f} | input_theta {:3.4f} '
                         'test_result F1 {:3.2f} | AUC {:3.2f}'.format(all_f1*100, input_theta, f1_w*100, all_auc*100))

        f1s = None
        if not self.is_test:  # For training, evaluation on dev set
            f1s = coref_vs_non_coref_performance(self.test_file, test_result[:w + 1])
            diff_dist_f1s = diff_dist_performance(self.test_file, test_result[:w + 1])
            f1s.update(diff_dist_f1s)

        w, pr_x, pr_y = self._compute_ign_f1(test_result, total_recall, input_theta)
        f1_arr = (2 * pr_x * pr_y / (pr_x + pr_y + 1e-20))
        f1_w = f1_arr[w]
        f1 = f1_arr.max()
        max_f1_pos = f1_arr.argmax()
        prec = pr_y[max_f1_pos]
        recall = pr_x[max_f1_pos]
        del f1_arr
        _out_file_path = f"{self.exp_id}_{self.test_prefix}_pr.npz"
        np.savez(_out_file_path, pr_x=pr_x, pr_y=pr_y)  # to draw PR curve
        auc = sklearn.metrics.auc(x=pr_x, y=pr_y)
        self.logging(
            'Ignore ma_f1 {:3.2f} | prec {:3.2f} | recall {:3.2f} |'
            ' input_theta {:3.4f} test_result F1 {:3.2f} | AUC {:3.2f}'.format(
                f1*100, prec*100, recall*100, input_theta, f1_w*100, auc*100))

        return f1s, all_f1, all_auc, input_theta, prec, recall, f1, auc

    def _save_test_output(self, test_result, theta, output_file):
        _output = []
        for x in test_result:
            if x[1] <= theta:  # If probability > threshold then see it as prediction
                continue
            _idx = x[-4]
            _ins = self.test_file[_idx]
            _h_idx, _t_idx = x[-3], x[-2]
            _h_r_t = '{} | {} | {}'.format(_ins['vertexSet'][_h_idx][0]['name'], self.relPxx2name[x[-5]], _ins['vertexSet'][_t_idx][0]['name'])
            _output.append({
                'index': x[-4],
                'h_idx': x[-3],
                't_idx': x[-2],
                'r_idx': x[-1],
                'r': x[-5],
                'title': x[-6],
                'name': _h_r_t,
                'prob': '{:.4f}'.format(x[1]),
                'intrain': x[2],
                'right': x[0]
            })
        json.dump(_output, open(output_file, "w"), ensure_ascii=False, indent=4)

    def _compute_f1(self, test_result, total_recall, input_theta):
        pr_x, pr_y = [], []
        correct = w = 0

        for i, item in enumerate(test_result):
            correct += item[0]
            pr_y.append(float(correct) / (i + 1))
            pr_x.append(float(correct) / total_recall)
            if item[1] > input_theta:
                w = i

        pr_x = np.asarray(pr_x, dtype='float32')
        pr_y = np.asarray(pr_y, dtype='float32')
        return w, pr_x, pr_y

    def _compute_ign_f1(self, test_result, total_recall, input_theta):
        pr_x, pr_y = [], []
        correct = correct_in_train = 0
        w = 0
        for i, item in enumerate(test_result):
            correct += item[0]
            if item[0] & item[2]:
                correct_in_train += 1
            if correct_in_train == correct:
                p = 0
            else:
                p = float(correct - correct_in_train) / (i + 1 - correct_in_train)
            pr_y.append(p)
            pr_x.append(float(correct) / total_recall)
            if item[1] > input_theta:
                w = i
        pr_x = np.asarray(pr_x, dtype='float32')
        pr_y = np.asarray(pr_y, dtype='float32')
        return w, pr_x, pr_y

    def testall(self, model_pattern, input_theta):
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
        if self.use_gpu:
            model.cuda()
        model.eval()
        rets = self.test(model, False, input_theta)
        f1s = rets[0]
        if not self.is_test and f1s is not None:  # dev
            for key, value in f1s.items():
                self.logging('{:<15} | {:.2f} '.format(key, value*100))

    def get_train_batch(self) -> Iterator[Dict]:
        raise NotImplementedError

    def get_test_batch(self) -> Iterator[Dict]:
        raise NotImplementedError

    def get_config_str(self) -> str:
        config_str = super(BaseTrainer, self).get_config_str()
        attrs = ['filter_by_entity_type']
        return config_str + "".join(self.format_params(attrs))






