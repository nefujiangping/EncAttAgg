# coding: utf-8
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import os
import time
import json
from torch.optim.lr_scheduler import ReduceLROnPlateau

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


class CDRBaseTrainer_Softmax(Functional):

    def __init__(self, params) -> None:
        super(CDRBaseTrainer_Softmax, self).__init__(params)
        self.lowercase = params['lowercase']
        self.train_hypernym_filter = params['train_hypernym_filter']
        self.test_hypernym_filter = params['test_hypernym_filter']
        self.train_on_trainanddev = params['train_on_trainanddev']
        self.avg_params = params['avg_params']
        self.averaged_params = {}

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
        predict_mins_per_epoch = int((self.train_batches / self.period) * elapsed / 60.)
        self.logging('| epoch {:2d} | step {:4d} |  min/{:<3d}b {:2.1f} | train loss {:5.5f} '
                     '| not NA acc: {:4.2f} | NA acc: {:4.2f} | min/epoch {:3d}'.format(epoch, global_step, self.period, elapsed/60., cur_loss, self.acc_not_NA.get(), self.acc_NA.get(), predict_mins_per_epoch))

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
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=self.init_lr)
        if state_dict:
            optimizer.load_state_dict(state_dict['optimizer'])

        cross_entropy_loss = nn.CrossEntropyLoss(reduction='none', ignore_index=IGNORE_INDEX)

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
                    relation_mask = data['relation_mask']
                    context_ner = data['context_ner']
                    context_char_idxs = data['context_char_idxs']
                    ht_pair_pos = data['ht_pair_pos']

                    dis_h_2_t = ht_pair_pos + 10
                    dis_t_2_h = -ht_pair_pos + 10
                    before_forward = time.time()
                    predict_re = model(context_idxs, context_pos, context_ner, context_char_idxs, input_mask,
                                       for_relation_repr, relation_mask, dis_h_2_t, dis_t_2_h, bert_feature,
                                       relation_label, True)
                    forward_time += (time.time() - before_forward)
                    assert predict_re.ndim == 3 and predict_re.size()[-1] == 2
                    N, R, _ = predict_re.size()
                    # (N*R, 2)  <-->  (N*R)
                    loss = torch.sum(cross_entropy_loss(predict_re.view(N*R, 2), relation_label.view(-1))
                                     * relation_mask.contiguous().view(N*R)) / torch.sum(relation_mask)
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
        data_idx = 0
        eval_start_time = time.time()

        test_result = []
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
                indexes = data['indexes']

                dis_h_2_t = ht_pair_pos + 10
                dis_t_2_h = -ht_pair_pos + 10

                predict_re, _ = model(context_idxs, context_pos, context_ner, context_char_idxs, input_mask,
                                      for_relation_repr, relation_mask, dis_h_2_t, dis_t_2_h,
                                      bert_feature, None, False)
                predict_re = torch.nn.functional.softmax(predict_re, dim=-1)
            # (N, R, 2) with value probability
            predict_re = predict_re.data.cpu().numpy()

            for ex_idx in range(len(labels)):
                index = indexes[ex_idx]
                ins = self.test_file[index]

                L = L_vertex[ex_idx]
                rel_idx = 0
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
                            r = np.argmax(predict_re[ex_idx, rel_idx])
                            if r == 1:
                                test_result.append({
                                    "index": index,
                                    "KB_ID": ins['title'],
                                    "h_idx": h_idx,
                                    "t_idx": t_idx,
                                    "r_idx": 1,  # to be compatible with DocRED
                                    "intrain": False  # to be compatible with DocRED
                                })

                            rel_idx += 1

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
        # if self.use_bert:
        self.train_bert_feature = f'{self.bert_embedding_dir}/{self.bert_embedd_dim}_dev_train.h5'

        print("Finish reading")
        assert self.data_train_word.shape[0] == 1000
        assert prefix == 'dev_train'

        if self.train_on_trainanddev:
            self.train_len = ins_num = self.data_train_word.shape[0]

            self.train_order = list(range(ins_num))
            self.train_batches = ins_num // self.batch_size
            if ins_num % self.batch_size != 0:
                self.train_batches += 1
        else:
            self.train_len = 500
            self.train_order = list(range(self.train_len))
            self.train_batches = self.train_len // self.batch_size
            if self.train_len % self.batch_size != 0:
                self.train_batches += 1

        print(f'train index: [{min(self.train_order)}, {max(self.train_order)}]')

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
        # assert (self.test_len == len(self.test_file))

        print("Finish reading")

        self.test_batches = self.test_len // self.test_batch_size
        if self.test_len % self.test_batch_size != 0:
            self.test_batches += 1

        if prefix == 'dev_train':  # dev is in train_file
            self.test_order = list(range(1000))[500:]
        elif prefix == 'dev_test':
            self.test_order = list(range(self.test_len))

        print(f'test index: [{min(self.test_order)}, {max(self.test_order)}]')

    def get_train_batch(self) -> Iterator[Dict]:
        raise NotImplementedError

    def get_test_batch(self) -> Iterator[Dict]:
        raise NotImplementedError

    def get_config_str(self) -> str:
        config_str = super(CDRBaseTrainer_Softmax, self).get_config_str()
        # Make sure that params have the same name with the property.
        # E.g., 'lowercase' --> self.lowercase
        params = ['lowercase', 'train_hypernym_filter', 'test_hypernym_filter', 'avg_params']
        config_lines = self.format_params(params)
        config_str += "".join(config_lines)
        return config_str


