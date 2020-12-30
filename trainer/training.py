from typing import Dict, Any
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from transformers import AutoTokenizer, AdamW
from misc.metrics import compute_f1, compute_ign_f1, Accuracy
from sklearn.metrics import auc

from data_loader.data_loader import BertEncDataSet, collate_fn_bert_enc
from tqdm import trange, tqdm
import time
import json
import wandb


keys = [
    "head_mentions_indices", "head_mentions_indices_mask",
    "tail_mentions_indices", "tail_mentions_indices_mask",
    "ht_comb_indices", "ht_comb_mask"
]


class Trainer(object):

    def __init__(self, config):
        self.config = config
        self.rel2id = json.load(open(f"{config.data_dir}/rel2id.json"))
        self.id2rel = {v: k for k, v in self.rel2id.items()}
        self.id2word = json.load(open(f'{config.data_dir}/word2id.json'))
        self.id2word = {idx: word for word, idx in self.id2word.items()}
        self.init()
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        use_gpu = True if config.device == 'gpu' else False
        self.device = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")
        self.train_data_loader = None
        self.test_data_loader = None
        self.acc_na = Accuracy()
        self.acc_not_na = Accuracy()
        self.acc_total = Accuracy()

    def train(self, model):
        raise NotImplementedError

    def test(self, model):
        raise NotImplementedError

    def evaluate(self, model):
        raise NotImplementedError

    def init(self) -> None:
        self.set_seed(self.config.random_seed)

    def set_seed(self, randomseed):
        import random
        import numpy
        torch.manual_seed(randomseed)
        torch.cuda.manual_seed(randomseed)
        torch.cuda.manual_seed_all(randomseed)
        random.seed(randomseed)
        numpy.random.seed(randomseed)
        torch.backends.cudnn.deterministic = True

    def move_to_device(self, idx, batch) -> Dict[str, Any]:
        for key in batch:
            if key == 'for_relation_repr':
                for k in batch[key]:
                    if idx < 1:
                        print(f"{k}: ", list(batch[key][k].size()))
                    batch[key][k] = batch[key][k].to(self.device)
            else:
                if isinstance(batch[key], torch.Tensor):
                    if idx < 1:
                        print(f"{key}: ", list(batch[key].size()))
                    batch[key] = batch[key].to(self.device)
        return batch

    def time_to_eval(self, epoch, step, config, num_steps) -> bool:
        if config.debug_test:
            return True
        if epoch < config.epoch_start_to_eval:
            return False
        eval_criteria = (step + 1) == len(self.train_data_loader) - 1
        eval_criteria = eval_criteria or (
                config.evaluation_steps > 0
                and num_steps % config.evaluation_steps == 0
                and step % config.accumulation_steps == 0)
        return eval_criteria

    def save_ckpt(self, best_score, epoch, num_steps, model, config) -> None:
        save_path = f"{config.checkpoint_dir}/{config.exp_id}.pt"
        torch.save(model.state_dict(), save_path)
        tlt = time.localtime()
        log_str = '=' * 50 + '\n'
        time_str = f"{tlt[0]}/{tlt[1]}/{tlt[2]} {tlt[3]}:{tlt[4]}"
        log_str += f'=== {time_str}  save model at epoch {epoch} ' + '=' * (50 - len(log_str)) + '\n'
        log_str += '=' * 50 + '\n'
        log_str += "====== Best score: {:.2f}%".format(best_score*100)
        config.logging(log_str)

    def _do_train_log(self, total_loss, epoch, start_time, global_step, period=50):
        cur_loss = total_loss / period
        elapsed = time.time() - start_time
        predict_mins_per_epoch = float((len(self.train_data_loader) / period) * elapsed / 60.)
        self.config.logging(
            '| epoch {:2d} | step {:4d} |  min/{:<3d}b {:2.2f} | train loss {:5.5f} '
            '| not NA acc: {:4.2f} | NA acc: {:4.2f} | min/epoch {:.2f}'.format(
                epoch, global_step, period, elapsed/60., cur_loss,
                self.acc_not_na.get(), self.acc_na.get(), predict_mins_per_epoch))

    def visualize_data(self, epoch, step, batch, num_steps, num_samples: int = 2, split: str = 'train'):
        if step >= 3 or epoch >= 3:
            return

        def to_list(from_data, dt='int32'):
            return from_data.cpu().numpy().astype(dt).tolist()

        def print_rank2(d, key, i):
            self.config.logging(f"{key} size : {str(list(d[key].size()))}, one sample (part):")
            d = to_list(d[key][i])
            if isinstance(d[0], list):
                for item in d[:5]:
                    self.config.logging(str(item))
            else:
                self.config.logging(str(d))
        if step == 0:
            self.config.logging(f"Epoch {epoch} ============================")
        for ex_idx in range(num_samples):
            self.config.logging(f"{split}, index: {batch['indexes'][ex_idx]}")
            if 'context_idxs' in batch:
                word_ids = to_list(batch['context_idxs'][ex_idx])
                words = [self.id2word[word] for word in word_ids]
                self.config.logging(" ".join(words))
                if 'entity_span_indices' in batch:
                    entity_span_indices = to_list(batch['entity_span_indices'][ex_idx])
                    self.config.logging(f"entity_span_indices: {str(entity_span_indices)}")
                    mentions = [" ".join(words[start: end+1]) for start, end in entity_span_indices]
                    self.config.logging("  |  ".join(mentions))
                if 'input_ids' in batch:
                    new_words = []
                    input_ids = self.tokenizer.convert_ids_to_tokens(batch['input_ids'][ex_idx])
                    t2p_map = to_list(batch['t2p_map'][ex_idx])
                    t2p_map_mask = to_list(batch['t2p_map_mask'][ex_idx])
                    for word, pieces_indices, mask in zip(words, t2p_map, t2p_map_mask):
                        new_words.append(f"{word}_[{','.join(input_ids[i] for i in pieces_indices)}]_{str(mask)}")
                    self.config.logging("  ".join(new_words))
            if 'entity_cluster_ids' in batch:
                entity_cluster_ids = to_list(batch['entity_cluster_ids'][ex_idx])
                self.config.logging(f"entity_cluster_ids (coref): {str(entity_cluster_ids)}")
            if 'entity_type_ids' in batch:
                entity_type_ids = to_list(batch['entity_type_ids'][ex_idx])
                self.config.logging(f"entity_type_ids (ner types): {str(entity_type_ids)}")
            if 'dis_h_2_t' in batch:
                print_rank2(batch, 'dis_h_2_t', ex_idx)
            if 'for_relation_repr' in batch:
                for key in keys:
                    print_rank2(batch['for_relation_repr'], key, ex_idx)
            self.config.logging("\n")
        self.config.logging("\n\n")

    def _compute_acc(self, relation_label, logits) -> None:
        output = torch.argmax(logits, dim=-1)
        output = output.data.cpu().numpy()
        for i in range(output.shape[0]):
            for j in range(output.shape[1]):
                label = relation_label[i][j]
                if label < 0:
                    break
                if label == 0:
                    self.acc_na.add(output[i][j] == label)
                else:
                    self.acc_not_na.add(output[i][j] == label)
                self.acc_total.add(output[i][j] == label)

    def _clear_for_acc(self):
        self.acc_na.clear()
        self.acc_not_na.clear()
        self.acc_total.clear()


class EAATrainer(Trainer):

    def __init__(self, config, only_predict: bool = False) -> None:
        super(EAATrainer, self).__init__(config)

        if not only_predict:
            print(f'load train_data_loader: {config.train_prefix} ...')
            train_dataset = BertEncDataSet(
                data_dir=config.data_dir,
                prefix=config.train_prefix,
                pad_token_id=self.tokenizer.pad_token_id,
                config=self.config,
                is_train=True,
            )
            self.train_data_loader = DataLoader(
                dataset=train_dataset,
                batch_size=config.batch_size,
                num_workers=12,
                shuffle=True,
                collate_fn=collate_fn_bert_enc
            )
        print(f'load test_data_loader: {config.test_prefix} ...')
        test_dataset = BertEncDataSet(
            data_dir=config.data_dir,
            prefix=config.test_prefix,
            pad_token_id=self.tokenizer.pad_token_id,
            config=self.config,
            is_train=False,
        )
        self.test_data_loader = DataLoader(
            dataset=test_dataset,
            batch_size=config.test_batch_size,
            num_workers=12,
            shuffle=False,
            collate_fn=collate_fn_bert_enc
        )

    def _get_optimizer(self, bert_layers, config, model):
        parameters = [
            {"params": [], "lr": config.bert_learning_rate},
            {"params": [], "lr": config.learning_rate}
        ]
        for n, p in model.named_parameters():
            requires_grad = p.requires_grad
            param_group_idx = 0 if any(nd in n for nd in bert_layers) else 1
            if requires_grad:
                parameters[param_group_idx]["params"].append(p)
            config.logging("{:<105} ; grad: {:<6} ; size: {:<15} ; lr: {}".format(
                n, str(requires_grad), str(list(p.size())), parameters[param_group_idx]['lr']))
        if len(parameters[0]['params']) == 0:
            parameters = parameters[1:]
        if config.optimizer == 'AdamW':
            optimizer = AdamW(parameters, lr=config.learning_rate)
        elif config.optimizer == 'Adam':
            optimizer = optim.Adam(parameters, lr=config.learning_rate)
        else:
            raise NotImplementedError("")
        return optimizer

    def train(self, model, bert_layers: list = None) -> None:
        config = self.config
        model = model.to(self.device)
        if config.device == 'gpu' and torch.cuda.is_available():
            dp_model = torch.nn.DataParallel(model)
        else:
            dp_model = model

        optimizer = self._get_optimizer(bert_layers if bert_layers else [], config, dp_model)
        loss_func = torch.nn.BCEWithLogitsLoss(reduction='none')

        total_steps = int(len(self.train_data_loader) * config.max_epoch // config.accumulation_steps)
        config.logging("Total steps: {}".format(total_steps))
        num_steps, total_loss, best_score = 0, 0.0, -1
        start_time = time.time()
        dp_model.train()
        optimizer.zero_grad()
        for epoch in trange(config.max_epoch, desc='Epoch'):
            epoch_start_time, forward_time = time.time(), 0.0
            self._clear_for_acc()
            for step, batch in tqdm(enumerate(self.train_data_loader), desc='Train-Batch', total=len(self.train_data_loader)):
                self.visualize_data(epoch, step, batch, num_steps)
                dp_model.train()
                batch = self.move_to_device(step, batch)
                before_forward = time.time()
                logits = dp_model(inputs=batch, is_train=True)  # main forward
                forward_time += (time.time() - before_forward)
                # loss = model.loss(logits, batch['relation_multi_label'], batch['relation_mask'])
                loss = loss_func(logits, batch['relation_multi_label']) * batch['relation_mask'].unsqueeze(2)
                loss = torch.sum(loss) / (self.config.relation_num * torch.sum(batch['relation_mask']))
                loss = loss / config.accumulation_steps
                loss.backward()

                if step % config.accumulation_steps == 0:
                    if config.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(dp_model.parameters(), config.max_grad_norm)
                    optimizer.step()
                    dp_model.zero_grad()
                    num_steps += 1
                    total_loss += loss.item() * config.accumulation_steps

                wandb.log({"loss": loss.item()}, step=num_steps)
                self._compute_acc(batch['relation_label'], logits)

                if num_steps % config.train_log_steps == 0:
                    self._do_train_log(total_loss, epoch, start_time, num_steps, period=config.train_log_steps)
                    total_loss, start_time = 0.0, time.time()

                if self.time_to_eval(epoch, step, config, num_steps):
                    dp_model.eval()
                    eval_start_time = time.time()
                    dev_score = self.evaluate(dp_model)
                    config.logging('| epoch {:3d} | eval_time: {:5.2f}s'.format(epoch, time.time() - eval_start_time))
                    wandb.log({"dev_ign_f1": dev_score*100}, step=num_steps)
                    config.logging("dev_ign_f1: {:.2f}%".format(dev_score*100))
                    if dev_score > best_score:
                        best_score = dev_score
                        self.save_ckpt(best_score, epoch, num_steps, model, config)

            # this epoch ends
            config.logging("Time usage of this epoch: {:.2f} secs".format(time.time() - epoch_start_time))
            config.logging("Time usage of this epoch (forward): {:.2f} secs".format(forward_time))

        config.logging("Final Best: {:.2f}".format(best_score*100))

    def evaluate(self, model, ckpt: str = None) -> float:
        config = self.config
        model.to(self.device)
        if ckpt:
            model.load_state_dict(torch.load(ckpt, map_location=self.device))
            config.logging(f"load checkpoint succeed: {ckpt} !")
        model.eval()
        test_result = []

        def fill_test_result(_ex_idx, _title, _index, _L, _label, _preds):
            rel_idx = 0
            pairs = [(h_idx, t_idx) for h_idx in range(_L) for t_idx in range(_L) if h_idx != t_idx]
            for h_idx, t_idx in pairs:
                for r in range(1, config.relation_num):
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
                rel_idx += 1

        total_recall = 0
        for step, batch in tqdm(enumerate(self.test_data_loader), desc='Test-Batch', total=len(self.test_data_loader)):
            with torch.no_grad():
                batch = self.move_to_device(step, batch)
                logits = model(inputs=batch, is_train=False)
                preds = torch.sigmoid(logits)
            preds = preds.data.cpu().numpy()
            labels = batch['labels']
            titles = batch['titles']
            indexes = batch['indexes']
            L = batch['L_vertex']
            for ex_idx in range(len(labels)):
                label = labels[ex_idx]
                index = indexes[ex_idx]
                total_recall += len(label)
                fill_test_result(ex_idx, titles[ex_idx], index, L[ex_idx], label, preds)

        if total_recall == 0:
            total_recall = 1  # for test
        config.logging(f"total_recall: {total_recall}")

        # sort by the probability, for computing AUC
        test_result.sort(key=lambda x: x[1], reverse=True)

        # ALL auc, p, r, f1
        w, pr_x, pr_y, f1_pos, all_f1 = compute_f1(test_result, total_recall, -1)
        all_auc = auc(x=pr_x, y=pr_y)
        del pr_x, pr_y
        theta = test_result[f1_pos][1]
        config.logging('ALL  : Theta {:3.4f} | F1 {:3.2f} | AUC {:3.2f}'.format(theta, all_f1 * 100, all_auc * 100))

        # Ign auc, p, r, f1
        input_theta = theta
        f1_w, pr_x, pr_y, max_f1_pos, f1 = compute_ign_f1(test_result, total_recall, input_theta)
        ign_auc = auc(x=pr_x, y=pr_y)
        prec = pr_y[max_f1_pos]
        recall = pr_x[max_f1_pos]
        del pr_x, pr_y
        config.logging(
            'Ignore ma_f1 {:3.2f} | prec {:3.2f} | recall {:3.2f} |'
            ' input_theta {:3.4f} test_result F1 {:3.2f} | AUC {:3.2f}'.format(
                f1 * 100, prec * 100, recall * 100, input_theta, f1_w * 100, ign_auc * 100))

        return f1

    def test(self, model, ckpt: str = None, input_theta: float = None, output: str = 'result.json'):
        assert input_theta is not None
        config = self.config
        model = model.to(self.device)
        model.load_state_dict(torch.load(ckpt, map_location=self.device))
        config.logging(f"load checkpoint succeed: {ckpt} !")
        model.eval()
        res = []
        for step, batch in tqdm(enumerate(self.test_data_loader), desc='Test-Batch', total=len(self.test_data_loader)):
            with torch.no_grad():
                batch = self.move_to_device(step, batch)
                logits = model(inputs=batch, is_train=False)
                preds = torch.sigmoid(logits)
            # [B, R, relation_num]
            preds = preds.data.cpu().numpy()
            titles = batch['titles']

            for ex_idx, L in enumerate(batch['L_vertex']):
                for rel_idx, (h_idx, t_idx) in enumerate(
                        [(h_idx, t_idx) for h_idx in range(L) for t_idx in range(L) if h_idx != t_idx]):
                    for r in range(1, config.relation_num):
                        prob = float(preds[ex_idx, rel_idx, r])
                        if prob >= input_theta:
                            res.append({
                                'title': titles[ex_idx],
                                'h_idx': h_idx,
                                't_idx': t_idx,
                                'r': self.id2rel[r],
                            })

        with open(output, "w") as fh:
            json.dump(res, fh)
        return 0.0
