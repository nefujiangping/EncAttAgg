import os
import yaml
from tensorboardX import SummaryWriter
from typing import Dict
from misc.constant import param_keys, param_second_order_keys


class TensorboardWriter:
    def __init__(self,
                 model_name: str,
                 summary_dir: str = 'summary',
                 summary_interval: int = 10) -> None:
        if model_name is not None:
            train_dir = os.path.join(summary_dir, model_name, "train")
            val_dir = os.path.join(summary_dir, model_name, "validation")
            for dir in [summary_dir, f'{summary_dir}/{model_name}', train_dir, val_dir]:
                if not os.path.exists(dir):
                    os.mkdir(dir)
            self._train_log = SummaryWriter(train_dir)
            self._validation_log = SummaryWriter(val_dir)
        else:
            self._train_log = self._validation_log = None
        self._counter = 0
        self._summary_interval = summary_interval

    def log_train_loss(self, val: float, step: int):
        self._train_log.add_scalar('train_loss', val, step)
        self._counter += 1

    def log_train_not_na_acc(self, val: float, step: int):
        self._train_log.add_scalar('not_na_acc', val, step)

    def log_train_lr(self, val: float, epoch: int):
        self._train_log.add_scalar('learning_rate', val, epoch)

    def log_text(self, text_string: str, epoch: int):
        self._train_log.add_text('log', text_string, epoch)

    def log_val_coref_f1(self, val: float, epoch: int):
        self._validation_log.add_scalar('val_coref_f1', val, epoch)

    def log_val_noncoref_f1(self, val: float, epoch: int):
        self._validation_log.add_scalar('val_non-coref_f1', val, epoch)

    def log_val_short_f1(self, val: float, epoch: int):
        self._validation_log.add_scalar('val_short_f1', val, epoch)

    def log_val_long_f1(self, val: float, epoch: int):
        self._validation_log.add_scalar('val_long_f1', val, epoch)

    def log_val_f1(self, val: float, epoch: int):
        self._validation_log.add_scalar('val_f1', val, epoch)

    def log_val_precision(self, val: float, epoch: int):
        self._validation_log.add_scalar('val_precision', val, epoch)

    def log_val_recall(self, val: float, epoch: int):
        self._validation_log.add_scalar('val_recall', val, epoch)

    def log_val_auc(self, val: float, epoch: int):
        self._validation_log.add_scalar('val_auc', val, epoch)

    def log_val_all_f1(self, val: float, epoch: int):
        self._validation_log.add_scalar('val_all_f1', val, epoch)

    def log_val_all_auc(self, val: float, epoch: int):
        self._validation_log.add_scalar('val_all_auc', val, epoch)

    def log_inference_theta(self, val: float, epoch: int):
        self._validation_log.add_scalar('val_theta', val, epoch)

    def close(self) -> None:
        """
        Calls the ``close`` method of the ``SummaryWriter`` s which makes sure that pending
        scalars are flushed to disk and the tensorboard event files are closed properly.
        """
        if self._train_log is not None:
            self._train_log.close()
        if self._validation_log is not None:
            self._validation_log.close()


def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


def load_params(params_file: str, args) -> Dict:
    def assert_bool_type(param, key):
        assert isinstance(param, bool), f"params.{key} Error, please check config file: {params_file}"
    params = yaml.load(open(params_file).read(), yaml.CLoader)
    for key, value in args.__dict__.items():
        assert key not in params, f"argparse and *.yaml have a same param: {key}"
        params[key] = value
    # params['nlayer'] ...
    for key in param_keys:
        if key in params:
            assert_bool_type(params[key], key)
    # params['mutual_attender']['shared'] ...
    for key in param_second_order_keys:
        pieces = key.split('.')
        assert len(pieces) == 2
        if pieces[0] in params and pieces[1] in params[pieces[0]]:
            assert_bool_type(params[pieces[0]][pieces[1]], key)
    return params

