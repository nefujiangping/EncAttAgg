from typing import List
import yaml
import os


class Config(object):

    def __init__(self, cmd_args, yaml_path):
        config_file_params = yaml.load(open(yaml_path, 'r'), yaml.FullLoader)
        # print(cmd_args)
        self.params = []
        # params from config file
        for key, value in config_file_params.items():
            self.__setattr__(key, value)
            self.params.append(key)
        # params from command line
        overwritten = []
        for key, value in vars(cmd_args).items():
            if key in config_file_params:
                if value is not None and value != self.__getattribute__(key):
                    overwritten.append([key, self.__getattribute__(key), value])
                    self.__setattr__(key, value)
            else:
                self.params.append(key)
                self.__setattr__(key, value)
        for _dir in [self.checkpoint_dir, self.log_dir, self.summary_dir]:
            os.makedirs(_dir, exist_ok=True)
        # logging overwritten params
        for idx, (_key, _from, _to) in enumerate(overwritten):
            if idx == 0:
                self.logging('='*36 + " overwritten params " + '='*36)
                self.logging("{:<23}: {:<20} ==> {:<20}".format('param', f'{os.path.basename(yaml_path)}', 'cmd_args'))
            self.logging("{:<23}: {:<20} ==> {:<20}".format(_key, str(_from), str(_to)))
            if idx == len(overwritten) - 1:
                self.logging("="*92 + '\n')

    def print_params(self):
        # logging final params
        self.logging(self.params_str())

    def format_params(self) -> List[str]:
        config_lines = list()
        for attr in self.params:
            try:
                value = self.__getattribute__(attr)
                if value is not None:
                    config_lines.append("{:<23}: {}\n".format(attr, str(value)))
            except AttributeError:
                continue
        return config_lines

    def params_str(self):
        config_lines = list()
        config_lines.append(f"{'=' * 18} Parameters {'=' * 18}\n")  # head line
        config_lines.extend(self.format_params())  # contents
        config_lines.append(f"{'=' * 48}\n\n")  # end line
        return "".join(config_lines)

    def logging(self, s, print_=True, log_=True):
        if print_:
            print(s)
        if log_:
            with open(os.path.join(os.path.join(self.log_dir, self.exp_id)), 'a+', encoding='utf8') as f_log:
                f_log.write(s + '\n')
