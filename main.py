import argparse
from misc.param import Config
from models.models import RelationExtraction
from trainer.training import EAATrainer
from misc.util import boolean_string
import wandb


def set_wandb(project, wandb_config_path='wandb/wandb_config.yaml'):
    import os
    import yaml
    wandb_config = yaml.load(open(wandb_config_path, 'r'), yaml.FullLoader)
    for param in wandb_config:
        os.environ[param] = wandb_config[param]
    wandb.init(project=project)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--param_file', type=str, required=True, help='params file path')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'dev', 'test'], help='train/dev/test')
    parser.add_argument('--device', type=str, default='gpu', choices=['gpu', 'cpu'], help='train on `cpu` or `gpu`')
    parser.add_argument('--gpu', type=str, default=None, help='For `device`=gpu, number(s) of gpu to use.')
    parser.add_argument('--exp_id', type=str, required=True,
                        help='ID of this experiment. Make sure that the given `exp_id` not exists, '
                             'or it will rewrite everything.')
    parser.add_argument('--note', type=str, default=None, help='Note of this experiment.')
    parser.add_argument('--randseed_no', type=int, default=0, choices=list(range(5)),
                        help='There are 5 seeds: [0: 1234, 1: 5678, 2: 2333, 3: 8765, 4: 4321]')
    parser.add_argument('--not_backup', action='store_true',
                        help='Do NOT backup codes. (The program will backup the code by default.)')
    parser.add_argument('--debug_test', action='store_true',
                        help='directly run test to debug whether test process works well (skip training).')

    parser.add_argument('--num_hidden_layers', type=int, default=12)
    parser.add_argument('--model_name', type=str, default='bert-base-cased')
    parser.add_argument('--transformer_type', type=str, default='bert')
    parser.add_argument('--test_prefix', type=str, default='dev_dev')

    # If is None, the use the value in *.yaml config file.
    parser.add_argument('--freeze_embed', type=boolean_string, default=None)
    parser.add_argument('--max_epoch', type=int, default=None)
    parser.add_argument('--epoch_start_to_eval', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=None)

    parser.add_argument('--input_theta', type=float, default=None, help="threshold for testing")
    cmd_args = parser.parse_args()

    if cmd_args.mode == 'test':
        assert cmd_args.input_theta and cmd_args.input_theta > 0.0, \
            "test mode, please give `input_theta` (which is obtained from dev set during training)"

    seeds = [1234, 5678, 2333, 8765, 4321]
    cmd_args.random_seed = seeds[cmd_args.randseed_no]

    config = Config(cmd_args, cmd_args.param_file)
    ckpt = f"{config.checkpoint_dir}/{config.exp_id}.pt"
    if cmd_args.mode == 'train':
        config.print_params()
        trainer = EAATrainer(config=config)
        model = RelationExtraction(config=config)
        bert_layers = ["word_embedding.doc_embed.document_transformer_encoder"]
        trainer.train(model, bert_layers=bert_layers)
    elif cmd_args.mode == 'dev':
        config.test_prefix = 'dev_dev'
        config.print_params()
        config.logging("run dev:")
        tester = EAATrainer(config=config, only_predict=True)
        model = RelationExtraction(config=config)
        tester.evaluate(model, ckpt=ckpt)
    elif cmd_args.mode == 'test':
        config.test_prefix = 'dev_test'
        config.print_params()
        config.logging("run test:")
        tester = EAATrainer(config=config, only_predict=True)
        model = RelationExtraction(config=config)
        tester.test(model, ckpt=ckpt, input_theta=cmd_args.input_theta, output='result.json')
    else:
        raise NotImplementedError(f"excepted mode is: train/dev/test, given: {cmd_args.mode} !")


if __name__ == '__main__':
    set_wandb(project="enc_att_agg")
    main()
