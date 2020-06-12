from trainer.CDRSimpleTrainer_Softmax import CDRSimpleTrainer_Softmax
from trainer.CDRFusionTrainer_Softmax import CDRFusionTrainer_Softmax
from trainer.DocREDSimpleTrainer import SimpleTrainer
from trainer.DocREDFusionTrainer import FusionTrainer
from models.models import RelationExtraction, GCNN
import argparse
import sys
import os
from misc import constant as CST
from misc.util import load_params
from trainer.GCNN.CDR_GCNN_Trainer import CDRGCNNTrainer_Softmax
from trainer.GCNN.DocRED_GCNN_Trainer import DocREDGCNNTrainer

seeds = [1234, 5678, 2333, 8765, 4321]
# import sklearn first, then allennlp toolkits, or Error occurs.
# In this case, `import FusionTrainer` first (because sklearn is imported in it), then `import RelationExtraction`
parser = argparse.ArgumentParser()
parser.add_argument('--param_file', type=str, required=True,
                    help='params file path')
parser.add_argument('--device', type=str, default='gpu', choices=['gpu', 'cpu'], help='train on `cpu` or `gpu`')
parser.add_argument('--gpu', type=str, default=None, help='For `device`=gpu, number(s) of gpu to use.')
parser.add_argument('--exp_id', type=str, required=True,
                    help='ID of this experiment. Make sure that the given `exp_id` not exists, '
                    'or it will rewrite everything.')
parser.add_argument('--note', type=str, default=None,
                    help='Note of this experiment.')
parser.add_argument('--randseed_no', type=int, default=0, choices=list(range(5)),
                    help='There are 5 seeds: [0: 1234, 1: 5678, 2: 2333, 3: 8765, 4: 4321]')
parser.add_argument('--debug', action='store_true', help='debug on cpu')
parser.add_argument('--not_backup', action='store_true',
                    help='Do NOT backup codes. (The program will backup the code by default.)')
parser.add_argument('--debug_test', action='store_true',
                    help='directly run test to debug whether test process works well (skip training).')
parser.add_argument('--decode_test', action='store_true',
                    help='For DocRED dataset, use the best checkpoint '
                    '(the checkpoint that performs best on Ign F1-score metric)'
                    'to decode test set after training ends.')
parser.add_argument('--check_get_batches', action='store_true',
                    help='Check whether trainer.get_train_batches/get_test_batches works fine.')
args = parser.parse_args()
if args.device == 'gpu':
    assert args.gpu is not None, 'Now training on gpu, please give gpu(s).'
    for n in args.gpu.split(','):
        assert n.isdigit(), 'gpu(s) should be numbers split by `,`'
params = load_params(args.param_file, args)
params['randomseed'] = seeds[params['randseed_no']]
params['write_weights'] = False  # For test.py

if params['gpu'] is not None:
    os.environ['CUDA_VISIBLE_DEVICES'] = params['gpu']

model_pattern = RelationExtraction
# train DocRED
if params['dataset'] == CST.datasets['DocRED']:
    if params['which_model'] in [CST.which_model['BRAN-M'],
                                 CST.which_model['EncAgg'],
                                 CST.which_model['EncAttAgg']]:
        trainer = FusionTrainer(params)
    elif params['which_model'] == CST.which_model['BiLSTM-M']:
        trainer = SimpleTrainer(params)
    elif params['which_model'] == CST.which_model['GCNN']:
        trainer = DocREDGCNNTrainer(params)
        model_pattern = GCNN
# train CDR
elif params['dataset'] == CST.datasets['CDR']:
    if params['which_model'] in [CST.which_model['BRAN-M'],
                                 CST.which_model['EncAgg'],
                                 CST.which_model['EncAttAgg']]:
        trainer = CDRFusionTrainer_Softmax(params)
    elif params['which_model'] == CST.which_model['BiLSTM-M']:
        trainer = CDRSimpleTrainer_Softmax(params)
    elif params['which_model'] == CST.which_model['GCNN']:
        trainer = CDRGCNNTrainer_Softmax(params)
        model_pattern = GCNN

# logging seed
trainer.logging(f" randseed NO.: {params['randseed_no']}\n"
                f"  random seed: {params['randomseed']}\n")

# logging command
args_list = sys.argv[:]
cmd = "python " + " ".join(args_list)
if params['gpu'] is not None:
    cmd = f"CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']} {cmd}"
trainer.logging(cmd)
params['cmd'] = cmd

# logging note of this experiment
if params['note']:
    trainer.logging(params['note'])

# logging config of this experiment
trainer.logging(trainer.get_config_str())
trainer.load_train_data()
trainer.load_test_data()

if params['debug']:
    trainer.set_on_cpu()
    if params['check_get_batches']:
        trainer.check_get_batches()
    else:
        trainer.train(model_pattern)
else:
    try:
        if params['check_get_batches']:
            trainer.check_get_batches()
        else:
            trainer.train(model_pattern)
    except BaseException as e:
        trainer.logging(str(type(e)))
        trainer.logging(f"{e.with_traceback(None)}")

if params['decode_test']:
    trainer.test_prefix = "dev_test"
    trainer.load_test_data()
    trainer.testall(model_pattern, trainer.input_theta_of_best_epoch)

