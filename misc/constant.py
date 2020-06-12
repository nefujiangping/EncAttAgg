# This program will backup the source codes for each training process.
# The following dirs together with ./*.py and `param_file` will be copied to dir `summary_dir`/`exp_id`
dirs_to_backup = [
    'misc',
    'models',
    'trainer'
]

# Prevent from typing error in config files (*.yaml),
# e.g., non-empty string will be considered as True in python.
# The program will validate the bool properties given in *.yaml at function `misc.util.load_params()`.
# Make sure that the given `param_keys`/`param_second_order_keys` are spelled correctly.
# params['cat_nlayer'] ...
param_keys = [
    'cat_nlayer', 'use_bert_embedding', 'use_distance', 'use_overlap',
    'use_bilinear', 'write_weights', 'decode_test', 'debug_test',
    'use_lr_scheduler', 'train_on_trainanddev', 'train_hypernym_filter',
    'test_hypernym_filter', 'lowercase', 'gcn_use_gate', 'gcn_residual',
    'use_neg_sample', 'filter_by_entity_type', 'parallel_forward'
]
# params['mutual_attender']['shared'] ...
param_second_order_keys = [
    'mutual_attender.shared',
    'drop_word.use_drop',
    'avg_params.use_avg_params'
]


# ===============================================================
# =========== some constants used in this program ===============
# ===============================================================
pooling_style = {
    'mean': 'mean',
    'max': 'max',
    'mean-max': 'mean-max',
    'attn': 'attn'
}

integration_attender = {
    'NONE': "NONE",
    'ML-MA': 'ML-MA',  # MultiLayer Multi-Head Attention
}

mutual_attender = {
    'NONE': "NONE",
    'ML-MMA': 'ML-MMA',  # MultiLayer Mutual Multi-Head Attention
}

which_model = {
    'BiLSTM-M': 'BiLSTM-M',
    'BRAN-M': 'BRAN-M',
    'EncAgg': 'EncAgg',
    'EncAttAgg': 'EncAttAgg',
    'GCNN': 'GCNN'
}

datasets = {
    'DocRED': 'DocRED',
    'CDR': 'CDR'
}

