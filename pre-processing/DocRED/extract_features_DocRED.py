# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Extract pre-computed feature vectors from BERT."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import codecs
import json
import re

import modeling
import tokenization
import tensorflow as tf
import numpy as np
import h5py

flags = tf.flags

FLAGS = flags.FLAGS

flags.DEFINE_string("input_file", None, "")

flags.DEFINE_string("data_dir", None, "")

flags.DEFINE_string("output_file", None, "")

flags.DEFINE_string("layers", "-1,-2,-3,-4", "")

flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_bool(
    "do_lower_case", False,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer("batch_size", 32, "Batch size for predictions.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

flags.DEFINE_string("master", None,
                    "If using a TPU, the address of the master.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")

flags.DEFINE_bool(
    "use_one_hot_embeddings", False,
    "If True, tf.one_hot will be used for embedding lookups, otherwise "
    "tf.nn.embedding_lookup will be used. On TPUs, this should be True "
    "since it is much faster.")


class InputExample(object):

    def __init__(self, unique_id, text_a, text_b):
        self.unique_id = unique_id
        self.text_a = text_a
        self.text_b = text_b


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, unique_id, tokens, input_ids, input_mask, input_type_ids, orig_tokens, orig_to_tok_map):
        self.unique_id = unique_id
        self.tokens = tokens
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.input_type_ids = input_type_ids
        self.orig_tokens = orig_tokens
        self.orig_to_tok_map = orig_to_tok_map


def input_fn_builder(features, seq_length):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    all_unique_ids = []
    all_input_ids = []
    all_input_mask = []
    all_input_type_ids = []

    for feature in features:
        all_unique_ids.append(feature.unique_id)
        all_input_ids.append(feature.input_ids)
        all_input_mask.append(feature.input_mask)
        all_input_type_ids.append(feature.input_type_ids)

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]

        num_examples = len(features)

        # This is for demo purposes and does NOT scale to large data sets. We do
        # not use Dataset.from_generator() because that uses tf.py_func which is
        # not TPU compatible. The right way to load data is with TFRecordReader.
        d = tf.data.Dataset.from_tensor_slices({
            "unique_ids":
                tf.constant(all_unique_ids, shape=[num_examples], dtype=tf.int32),
            "input_ids":
                tf.constant(
                    all_input_ids, shape=[num_examples, seq_length],
                    dtype=tf.int32),
            "input_mask":
                tf.constant(
                    all_input_mask,
                    shape=[num_examples, seq_length],
                    dtype=tf.int32),
            "input_type_ids":
                tf.constant(
                    all_input_type_ids,
                    shape=[num_examples, seq_length],
                    dtype=tf.int32),
        })

        d = d.batch(batch_size=batch_size, drop_remainder=False)
        return d

    return input_fn


def model_fn_builder(bert_config, init_checkpoint, layer_indexes, use_tpu,
                     use_one_hot_embeddings):
    """Returns `model_fn` closure for TPUEstimator."""

    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""

        unique_ids = features["unique_ids"]
        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        input_type_ids = features["input_type_ids"]

        model = modeling.BertModel(
            config=bert_config,
            is_training=False,
            input_ids=input_ids,
            input_mask=input_mask,
            token_type_ids=input_type_ids,
            use_one_hot_embeddings=use_one_hot_embeddings)

        if mode != tf.estimator.ModeKeys.PREDICT:
            raise ValueError("Only PREDICT modes are supported: %s" % (mode))

        tvars = tf.trainable_variables()
        scaffold_fn = None
        (assignment_map,
         initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(
            tvars, init_checkpoint)
        if use_tpu:

            def tpu_scaffold():
                tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
                return tf.train.Scaffold()

            scaffold_fn = tpu_scaffold
        else:
            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        tf.logging.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                            init_string)

        all_layers = model.get_all_encoder_layers()

        predictions = {
            "unique_id": unique_ids,
        }

        for (i, layer_index) in enumerate(layer_indexes):
            predictions["layer_output_%d" % i] = all_layers[layer_index]

        output_spec = tf.contrib.tpu.TPUEstimatorSpec(
            mode=mode, predictions=predictions, scaffold_fn=scaffold_fn)
        return output_spec

    return model_fn


def convert_examples_to_features(examples, seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    features = []
    for (ex_index, example) in enumerate(examples):
        # tokens_a = tokenizer.tokenize(example.text_a)
        tokens_a, orig_to_tok_map = [], []
        for iii, orig_token in enumerate(example.text_a.split(' ')):
            toks_to_extend = tokenizer.tokenize(orig_token)
            if len(toks_to_extend) == 0:
                toks_to_extend = ['[UNK]']
            tokens_a.extend(toks_to_extend)
            to_append = len(tokens_a) + 1  # for [CLS]
            orig_to_tok_map.append(to_append)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)

        if tokens_b:
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > seq_length - 2:
                tokens_a = tokens_a[0:(seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0     0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = []
        input_type_ids = []
        tokens.append("[CLS]")
        input_type_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            input_type_ids.append(0)
        tokens.append("[SEP]")
        input_type_ids.append(0)

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                input_type_ids.append(1)
            tokens.append("[SEP]")
            input_type_ids.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < seq_length:
            input_ids.append(0)
            input_mask.append(0)
            input_type_ids.append(0)

        assert len(input_ids) == seq_length
        assert len(input_mask) == seq_length
        assert len(input_type_ids) == seq_length

        if ex_index < 5:
            tf.logging.info("*** Example ***")
            tf.logging.info("unique_id: %s" % (example.unique_id))
            tf.logging.info("tokens: %s" % " ".join(
                [tokenization.printable_text(x) for x in tokens]))
            tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            tf.logging.info(
                "input_type_ids: %s" % " ".join([str(x) for x in input_type_ids]))

        features.append(
            InputFeatures(
                unique_id=example.unique_id,
                tokens=tokens,
                input_ids=input_ids,
                input_mask=input_mask,
                input_type_ids=input_type_ids,
                orig_tokens=example.text_a.split(' '),
                orig_to_tok_map=orig_to_tok_map))
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def read_examples(input_file):
    """Read a list of `InputExample`s from an input file."""
    examples = []
    unique_id = 0
    with tf.gfile.GFile(input_file, "r") as reader:
        while True:
            line = tokenization.convert_to_unicode(reader.readline())
            if not line:
                break
            line = line.strip()
            text_a = None
            text_b = None
            m = re.match(r"^(.*) \|\|\| (.*)$", line)
            if m is None:
                text_a = line
            else:
                text_a = m.group(1)
                text_b = m.group(2)
            examples.append(
                InputExample(unique_id=unique_id, text_a=text_a, text_b=text_b))
            unique_id += 1
    return examples


def json2text(data_dir, split):
    counter = 0
    with open('%s/dev_%s_text.txt' % (data_dir, split), 'w', encoding='utf-8') as writer:
        for i, ins in enumerate(json.load(open('%s/dev_%s.json' % (data_dir, split), 'r', encoding='utf8'))):
            toks = []
            for sent in ins['sents']:
                toks.extend(sent)
            if len(toks) > 450:
                counter += 1
            s = ' '.join(toks)
            s1 = '   '  # 3 space
            s2 = '   '  # space + chinese_space + space
            # here s1 and s2 are not the same
            s = s.replace(s1, s2)
            s = s.replace('  ', ' ')  # replace 2 space with 1 space
            if i > 0:
                writer.write('\n' + s)
            else:
                writer.write(s)
    print('%s, greater than 450: %d' % (split, counter))
    # dev 10, train 15, test 7


def post_process(data_dir, split, num_example, padding=True, bert_embedd_dim=768):
    orig_text = codecs.open('%s/dev_%s_text.txt' % (data_dir, split), 'r', encoding='utf8').readlines()
    bert_text = codecs.open('%s/dev_%s_bert_text.json' % (data_dir, split), 'r', encoding='utf8').readlines()
    with h5py.File('%s/dev_%s_text.h5' % (data_dir, split), 'r') as in_h5, \
            h5py.File('%s/%d_dev_%s.h5' % (data_dir, bert_embedd_dim, split), 'w') as out_h5:
        for idx in range(num_example):
            bert_ex = json.loads(bert_text[idx].strip())
            bert_tokens = bert_ex['tokens'].split(' ')
            orig_tokens_len = len(orig_text[idx].strip().split(' '))
            ex_feature = in_h5[str(idx) + 'feature']
            orig_to_tok_map = bert_ex['orig_to_tok_map']
            ex_len_counter = 0
            new_feature = []
            for i in range(len(orig_to_tok_map)):
                _start = 1 if i == 0 else orig_to_tok_map[i-1]
                _end = orig_to_tok_map[i]
                _word_pieces = bert_tokens[_start: _end]
                if len(_word_pieces) == 0:
                    break
                if _word_pieces[-1] == '[SEP]':
                    _word_pieces = _word_pieces[:-1]
                    _end -= 1
                    if len(_word_pieces) == 0:
                        break
                # if _start >= 505:
                # print('{:<20} | {}[{}, {}]'.format(orig_tokens[i], ' '.join(_word_pieces), _start, _end))
                # for single-piece-token use vector itself, for multi-piece-token use average pooling over pieces.
                align_feature = ex_feature[_start, :] if _start == _end else np.mean(ex_feature[_start: _end, :], axis=0)
                new_feature.append(align_feature)
                ex_len_counter += 1
            assert ex_len_counter <= orig_tokens_len, 'ERROR'
            if ex_len_counter < orig_tokens_len and padding:
                pad = [0.]*bert_embedd_dim
                for _ in range(orig_tokens_len-ex_len_counter):
                    new_feature.append(pad)
            new_feature = np.array(new_feature).astype(np.float32)
            if padding:
                assert new_feature.shape == (orig_tokens_len, bert_embedd_dim), 'ERROR'
            out_h5.create_dataset(name=str(idx),
                                  shape=new_feature.shape,
                                  dtype='float32',
                                  data=new_feature)


def process_main(split):
    tf.logging.set_verbosity(tf.logging.INFO)

    layer_indexes = [int(x) for x in FLAGS.layers.split(",")]
    assert len(layer_indexes) == 1 and layer_indexes[0] == -1, 'This support `-1` layer only.'

    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

    tokenizer = tokenization.FullTokenizer(
        vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

    is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
    run_config = tf.contrib.tpu.RunConfig(
        master=FLAGS.master,
        tpu_config=tf.contrib.tpu.TPUConfig(
            num_shards=FLAGS.num_tpu_cores,
            per_host_input_for_training=is_per_host))

    data_dir = FLAGS.data_dir

    examples = read_examples("%s/dev_%s_text.txt" % (data_dir, split))

    features = convert_examples_to_features(
        examples=examples, seq_length=FLAGS.max_seq_length, tokenizer=tokenizer)

    unique_id_to_feature = {}
    for feature in features:
        unique_id_to_feature[feature.unique_id] = feature

    num_dict = {'train': 3053, 'dev': 1000, 'test': 1000}

    NUM = num_dict[split]
    with open('%s/dev_%s_bert_text.json' % (data_dir, split), 'w', encoding='utf8') as writer:
        for idx in range(NUM):
            bert_tokens = unique_id_to_feature[idx].tokens
            out_ex = {
                'tokens': ' '.join(bert_tokens),
                'orig_to_tok_map': unique_id_to_feature[idx].orig_to_tok_map
            }
            if idx > 0:
                writer.write('\n' + json.dumps(out_ex, ensure_ascii=False))
            else:
                writer.write(json.dumps(out_ex, ensure_ascii=False))

    model_fn = model_fn_builder(
        bert_config=bert_config,
        init_checkpoint=FLAGS.init_checkpoint,
        layer_indexes=layer_indexes,
        use_tpu=FLAGS.use_tpu,
        use_one_hot_embeddings=FLAGS.use_one_hot_embeddings)

    # If TPU is not available, this will fall back to normal Estimator on CPU
    # or GPU.
    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=FLAGS.use_tpu,
        model_fn=model_fn,
        config=run_config,
        predict_batch_size=FLAGS.batch_size)

    input_fn = input_fn_builder(
        features=features, seq_length=FLAGS.max_seq_length)

    with h5py.File("%s/dev_%s_text.h5" % (data_dir, split), 'w') as writer:
        for result in estimator.predict(input_fn, yield_single_examples=True):
            unique_id = int(result["unique_id"])
            feature = unique_id_to_feature[unique_id]
            sent_repr = []
            for (i, token) in enumerate(feature.tokens):
                layer_output = result["layer_output_0"]
                sent_repr.append([
                    round(float(x), 6) for x in layer_output[i:(i + 1)].flat
                ])
            bert_tokens = [t.encode('utf-8') for t in feature.tokens]
            orig_tokens = [t.encode('utf-8') for t in feature.orig_tokens]
            sent_repr = np.array(sent_repr)
            # print(feature.tokens)
            token_max_len = max([len(w) for w in feature.orig_tokens])
            # print(feature.orig_tokens)
            bert_token_max_len = max([len(w) for w in feature.tokens])
            writer.create_dataset(name=str(unique_id) + 'orig_tokens',
                                  shape=(len(orig_tokens), ),
                                  dtype='S{}'.format(token_max_len),
                                  data=orig_tokens)
            writer.create_dataset(name=str(unique_id) + 'bert_tokens',
                                  shape=(len(bert_tokens), ),
                                  dtype='S{}'.format(bert_token_max_len),
                                  data=bert_tokens)
            writer.create_dataset(name=str(unique_id) + 'feature',
                                  shape=sent_repr.shape,
                                  dtype='float32',
                                  data=sent_repr)

    post_process(data_dir, split, NUM)


def main(_):
    for split in ['test', 'dev', 'train']:
        json2text(FLAGS.data_dir, split)
        process_main(split)


if __name__ == "__main__":
    flags.mark_flag_as_required("data_dir")
    flags.mark_flag_as_required("vocab_file")
    flags.mark_flag_as_required("bert_config_file")
    flags.mark_flag_as_required("init_checkpoint")
    tf.app.run()


