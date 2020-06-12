import json
import h5py
import numpy as np
import codecs


def three_samples():
    lines = open('data/test_inp.txt', encoding='utf8').readlines()
    lines = [line.strip() for line in lines]
    with h5py.File('data/output4.h5', 'r') as fin:
        for idx in range(len(lines)):
            orig_tokens = [t.decode('utf8') for t in fin[str(idx) + 'orig_tokens']]
            bert_tokens = [t.decode('utf8') for t in fin[str(idx) + 'bert_tokens']]
            feature = fin[str(idx) + 'feature']
            print('file', len(lines[idx].split(' ')), lines[idx])
            print('orig', len(orig_tokens), ' '.join(orig_tokens))
            print('bert', len(bert_tokens), ' '.join(bert_tokens))
            print(feature.shape)
            print('='*100)


def json2text():
    for split in ['dev', 'train', 'test']:
        counter = 0
        with open('data/dev_%s_text.txt' % split, 'w', encoding='utf-8') as writer:
            for i, ins in enumerate(json.load(open('prepro_data/dev_%s.json' % split, 'r', encoding='utf8'))):
                toks = []
                for sent in ins['sents']:
                    toks.extend(sent)
                if len(toks) > 450:
                    counter += 1
                if i > 0:
                    writer.write('\n' + ' '.join(toks))
                else:
                    writer.write(' '.join(toks))
        print('%s, greater than 450: %d' % (split, counter))
        # dev 10, train 15, test 7


num_dict = {'train': 1000, 'dev': 500}
in_data_dir = '/home/jp/workspace2/datasets/CDR'
out_embedding_dir = '/home2/public/jp/CDR_bert_features'


def real_post_process(split, bert_embedd_dim, padding=True):
    orig_text = codecs.open(f'{in_data_dir}/dev_{split}_text.txt', 'r', encoding='utf8').readlines()
    bert_text = codecs.open(f'{in_data_dir}/dev_{split}_bert_text.json', 'r', encoding='utf8').readlines()
    with h5py.File(f'{out_embedding_dir}/768_dev_{split}_text.h5', 'r') as in_h5, \
            h5py.File(f'{out_embedding_dir}/768_dev_{split}.h5', 'w') as out_h5:
        for idx in range(num_dict[split]):
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


for split in ['dev', 'train']:
    real_post_process(split, 768)

# split = 'train'
# with h5py.File(f'{out_embedding_dir}/768_dev_{split}_text.h5', 'r') as in_h5:
#     sent990 = np.array(in_h5[str(990)+'feature'])
#     print(sent990.shape)






