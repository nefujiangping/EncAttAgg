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


def post_process(split='dev'):
    num_dict = {'train': 3053, 'dev': 1000, 'test': 1000}
    NUM = num_dict[split]
    bert_text = codecs.open('/home/jp/workspace2/DocRED/code/prepro_data/dev_%s_bert_text.txt' % split, 'r', encoding='utf8').readlines()
    orig_text = codecs.open('/home/jp/workspace2/DocRED/code/prepro_data/dev_%s_text.txt' % split, 'r', encoding='utf8').readlines()
    counter = 0
    with h5py.File('/home2/public/jp/DocRED/dev_%s_text.h5' % split, 'r') as in_h5, \
            h5py.File('/home2/public/jp/DocRED/dev_%s.h5' % split, 'w') as out_h5:
        for idx in range(NUM):
            orig_tokens = orig_text[idx].strip().split(' ')
            bert_tokens = bert_text[idx].strip().split(' ')
            # orig_tokens = []
            # for t in in_h5[str(idx) + 'orig_tokens']:
            #     try:
            #         orig_tokens.append(t.decode('utf8'))
            #     except UnicodeDecodeError as ude:
            #         print('orig_tokens', t)
            # print(len(orig_tokens))
            # print(' '.join(orig_tokens))
            # bert_tokens = []
            # for t in in_h5[str(idx) + 'bert_tokens']:
            #     try:
            #         bert_tokens.append(t.decode('utf8'))
            #     except UnicodeDecodeError as ude:
            #         print('bert_tokens', t)
            # print(' '.join(bert_tokens))
            feature = in_h5[str(idx) + 'feature']
            new_feature = []
            j = 1
            bert_end = False
            for orig_token in orig_tokens:
                _start = j
                if bert_tokens[j] == '[SEP]':
                    break
                current_tok = bert_tokens[j]
                # print('{:<25} | {:>9}, {:>9}, {:>9}, {:>9}'.format(current_tok, feature[j][0], feature[j][1], feature[j][2], feature[j][3]))
                while current_tok != orig_token:
                    j += 1
                    if bert_tokens[j] == '[SEP]':
                        bert_end = True
                        break
                    current_tok += bert_tokens[j].replace('##', '')
                    # print('{:<25} | {:>9}, {:>9}, {:>9}, {:>9}'.format(current_tok, feature[j][0], feature[j][1], feature[j][2], feature[j][3]))
                if bert_end:
                    break
                _end = j
                if _start == _end:
                    new_feature.append(feature[_start, :])
                else:
                    new_feature.append(np.mean(feature[_start:_end+1, :], axis=0))
                # print('{:<25} | {:>9}, {:>9}, {:>9}, {:>9}'.format(orig_token, new_feature[-1][0], new_feature[-1][1], new_feature[-1][2], new_feature[-1][3]))
                j += 1
            new_feature = np.array(new_feature)
            out_h5.create_dataset(name=str(idx),
                                  shape=new_feature.shape,
                                  dtype='float32',
                                  data=new_feature)
            # assert new_feature.shape == (len(orig_tokens), 768), 'ERROR'
            if new_feature.shape != (len(orig_tokens), 768):
                counter += 1
                print('idx: ', idx)
                print('orig_tokens len', len(orig_tokens))
                print('bert_tokens len', len(bert_tokens))
                print(' '.join(bert_tokens))
    print('%s : %s' % (split, counter))


def validate(split='dev'):
    num_dict = {'train': 3053, 'dev': 1000, 'test': 1000}
    NUM = num_dict[split]

    # obtained by whole sentence
    bert_text_true = codecs.open('prepro_data/dev_%s_bert_text.txt' % split, 'r', encoding='utf8').readlines()
    # orig_text_true = codecs.open('prepro_data/dev_%s_text.txt' % split, 'r', encoding='utf8').readlines()

    # obtained by each token
    bert_text = codecs.open('prepro_data/dev_%s_bert_text.json' % split, 'r', encoding='utf8').readlines()
    # orig_text = codecs.open('prepro_data/dev_%s_text.json' % split, 'r', encoding='utf8').readlines()
    counter = 0
    # with h5py.File('/home2/public/jp/DocRED/dev_%s_text.h5' % split, 'r') as in_h5:
    for idx in range(NUM):
        # orig_tokens_true = orig_text_true[idx].strip().split(' ')
        bert_tokens_true = bert_text_true[idx].strip().split(' ')

        # orig_tokens = json.loads(orig_text[idx].strip(), encoding='utf8')['tokens'].split(' ')
        bert_tokens = json.loads(bert_text[idx].strip(), encoding='utf8')['tokens'].split(' ')

        # assert len(orig_tokens_true) == len(orig_tokens), 'ORIG len ERROR %d' % idx
        # assert orig_tokens_true == orig_tokens, 'ORIG token ERROR %d' % idx

        assert len(bert_tokens_true) == len(bert_tokens), 'BERT len ERROR %d' % idx
        assert bert_tokens_true == bert_tokens, 'BERT token ERROR %d' % idx


# for split in ['dev']:
#     validate(split)


num_dict = {'train': 3053, 'dev': 1000, 'test': 1000, 'error': 7}
longer_than512_dict = {
    'train': [140, 513, 551, 825, 952, 1176, 1337, 1350, 1783, 1856, 1985, 2670, 2841, 2844, 3024],
    'dev': [234, 467, 526, 573, 633, 725, 836, 962],
    'test': [6, 30, 571, 611, 719, 883, 936]
}


# counter = 0
# for idx in range(num_dict[split]):
#     bert_ex = json.loads(bert_text[idx].strip())
#     orig_to_tok_map = bert_ex['orig_to_tok_map']
#     for i in range(len(orig_to_tok_map)-1):
#         if orig_to_tok_map[i] == orig_to_tok_map[i+1]:
#             print('{} {}==='.format(split, idx))
#             counter += 1
#             break
# print('{} {}\n=============================='.format(split, counter))
# counter = 0
# for idx in range(num_dict[split]):
#     bert_ex = json.loads(bert_text[idx].strip())
#     bert_tokens = bert_ex['tokens'].split(' ')
#     if u'\u00a0' in bert_tokens:
#         print('{} {}==='.format(split, idx))
#         counter += 1
# print('{} {}'.format(split, counter))


def real_post_process(split='error', padding=True, bert_embedd_dim=1024):
    orig_text = codecs.open('prepro_data/dev_%s_text.txt' % split, 'r', encoding='utf8').readlines()
    bert_text = codecs.open('prepro_data/dev_%s_bert_text.json' % split, 'r', encoding='utf8').readlines()
    with h5py.File('/home2/public/jp/DocRED/1024_dev_%s_text.h5' % split, 'r') as in_h5, \
            h5py.File('/home2/public/jp/DocRED/1024_dev_%s.h5' % split, 'w') as out_h5:
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


for split in ['dev', 'test', 'train']:
    real_post_process(split)

"""
train [140, 513, 551, 825, 952, 1176, 1337, 1350, 1783, 1856, 1985, 2670, 2841, 2844, 3024]
dev [234, 467, 526, 573, 633, 725, 836, 962]
test [6, 30, 571, 611, 719, 883, 936]

train
1034 | 228 229
2147 | 202 203
1009 | 144 146
2404 | 216 217
dev 
164 | 173 174
549 | 174 175
test
214 | 181 182
"""


def fix_f___(split='dev'):
    to_fix_indecx = {
        'train': [1034, 2147, 1009, 2404],
        'dev': [164, 549],
        'test': [214]
    }
    index_mapping = {
        1034: 0, 2147: 1, 1009: 2, 2404: 3,
        164: 4, 549: 5,
        214: 6
    }
    # error = codecs.open('prepro_data/dev_error_bert_text.json' % split, 'r', encoding='utf8').readlines()
    # bert_text = codecs.open('prepro_data/dev_%s_bert_text.json' % split, 'r', encoding='utf8').readlines()
    # orig_text = codecs.open('prepro_data/dev_%s_text.txt' % split, 'r', encoding='utf8').readlines()
    error_h5 = h5py.File('/home2/public/jp/DocRED/dev_error.h5', 'r')
    def get_right(_idx):
        fet = error_h5[str(index_mapping[_idx])]
        fet = np.array(fet)
        return fet
    with h5py.File('/home2/public/jp/DocRED/dev_%s.h5' % split, 'r') as in_h5, \
            h5py.File('/home2/public/jp/DocRED/dev_%s_new.h5' % split, 'w') as out_h5:
        for idx in range(num_dict[split]):
            if idx not in to_fix_indecx[split]:
                feature = np.array(in_h5[str(idx)])
            else:
                feature = get_right(idx)
            out_h5.create_dataset(name=str(idx),
                                  shape=feature.shape,
                                  dtype='float32',
                                  data=feature)
    error_h5.close()


# for split in ['dev', 'test', 'train']:
#     fix_f___(split)






