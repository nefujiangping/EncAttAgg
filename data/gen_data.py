import numpy as np
import os
import json
import argparse
from transformers import AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument('--in_path', type=str, default="./prepro_data/DocRED")
parser.add_argument('--out_path', type=str, default="./prepro_data/DocRED")
parser.add_argument('--pretrained_model_name_or_path', type=str, default="bert-base-cased")

args = parser.parse_args()
in_path = args.in_path
out_path = args.out_path
tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path)
pad_token_id = tokenizer.pad_token_id
pieces_per_token_limit = 6  # 99.9% tokens have #word-pieces smaller than 6

train_distant_file_name = os.path.join(in_path, 'train_distant.json')
train_annotated_file_name = os.path.join(in_path, 'train_annotated.json')
dev_file_name = os.path.join(in_path, 'dev.json')
test_file_name = os.path.join(in_path, 'test.json')

rel2id = json.load(open(os.path.join(out_path, 'rel2id.json'), "r"))
id2rel = {v: u for u, v in rel2id.items()}
json.dump(id2rel, open(os.path.join(out_path, 'id2rel.json'), "w"))
fact_in_train = set([])
fact_in_dev_train = set([])


def init(data_file_name, rel2id, max_length=512, is_training=True, suffix=''):
    ori_data = json.load(open(data_file_name))
    if is_training:
        name_prefix = "train"
    else:
        name_prefix = "dev"
    Ma = 0
    Ma_e = 0
    data = []
    for i in range(len(ori_data)):
        Ls = [0]
        L = 0
        for x in ori_data[i]['sents']:
            L += len(x)
            Ls.append(L)

        vertexSet = ori_data[i]['vertexSet']
        # point position added with sent start position
        for j in range(len(vertexSet)):
            for k in range(len(vertexSet[j])):
                vertexSet[j][k]['sent_id'] = int(vertexSet[j][k]['sent_id'])

                sent_id = vertexSet[j][k]['sent_id']
                dl = Ls[sent_id]
                pos1 = vertexSet[j][k]['pos'][0]
                pos2 = vertexSet[j][k]['pos'][1]
                vertexSet[j][k]['pos'] = (pos1 + dl, pos2 + dl)

        ori_data[i]['vertexSet'] = vertexSet

        item = {}
        item['vertexSet'] = vertexSet
        labels = ori_data[i].get('labels', [])

        train_triple = set([])
        new_labels = []
        for label in labels:
            rel = label['r']
            assert (rel in rel2id)
            label['r'] = rel2id[label['r']]

            train_triple.add((label['h'], label['t']))

            if suffix == '_train':
                for n1 in vertexSet[label['h']]:
                    for n2 in vertexSet[label['t']]:
                        fact_in_dev_train.add((n1['name'], n2['name'], rel))

            if is_training:
                for n1 in vertexSet[label['h']]:
                    for n2 in vertexSet[label['t']]:
                        fact_in_train.add((n1['name'], n2['name'], rel))

            else:
                # fix a bug here
                label['intrain'] = False
                label['indev_train'] = False

                for n1 in vertexSet[label['h']]:
                    for n2 in vertexSet[label['t']]:
                        if (n1['name'], n2['name'], rel) in fact_in_train:
                            label['intrain'] = True

                        if suffix == '_dev' or suffix == '_test':
                            if (n1['name'], n2['name'], rel) in fact_in_dev_train:
                                label['indev_train'] = True

            new_labels.append(label)

        item['labels'] = new_labels
        item['title'] = ori_data[i]['title']

        na_triple = []
        for j in range(len(vertexSet)):
            for k in range(len(vertexSet)):
                if (j != k):
                    if (j, k) not in train_triple:
                        na_triple.append((j, k))

        item['na_triple'] = na_triple
        item['Ls'] = Ls
        item['sents'] = ori_data[i]['sents']
        data.append(item)

        Ma = max(Ma, len(vertexSet))
        Ma_e = max(Ma_e, len(item['labels']))

    print('data_len:', len(ori_data))

    # saving
    print("Saving files")

    json.dump(data, open(os.path.join(out_path, name_prefix + suffix + '.json'), "w"))

    word2id = json.load(open(os.path.join(out_path, "word2id.json")))
    ner2id = json.load(open(os.path.join(out_path, "ner2id.json")))

    sen_tot = len(ori_data)
    sen_word = np.zeros((sen_tot, max_length), dtype=np.int64)
    sen_pos = np.zeros((sen_tot, max_length), dtype=np.int64)
    sen_ner = np.zeros((sen_tot, max_length), dtype=np.int64)
    token_pieces_map = np.zeros((sen_tot, max_length, pieces_per_token_limit), dtype=np.int64)
    token_pieces_map_mask = np.zeros((sen_tot, max_length, pieces_per_token_limit), dtype=np.int64)
    input_ids = np.ones((sen_tot, 1024), dtype=np.int64) * int(pad_token_id)

    offset = 1  # [CLS]

    for i in range(len(ori_data)):
        item = ori_data[i]
        words = []
        for sent in item['sents']:
            words += sent

        doc_pieces = []
        first_piece_idx = offset
        for j, word in enumerate(words):

            if j < max_length:
                if word.lower() in word2id:
                    sen_word[i][j] = word2id[word.lower()]
                else:
                    sen_word[i][j] = word2id['UNK']

                word_pieces = tokenizer.tokenize(word)
                doc_pieces += word_pieces
                num_pieces = min(len(word_pieces), pieces_per_token_limit)
                token_pieces_map[i, j, :num_pieces] = list(range(first_piece_idx, first_piece_idx + num_pieces))
                token_pieces_map_mask[i, j, :num_pieces] = 1
                first_piece_idx = first_piece_idx + len(word_pieces)

        doc_pieces = tokenizer.convert_tokens_to_ids(doc_pieces)
        doc_pieces = tokenizer.build_inputs_with_special_tokens(doc_pieces)
        input_ids[i, :len(doc_pieces)] = doc_pieces

        for j in range(j + 1, max_length):
            sen_word[i][j] = word2id['BLANK']

        vertexSet = item['vertexSet']

        for idx, vertex in enumerate(vertexSet, 1):
            for v in vertex:
                sen_pos[i][v['pos'][0]:v['pos'][1]] = idx
                sen_ner[i][v['pos'][0]:v['pos'][1]] = ner2id[v['type']]

        if i < 2:
            for e_i, (token, ner, cluster, wordpieces) in enumerate(
                    zip(words, sen_ner[i], sen_pos[i], token_pieces_map[i])):
                print(e_i, token, ner, cluster, wordpieces, tokenizer.convert_ids_to_tokens(
                    input_ids[i, wordpieces[0]: wordpieces[(wordpieces > 0).sum() - 1] + 1]))

    print("Finishing processing")
    np.save(os.path.join(out_path, name_prefix + suffix + '_word.npy'), sen_word)
    np.save(os.path.join(out_path, name_prefix + suffix + '_input_ids.npy'), input_ids)
    np.save(os.path.join(out_path, name_prefix + suffix + '_pieces_token_map.npy'), token_pieces_map)
    np.save(os.path.join(out_path, name_prefix + suffix + '_pieces_token_map_mask.npy'), token_pieces_map_mask)
    np.save(os.path.join(out_path, name_prefix + suffix + '_pos.npy'), sen_pos)
    np.save(os.path.join(out_path, name_prefix + suffix + '_ner.npy'), sen_ner)
    print("Finish saving")


init(train_distant_file_name, rel2id, max_length=512, is_training=True, suffix='')
init(train_annotated_file_name, rel2id, max_length=512, is_training=False, suffix='_train')
init(dev_file_name, rel2id, max_length=512, is_training=False, suffix='_dev')
init(test_file_name, rel2id, max_length=512, is_training=False, suffix='_test')
