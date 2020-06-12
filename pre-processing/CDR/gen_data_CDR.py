import os
from copy import deepcopy
import json
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt

data_dir = 'E:/workspace/repo/bran/data/cdr/CDR_DevelopmentSet.PubTator.txt/train_test_dev/'


def mention_statistic():
    ner2id = {"BLANK": 0, "Disease": 1, "Chemical": 2}
    ner_counter = Counter()
    num_rels = 0
    for line in open(data_dir+'train.json', 'r', encoding='utf8'):
        ex = json.loads(line.strip())
        vertexSet = ex['vertexSet']
        for vertex in vertexSet:
            for mention in vertex:
                ner_counter.update([mention['type']])
        num_rels += len(ex['labels'])

    print(ner_counter.items())
    print(num_rels)  # test 1048, dev 974, train 1007


properties = ["pair_dist", "title", "labels", "na_triple", "Ls", "sents", "vertexSet", "index"]


def train_and_dev_to_real_train():
    with open(data_dir+'train_and_dev.json') as in_f, open(data_dir+'real_train_and_dev.json', 'w', encoding='utf8') as out_f:
        counter = 0
        out_dict = []
        for line in in_f:
            ex = json.loads(line.strip())
            out_ex = deepcopy(ex)
            out_ex['index'] = counter
            if counter == 0:
                print(out_ex.keys())
            out_dict.append(out_ex)
            counter += 1
        json.dump(out_dict, out_f, ensure_ascii=False)
    assert counter == 1000


def test_to_dev():
    with open(data_dir+'test.json') as in_f, open(data_dir+'dev.json', 'w', encoding='utf8') as out_f:
        counter = 0
        out_dict = []
        for line in in_f:
            ex = json.loads(line.strip())
            out_ex = deepcopy(ex)
            assert ex['index'] == counter
            out_dict.append(out_ex)
            counter += 1
        json.dump(out_dict, out_f, ensure_ascii=False)
        assert counter == 500


# train_and_dev_to_real_train()
# test_to_dev()


def all_ex_has_relations():  # all examples has relations
    for split in ['train', 'dev']:
        data_file_name = data_dir + split + '.json'
        ori_data = json.load(open(data_file_name))
        doc_lens = []
        doc_num_rels = []  # number of rels of each doc
        doc_num_vertex = []
        doc_num_mention = []
        num_coref_per_vertex = []
        num_head_corefs_X_tail_corefs = []
        counter = 0
        for idx, ex in enumerate(ori_data):
            if len(ex['labels']) == 0:
                print(idx)
            doc_num_rels.append(len(ex['labels']))
            l = 0
            for sent in ex['sents']:
                l += len(sent)
            doc_lens.append(l)
            n_mentions = 0
            lis = []
            doc_num_vertex.append(len(ex['vertexSet']))
            for vertex in ex['vertexSet']:
                num_coref_per_vertex.append(len(vertex))
                lis.append(len(vertex))
                n_mentions += len(vertex)
            doc_num_mention.append(n_mentions)
            lis.sort(reverse=True)
            num_head_corefs_X_tail_corefs.append(lis[0] * lis[1])
            counter += 1
        plt.figure()
        plt.hist(num_coref_per_vertex)
        plt.show()
        if split == 'train':
            assert counter == 1000
        if split == 'dev':
            assert counter == 500

        print(f"====={split}========\n"
              f"doc_lens      max: {np.max(doc_lens)}, avg: {np.mean(doc_lens)}, {np.sum((np.array(doc_lens) > 500))/float(counter)}\n"
              f"doc_num_rels      max: {np.max(doc_num_rels)}, avg: {np.mean(doc_num_rels)}\n"
              f"doc_num_vertex      max: {np.max(doc_num_vertex)}, avg: {np.mean(doc_num_vertex)}\n"
              f"doc_num_mention      max: {np.max(doc_num_mention)}, avg: {np.mean(doc_num_mention)}\n"
              f"num_coref_per_vertex      max: {np.max(num_coref_per_vertex)}, avg: {np.mean(num_coref_per_vertex)}\n"
              f"num_head_corefs_X_tail_corefs      max: {np.max(num_head_corefs_X_tail_corefs)}, avg: {np.mean(num_head_corefs_X_tail_corefs)}\n")
        """
        =====train========
        doc_lens      max: 620, avg: 226.849, 0.017
        doc_num_rels      max: 16, avg: 1.981
        doc_num_vertex      max: 22, avg: 6.738
        doc_num_mention      max: 60, avg: 18.976
        num_coref_per_vertex      max: 29, avg: 2.8162659542891064
        num_head_corefs_X_tail_corefs      max: 288, avg: 33.585
        
        =====dev========
        doc_lens      max: 614, avg: 238.07, 0.012
        doc_num_rels      max: 21, avg: 2.096
        doc_num_vertex      max: 23, avg: 6.838
        doc_num_mention      max: 69, avg: 19.618
        num_coref_per_vertex      max: 25, avg: 2.868967534366774
        num_head_corefs_X_tail_corefs      max: 238, avg: 37.226

        """


# all_ex_has_relations()


def get_dist(h_vertex, t_vertex):
    min_d = 620
    for h_m in h_vertex:
        for t_m in t_vertex:
            d = abs(h_m['pos'][0] - t_m['pos'][0])
            min_d = min(min_d, d)
    return min_d


def to_train_dev_test():
    for split in ['train', 'dev', 'test']:
        with open(f"{data_dir}{split}.json") as in_f, open(f"{data_dir}dev_{split}.json", 'w', encoding='utf8') as out_f:
            pass


def gen_data():
    doc_id_to_index = json.load(open(f'{data_dir}doc_id_to_index.json'))
    for split in ['train', 'dev']:
        # split = 'train'
        data_file_name = data_dir + split + '.json'
        ori_data = json.load(open(data_file_name))
        name_prefix = "dev"
        max_length = 620
        data = []
        for idx, ex in enumerate(ori_data):
            assert ex['index'] == doc_id_to_index[ex['title']]
            # "Ls"
            Ls = [0]
            L = 0
            for x in ori_data[idx]['sents']:
                L += len(x)
                Ls.append(L)
            out_ex = deepcopy(ex)
            out_ex['Ls'] = Ls
            # "pair_dist"
            vertexSet = ex['vertexSet']
            num_vertex = len(vertexSet)
            pair_dist = np.zeros((num_vertex, num_vertex))
            for i in range(num_vertex):
                for j in range(num_vertex):
                    if i == j:
                        continue
                    pair_dist[i, j] = get_dist(vertexSet[i], vertexSet[j])
            out_ex['pair_dist'] = pair_dist.tolist()
            # "na_triple"
            train_triple = set([])
            labels = ex['labels']
            assert labels
            for label in labels:
                train_triple.add((label['h'], label['t']))
            for vertex in vertexSet:
                entity_type = [men['type'] for men in vertex]
                entity_type = set(entity_type)
                if len(entity_type) != 1:
                    print(f"{split}, {idx}, {vertex[0]}, {entity_type}")
                    # exit(0)

            num_vertex = len(vertexSet)
            na_triple = []
            for h_idx in range(num_vertex):
                for t_idx in range(num_vertex):
                    h_type = vertexSet[h_idx][0]['type']
                    t_type = vertexSet[t_idx][0]['type']
                    if h_idx != t_idx and (h_type == 'Chemical' and t_type == 'Disease') and (h_idx, t_idx) not in train_triple:
                        na_triple.append((h_idx, t_idx))

            out_ex['na_triple'] = na_triple

            data.append(out_ex)
        # json.dump(data, open(data_dir + name_prefix + '_' + split + '.json', "w"))

        #
        char2id = json.load(open(data_dir + "char2id.json"))
        word2id = json.load(open(data_dir + "word2id.json"))
        ner2id = json.load(open(data_dir + "ner2id.json"))
        char_limit = 16
        sen_tot = len(ori_data)
        sen_word = np.zeros((sen_tot, max_length), dtype=np.int64)
        sen_pos = np.zeros((sen_tot, max_length), dtype=np.int64)
        sen_ner = np.zeros((sen_tot, max_length), dtype=np.int64)
        sen_char = np.zeros((sen_tot, max_length, char_limit), dtype=np.int64)

        for i in range(len(ori_data)):
            item = ori_data[i]
            words = []
            for sent in item['sents']:
                words += sent

            for j, word in enumerate(words):
                word = word.lower()

                if j < max_length:
                    if word in word2id:
                        sen_word[i][j] = word2id[word]
                    else:
                        sen_word[i][j] = word2id['UNK']

                    for c_idx, k in enumerate(list(word)):
                        if c_idx >= char_limit:
                            break
                        sen_char[i, j, c_idx] = char2id.get(k, char2id['UNK'])

            for j in range(j + 1, max_length):
                sen_word[i][j] = word2id['BLANK']

            vertexSet = item['vertexSet']

            for idx, vertex in enumerate(vertexSet, 1):
                for v in vertex:
                    sen_pos[i][v['pos'][0]:v['pos'][1]] = idx
                    sen_ner[i][v['pos'][0]:v['pos'][1]] = ner2id[v['type']]

        print("Finishing processing")
        np.save(data_dir + name_prefix + '_' + split + '_word_nolower.npy', sen_word)
        # np.save(data_dir + name_prefix + '_' + split + '_pos.npy', sen_pos)
        # np.save(data_dir + name_prefix + '_' + split + '_ner.npy', sen_ner)
        # np.save(data_dir + name_prefix + '_' + split + '_char.npy', sen_char)
        print("Finish saving")


# gen_data()


def extract_text():
    """ 5 bad cases: replace these token (space in token) with hyphen
        train_990  "4 1/2"  --> "4-1/2"
        train_178  "2 1/2"  --> "2-1/2

        dev_301   "1 1/2"  -->  ""1-1/2""
        dev_81    "<VGPR before consolidation therapy to >"  --> "<VGPR-before-consolidation-therapy-to->"
        dev_249   "2 1/2" --> "2-1/2"
    """
    bad_counter = 0
    for split in ['train', 'dev']:
        data = json.load(open(f"{data_dir}dev_{split}.json"))
        with open(f"{data_dir}dev_{split}_text.txt", 'w', encoding='utf8') as writer:
            for ex in data:
                text = []
                for sent in ex['sents']:
                    for tok in sent:
                        if ' ' in tok:
                            bad_counter += 1
                            tok = tok.replace(' ', '-')
                        text.append(tok)
                writer.write(' '.join(text) + '\n')
    assert bad_counter == 5


def num_pos_vs_neg():
    # Test entity without KB ID should be included; filter hypernyms should not be conducted on test set
    # This two settings will reduce the input of #negative pairs
    negs_to_drop = json.load(open(data_dir+"filter_out_negs.json"))
    split = 'test'
    pos_counter = 0
    all_counter = 0
    before_drop = 0
    for idx, ex in enumerate(json.load(open(data_dir+f"dev_{split}.json"))):
        pos_counter += len(ex['labels'])
        L = len(ex['vertexSet'])
        for h_idx in range(L):
            for t_idx in range(L):
                hlist = ex['vertexSet'][h_idx]
                tlist = ex['vertexSet'][t_idx]
                h_type = hlist[0]['type']
                t_type = tlist[0]['type']
                if h_idx != t_idx and (h_type == 'Chemical' and t_type == 'Disease'):
                    neg = [str(ex['title']), hlist[0]['KB_ID'], tlist[0]['KB_ID']]
                    before_drop += 1
                    if neg not in negs_to_drop:
                        all_counter += 1
    print(f"{split}: all: {0}, pos: {pos_counter}, neg: {all_counter-pos_counter}, drop: {before_drop-all_counter}")

    split = 'train'
    pos_counter = 0
    neg_counter = 0
    before_drop = 0
    for idx, ex in enumerate(json.load(open(data_dir+f"dev_train.json"))[:500]):
        pos_counter += len(ex['labels'])
        for h_idx, t_idx in ex['na_triple']:
            hlist = ex['vertexSet'][h_idx]
            tlist = ex['vertexSet'][t_idx]
            h_type = hlist[0]['type']
            t_type = tlist[0]['type']
            assert h_type in ['Chemical', 'Disease']
            assert t_type in ['Chemical', 'Disease']
            if h_idx != t_idx and (h_type == 'Chemical' and t_type == 'Disease'):
                neg = [str(ex['title']), hlist[0]['KB_ID'], tlist[0]['KB_ID']]
                before_drop += 1
                if neg not in negs_to_drop:
                    neg_counter += 1
    print(f"{split}: all: {0}, pos: {pos_counter}, neg: {neg_counter}, drop: {before_drop-neg_counter}")

    split = 'dev'
    pos_counter = 0
    neg_counter = 0
    before_drop = 0
    for idx, ex in enumerate(json.load(open(data_dir+f"dev_train.json"))[500:]):
        pos_counter += len(ex['labels'])
        for h_idx, t_idx in ex['na_triple']:
            hlist = ex['vertexSet'][h_idx]
            tlist = ex['vertexSet'][t_idx]
            h_type = hlist[0]['type']
            t_type = tlist[0]['type']
            assert h_type in ['Chemical', 'Disease']
            assert t_type in ['Chemical', 'Disease']
            if h_idx != t_idx and (h_type == 'Chemical' and t_type == 'Disease'):
                neg = [str(ex['title']), hlist[0]['KB_ID'], tlist[0]['KB_ID']]
                before_drop += 1
                if neg not in negs_to_drop:
                    neg_counter += 1
    print(f"{split}: all: {0}, pos: {pos_counter}, neg: {neg_counter}, drop: {before_drop-neg_counter}")
    # train: all: 0, pos: 1006, neg: 4088, drop: 181
    # dev: all: 0, pos:    974, neg: 3974, drop: 168
    # test: all: 0, pos:  1048, neg: 4034, drop: 192


num_pos_vs_neg()


# withID = open(r'E:\workspace\repo\bran\data\cdr\CDR_DevelopmentSet.PubTator.txt\with_KBID\dev_test.json', 'r', encoding='utf8').readlines()
# with open(data_dir+'dev_dev.json', 'r', encoding='utf8') as in_f1:
#     for idx, now_ex in enumerate(json.load(in_f1)):
#         withID_ex = json.loads(withID[idx].strip())
#         withID_labels = withID_ex['labels']
#         for _ii, label in enumerate(now_ex['labels']):
#             assert label['h'] == withID_labels[_ii]['h'], f"{idx}, {now_ex['title']}, _ii:{_ii}"


def num2_p_r_f(num_true_all, num_pred_all, pred_correct):
    print(num_true_all, num_pred_all, pred_correct)
    precision = (1.0*pred_correct / num_pred_all) if num_pred_all > 0 else 0.
    recall = (1.0*pred_correct / num_true_all) if num_true_all > 0 else 0.
    f1 = (2.0*precision*recall / (precision + recall)) if (precision + recall) > 0 else 0.
    return precision, recall, f1


recall = 0
true_rel_facts = set()
with open(r'E:\workspace\repo\bran\data\cdr\CDR_DevelopmentSet.PubTator.txt\with_KBID\raw_test.json', 'r', encoding='utf8') as in_fi:
    idx = 0
    for line in in_fi:
        ex = json.loads(line.strip())
        recall += len(ex['rels'])
        for rel in ex['rels']:
            true_rel_facts.add((str(idx), rel[2], rel[3]))  # (DOC_idx, h_ID, t_ID)
        idx += 1
        # if idx == 5:
        #     print(true_rel_facts)
assert recall == 1066
assert len(true_rel_facts) == 1066
pos_to_KB_ID = {}
is_inter = {}
with open(r'E:\workspace\repo\bran\data\cdr\CDR_DevelopmentSet.PubTator.txt\with_KBID\dev_test.json', 'r', encoding='utf8') as in_fi:
    idx = 0
    for line in in_fi:
        ex = json.loads(line.strip())
        for vertex in ex['vertexSet']:
            for m in vertex:
                key = f"{idx}_{m['pos'][0]}_{m['pos'][1]}"
                pos_to_KB_ID[key] = m['KB_ID'].split('|')
        idx += 1

doc_idx_vertex_idx_to_pos = {}
with open(r'E:\DASFAA2020\experiments\CDR_results\server_dev_dev.json', 'r', encoding='utf8') as in_fi:
    for ex in json.load(in_fi):
        for vertex_idx, vertex in enumerate(ex['vertexSet']):
            key = f"{ex['index']}_{vertex_idx}"
            doc_idx_vertex_idx_to_pos[key] = (vertex[0]['pos'][0], vertex[0]['pos'][1])


def idx_to_IDs(doc_idx, vertex_idx):
    key = f"{doc_idx}_{vertex_idx}"
    pos0, pos1 = doc_idx_vertex_idx_to_pos[key]
    key = f"{doc_idx}_{pos0}_{pos1}"
    return pos_to_KB_ID[key]


def p_r_f():
    # 'E:/DASFAA2020/experiments/CDR_results/Baseline/CDR_BiLSTM-seed%s_dev_dev.json'
    # 'E:/DASFAA2020/experiments/CDR_results/bran/CDR_bran-seed%s_dev_dev.json'
    # 'E:/DASFAA2020/experiments/CDR_results/NONE/CDR_NONE-seed%s_dev_dev.json'
    # 'E:/DASFAA2020/experiments/CDR_results/Ours/CDR_Ours-v4-seed%s_dev_dev.json'

    # 'E:/DASFAA2020/experiments/CDR_results/GloVe-Ours/CDR_PubMed_Ours-seed%s_dev_dev.json'
    results_file_pattern = r'E:\DASFAA2020\experiments\CDR_results\GloVe-real_Ours\CDR_PubMed_Ours-seed%s_dev_dev.json'

    ps, rs, f1s = [], [], []
    for file_no in range(1):
        preds = json.load(open(results_file_pattern % file_no))
        pred_rel_facts = set()
        for pred_idx, pred in enumerate(preds):
            doc_idx = pred['index']
            h_idx = pred['h_idx']
            t_idx = pred['t_idx']
            h_IDs = idx_to_IDs(doc_idx, h_idx)
            t_IDs = idx_to_IDs(doc_idx, t_idx)
            for hID in h_IDs:
                for tID in t_IDs:
                    pred_rel_facts.add((str(doc_idx), hID, tID))
            # if pred_idx == 5:
            #     print(pred_rel_facts)
        # server_facts_list = json.load(open('E:\DASFAA2020\experiments\CDR_results\GloVe-real_Ours\CDR_PubMed_Ours-seed1_dev_dev_facts.json'))
        # server_facts = set()
        # for item in server_facts_list:
        #     server_facts.add((item[0], item[1], item[2]))
        # for item in server_facts - pred_rel_facts:
        #     print(item)
        precision, recall, f1 = num2_p_r_f(len(true_rel_facts), len(pred_rel_facts), len(true_rel_facts & pred_rel_facts))
        ps.append(precision)
        rs.append(recall)
        f1s.append(f1)
        print("seed: {} | precision: {:.2f}, recall: {:.2f}, f1: {:.2f}".format(file_no, precision*100, recall*100, f1*100))
    print("AVG: precision: {:.2f}, recall: {:.2f}, f1: {:.2f}".format(np.mean(ps)*100, np.mean(rs)*100, np.mean(f1s)*100))
    # + Base AVG: precision: 57.36, recall: 72.47, f1: 64.00
    # + bran AVG: precision: 59.71, recall: 72.69, f1: 65.43
    # + NONE AVG: precision: 61.22, recall: 70.91, f1: 65.65
    # + Ours AVG: precision: 61.91, recall: 71.63, f1: 66.36
    # + Ours-NONE  AVG: precision: 53.68, recall: 71.95, f1: 61.40
    # + Ours-Glove AVG: precision: 56.81, recall: 73.31, f1: 63.99


# p_r_f()


def gen_word2id_and_wordvec():
    word_dim = 200
    word2id = {'BLANK': 0, 'UNK': 1}
    vectors = []
    vectors.append(np.zeros(word_dim, dtype=np.float))
    vectors.append(np.zeros(word_dim, dtype=np.float))
    with open(r'E:\workspace\repo\bran\data\cdr\CDR_DevelopmentSet.PubTator.txt\with_KBID\PubMed-CDR.txt', 'r') as lines:
        idx = 2
        for x, line in enumerate(lines):

            if x == 0 and len(line.split()) == 2:
                words, num = map(int, line.strip().split())
                print(f"words:{words}, num: {num}")
            else:
                word = line.strip().split()[0]
                vec = line.strip().split()[1:]
                n = len(vec)
                if n != word_dim:
                    print('Wrong dimensionality! -- line No{}, word: {}, len {}'.format(x, word, n))
                    continue
                word2id[word] = idx
                vectors.append(np.asarray(vec, 'f'))
                idx += 1
    assert len(vectors) == idx and len(vectors) == len(word2id)
    print('  Found pre-trained word embeddings: {} x {}'.format(len(vectors), word_dim), end="")  # 18607 x 200
    json.dump(word2id, open(r'E:\workspace\repo\bran\data\cdr\CDR_DevelopmentSet.PubTator.txt\with_KBID\PubMed_word2id.json', 'w', encoding='utf8'), ensure_ascii=False)
    np.save(r'E:\workspace\repo\bran\data\cdr\CDR_DevelopmentSet.PubTator.txt\with_KBID\PubMed_vec.npy', vectors)


# p_r_f()
# train filter-out 192
# dev   filter-out 174
# test  filter-out 201
# dev 1728915, D002220, D006331


def get_split_filter_out_negs(split):
    all_neg_rels = []
    with open(r'E:\workspace\repo\bran\data\cdr\processed\just_train_2500\negative_0_CDR_%s.txt' % split, 'r', encoding='utf8') as all:
        for line in all:
            line = line.strip()
            pieces = line.split('\t')
            ID, hID, tID = pieces[10], pieces[0], pieces[5]
            all_neg_rels.append((ID, hID, tID))
    filtered_neg_rels = []
    with open(r'E:\workspace\repo\bran\data\cdr\processed\just_train_2500\negative_0_CDR_%s_filtered.txt' % split, 'r', encoding='utf8') as filtered:
        for line in filtered:
            line = line.strip()
            pieces = line.split('\t')
            ID, hID, tID = pieces[10], pieces[0], pieces[5]
            filtered_neg_rels.append((ID, hID, tID))
    filter_out_neg_rels = []
    for neg in set(all_neg_rels) - set(filtered_neg_rels):
        ID, hID, tID = list(neg)
        filter_out_neg_rels.append((ID, hID, tID))
    return filter_out_neg_rels


def get_all_filter_out_negs():
    train_out = set(get_split_filter_out_negs('train'))
    dev_out = set(get_split_filter_out_negs('dev'))
    test_out = set(get_split_filter_out_negs('test'))

    assert len(train_out | dev_out | test_out) == 192+174+201  # 567
    assert len(train_out & dev_out & test_out) == 0
    filter_out_negs = []
    for neg in train_out | dev_out | test_out:
        filter_out_negs.append(list(neg))
    json.dump(filter_out_negs, open(r'E:\workspace\repo\bran\data\cdr\processed\just_train_2500\filter_out_negs.json', 'w'), ensure_ascii=False, indent=2)


def gen_DocID_to_index():
    doc_id_to_index = {}
    idx = 0
    for ex in json.load(open(r'E:\workspace\repo\bran\data\cdr\CDR_DevelopmentSet.PubTator.txt\train.json', 'r', encoding='utf8')):
        assert ex['index'] == idx
        assert ex['title'] not in doc_id_to_index
        doc_id_to_index[ex['title']] = idx
        idx += 1
    idx = 0
    for ex in json.load(open(r'E:\workspace\repo\bran\data\cdr\CDR_DevelopmentSet.PubTator.txt\dev.json', 'r', encoding='utf8')):
        assert ex['index'] == idx
        assert ex['title'] not in doc_id_to_index
        doc_id_to_index[ex['title']] = idx
        idx += 1

    json.dump(doc_id_to_index, open(r'E:\workspace\repo\bran\data\cdr\CDR_DevelopmentSet.PubTator.txt\doc_id_to_index.json', 'w'))


def checkUpperLower():
    word2id = json.load(open(data_dir + "word2id.json"))
    id2word = {v: k for k, v in word2id.items()}
    dev_test_word = np.load(r'E:\DASFAA2020\experiments\CDR_results\logs\dev_test_word_nolower.npy')
    for i in range(3):
        doc = [id2word[wordid] for wordid in dev_test_word[i, :]]
        print(' '.join(doc))

# gen_DocID_to_index()


# gen_word2id_and_wordvec()

# vectors = np.load(r'E:\workspace\repo\DocRED\code\prepro_data\vec.npy')
# print(vectors[0])
# print(vectors[1])

# num_pos_vs_neg()
# num_pos_vs_neg()
# extract_text()

"""

extract_bert_tokens
python extract_features.py \
--input_file=E:/workspace/repo/bran/data/cdr/CDR_DevelopmentSet.PubTator.txt/dev_train_text.txt \
--output_file=E:/workspace/repo/bran/data/cdr/CDR_DevelopmentSet.PubTator.txt/dev_train_text.txt \
--vocab_file=F:/sciBERT/vocab.txt \
--bert_config_file=F:/sciBERT/bert_config.json \
--init_checkpoint=F:/sciBERT/bert_model.ckpt \
--layers=-1 \
--max_seq_length=512 \
--batch_size=8 \
--do_lower_case False/True

extract_bert_embeddings
CUDA_VISIBLE_DEVICES=7 python extract_features.py \
  --input_file=/home/jp/workspace2/datasets/CDR/dev_train_text.txt \
  --output_file=/home2/public/jp/CDR_bert_features/768_dev_train_text.h5 \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --layers=-1 \
  --max_seq_length=512 \
  --batch_size=8 \
  --do_lower_case False/True
"""

"""
Train
λ python filter_hypernyms.py -p positive_0_CDR_train.txt -n negative_0_CDR_train.txt -m E:\workspace\repo\bran\data\2017MeshTree.txt
 -o negative_0_CDR_train_filtered.txt                                                                                               
Loading positive examples from positive_0_CDR_train.txt                                                                             
Loading negative examples from negative_0_CDR_train.txt                                                                             
Mesh entities: 28536                                                                                                                
Positive Docs: 500                                                                                                                  
Negative Docs: 475                                                                                                                  
Positive Count: 1038   Initial Negative Count: 4472   Final Negative Count: 4280    drop: 192   Hyponyms: 336      
                              
Dev                                                                                                                                   
E:\workspace\repo\bran\data\cdr\processed\just_train_2500 (master -> origin)                                                        
λ python filter_hypernyms.py -p positive_0_CDR_dev.txt -n negative_0_CDR_dev.txt -m E:\workspace\repo\bran\data\2017MeshTree.txt -o 
negative_0_CDR_dev_filtered.txt                                                                                                     
Loading positive examples from positive_0_CDR_dev.txt                                                                               
Loading negative examples from negative_0_CDR_dev.txt                                                                               
Mesh entities: 28542                                                                                                                
Positive Docs: 500                                                                                                                  
Negative Docs: 482                                                                                                                  
Positive Count: 1012   Initial Negative Count: 4310   Final Negative Count: 4136    drop: 174   Hyponyms: 287                                    

Test                                                                                                                                    
E:\workspace\repo\bran\data\cdr\processed\just_train_2500 (master -> origin)                                                        
λ python filter_hypernyms.py -p positive_0_CDR_test.txt -n negative_0_CDR_test.txt -m E:\workspace\repo\bran\data\2017MeshTree.txt -
o negative_0_CDR_test_filtered.txt                                                                                                  
Loading positive examples from positive_0_CDR_test.txt                                                                              
Loading negative examples from negative_0_CDR_test.txt                                                                              
Mesh entities: 28539                                                                                                                
Positive Docs: 500                                                                                                                  
Negative Docs: 477                                                                                                                  
Positive Count: 1066   Initial Negative Count: 4471   Final Negative Count: 4270    drop: 201   Hyponyms: 359                                    

"""


























