from stanfordcorenlp import StanfordCoreNLP
import json
import numpy as np
from copy import deepcopy
import logging

"""
stanford-corenlp-full-2018-02-27 (v3.9.1)
"""

data_dir = 'E:/workspace/repo/bran/data/cdr/CDR_DevelopmentSet.PubTator.txt/train_test_dev'
corenlp_home = 'F:/UbuntuShare/tools/stanford-corenlp-full-2018-02-27'
# txt --> raw json
def step1_to_json(split):
    with open('%s/CDR_%sSet.PubTator.txt' % (data_dir, split), 'r', encoding='utf8') as in_f, \
            open('%s/raw_%s.json' % (data_dir, split), 'w', encoding='utf8') as out_f:
        title, abstract = None, None
        rels, mentions = [], []
        for line in in_f:
            line = line.strip()
            if line:
                if "|t|" in line:
                    mark_idx = line.index("|")
                    doc_idx = line[:mark_idx]
                    title = line[mark_idx+3:]
                elif "|a|" in line:
                    mark_idx = line.index("|")
                    abstract = line[mark_idx+3:]
                else:
                    pieces = line.split('\t')
                    len_pieces = len(pieces)
                    if len_pieces == 4:  # relation
                        assert "CID" in line, f'{doc_idx}: CID in line'
                        if pieces[2] != pieces[3]:  # drop "self-relation"
                            rels.append(pieces)
                        else:
                            print(doc_idx, 'This doc has self-relation.')
                    else:
                        assert len_pieces in [6,7], f'{doc_idx}: len in [5,6]'
                        if len_pieces == 7:
                            assert '|' in line, 'len_pieces=7, | in line'
                            pieces = pieces[:-1]
                            print(line)
                        assert len(pieces) == 6, ''
                        if pieces[5] != "-1":  # drop mention without KB ID
                            mentions.append(pieces)
            else:
                out_ex = {
                    'sents': title + " " + abstract,
                    'title': doc_idx,
                    'rels': rels,
                    'mentions': mentions
                }
                title, abstract = None, None
                rels, mentions = [], []
                out_f.write(json.dumps(out_ex) + '\n')


def value_to_index(lis, tgt_value):
    out_idx = []
    for idx, value in enumerate(lis):
        if value == tgt_value:
            out_idx.append(idx)
    # print(out_idx)
    return out_idx


def offsets_to_token_idx(meta, begin_offsets, end_offsets):
    pos_start, pos_end, sent_ids = np.array([-1]*len(begin_offsets)), np.array([-1]*len(begin_offsets)), np.array([-1]*len(begin_offsets))
    for _meta in meta:
        for token in _meta['tokens']:
            begin = token['characterOffsetBegin']
            if begin in begin_offsets:
                indices = value_to_index(begin_offsets, begin)
                pos_start[indices] = token['token_abs_idx']
                sent_ids[indices] = _meta['sent_id']
            end = token['characterOffsetEnd']
            if end in end_offsets:
                indices = value_to_index(end_offsets, end)
                pos_end[indices] = token['token_abs_idx']
        # print(f"begin, end: [{begin},{end}]")
    # assert np.min(pos_start) >= 0
    # assert np.min(pos_end) >= 0
    # assert np.min(sent_ids) >= 0
    pos_end += 1
    return pos_start.tolist(), pos_end.tolist(), sent_ids.tolist()


def intersection(to_check, lis):
    s,e = to_check
    for intervel in lis:
        lis_1, lis_2 = intervel
        if not (lis_2<=s or lis_1>=e):
            return True
    return False


def to_pieces(s, spliter):
    toks = s.split(spliter)
    new_toks = []
    for t in toks:
        new_toks.append(t)
        new_toks.append(spliter)
    new_toks = new_toks[:-1]
    return new_toks


def step3(split):
    BLANK_STR = ' '*80
    bad_mentions_file = '%s/bad_mentions.txt' % data_dir
    bad_mentions = {}
    with open(bad_mentions_file) as bad_f:
        for line in bad_f:
            ID, tok, p1, p2 = line.strip().split('\t')
            assert tok == p1+p2
            if ID not in bad_mentions:
                bad_mentions[ID] = {'bad_words': [tok], 'pieces': [[p1, p2]]}
            else:
                bad_mentions[ID]['bad_words'].append(tok)
                bad_mentions[ID]['pieces'].append([p1,p2])
    props = {'annotators': 'tokenize,ssplit','pipelineLanguage':'en','outputFormat':'json', 'tokenize.options': "splitHyphenated=true"}
    with StanfordCoreNLP(corenlp_home, memory='8g') as nlp, \
            open('%s/raw_v2_%s.json' % (data_dir, split), 'r', encoding='utf8') as in_f, \
            open('%s/dev_%s.json' % (data_dir, split), 'w', encoding='utf8') as out_file:
        idx = 0
        extreme_bad, bad = 0, 0
        for line in in_f:
            obj = json.loads(line.strip())

            text = obj['sents']

            res = nlp.annotate(text, props)
            json_obj = json.loads(res)

            begin_offsets, end_offsets = [],[]
            for m in obj['mentions']:
                begin_offsets.append(int(m[1]))
                end_offsets.append(int(m[2]))

            sen_list = []
            sent_meta = []
            abs_idx = 0
            doc_ID = obj['title']
            lisss = list(zip(begin_offsets, end_offsets))
            for sent_idx, sentence in enumerate(json_obj['sentences']):
                tokens = []
                relative_idx = 0
                _meta = {"sent_id": sent_idx, 'tokens': []}
                for token in sentence['tokens']:
                    __sss, __eee = token['characterOffsetBegin'], token['characterOffsetEnd']
                    word = text[__sss : __eee]

                    if doc_ID == '19370593' and word == 'embolism.One':
                        for w in ['embolism', '.', 'One']:
                            tokens.append(w)
                            _meta['tokens'].append({
                                "token_abs_idx": abs_idx,
                                "token_relative_idx": relative_idx,
                                "characterOffsetBegin": __sss,
                                "characterOffsetEnd": __sss+len(w)
                            })
                            __sss += len(w)
                            abs_idx += 1
                            relative_idx += 1
                        continue

                    if doc_ID == '11672959' and word == 'reaction.A':
                        for w in ['reaction', '.', 'A']:
                            tokens.append(w)
                            _meta['tokens'].append({
                                "token_abs_idx": abs_idx,
                                "token_relative_idx": relative_idx,
                                "characterOffsetBegin": __sss,
                                "characterOffsetEnd": __sss+len(w)
                            })
                            __sss += len(w)
                            abs_idx += 1
                            relative_idx += 1
                        continue


                    if doc_ID == '16298782' and word == 'this.Aminoglycoside':
                        for w in ['this', '.', 'Aminoglycoside']:
                            tokens.append(w)
                            _meta['tokens'].append({
                                "token_abs_idx": abs_idx,
                                "token_relative_idx": relative_idx,
                                "characterOffsetBegin": __sss,
                                "characterOffsetEnd": __sss+len(w)
                            })
                            __sss += len(w)
                            abs_idx += 1
                            relative_idx += 1
                        continue


                    # handle bad mentions

                    if doc_ID in bad_mentions and word in bad_mentions[doc_ID]['bad_words']:
                        bad_words_ofThisDoc = bad_mentions[doc_ID]['bad_words']
                        p1, p2 = bad_mentions[doc_ID]['pieces'][bad_words_ofThisDoc.index(word)]
                        print(word)
                        for w in [p1, p2]:
                            tokens.append(w)
                            _meta['tokens'].append({
                                "token_abs_idx": abs_idx,
                                "token_relative_idx": relative_idx,
                                "characterOffsetBegin": __sss,
                                "characterOffsetEnd": __sss+len(w)
                            })
                            __sss += len(w)
                            abs_idx += 1
                            relative_idx += 1

                    else:

                        spliter1 = "/"
                        spliter2 = "-"
                        spliter3 = "+"
                        if len(word) >= 2 and (spliter1 in word or spliter2 in word or spliter3 in word) and intersection((__sss, __eee), lisss):
                            new_words = to_pieces(word, spliter1)
                            if spliter2 in word:
                                new_new_words = []
                                for w in new_words:
                                    if spliter2 in w:
                                        new_new_words.extend(to_pieces(w, spliter2))
                                    else:
                                        new_new_words.append(w)
                            else:
                                new_new_words = new_words
                            new_words = new_new_words

                            if spliter3 in word:
                                new3_words = []
                                for w in new_new_words:
                                    if spliter3 in w:
                                        new3_words.extend(to_pieces(w, spliter3))
                                    else:
                                        new3_words.append(w)
                                new_words = new3_words


                            _s_s_s = __sss
                            for w in new_words:
                                tokens.append(w)
                                _meta['tokens'].append({
                                    "token_abs_idx": abs_idx,
                                    "token_relative_idx": relative_idx,
                                    "characterOffsetBegin": _s_s_s,
                                    "characterOffsetEnd": _s_s_s+len(w)
                                })
                                _s_s_s += len(w)
                                abs_idx += 1
                                relative_idx += 1
                        else:
                            tokens.append(word)
                            _meta['tokens'].append({
                                "token_abs_idx": abs_idx,
                                "token_relative_idx": relative_idx,
                                "characterOffsetBegin": __sss,
                                "characterOffsetEnd": __eee
                            })
                            abs_idx += 1
                            relative_idx += 1

                sen_list.append(tokens)
                sent_meta.append(_meta)
            assert sent_meta[-1]["tokens"][-1]["characterOffsetEnd"] == len(text)


            text_toks = []
            for sent in sen_list:
                for tok in sent:
                    text_toks.append(tok)

            # if doc_ID == '11672959':
            # 	print(text_toks)

            # print(m[1], m[2])

            pos_start, pos_end, sent_ids = offsets_to_token_idx(sent_meta, begin_offsets, end_offsets)
            # if obj['title'] == '2343592':
            # 	print(sent_meta)
            # print(pos_start)
            # print(pos_end)
            ex_bad_counter = 0
            for i in range(len(pos_start)):
                # print(f"sent_id: {sent_ids[i]}", f"[{pos_start[i]},{pos_end[i]}]",' '.join(text_toks[pos_start[i]: pos_end[i]]))
                if pos_start[i] == -1 or pos_end[i] == 0 or sent_ids[i] == -1:
                    print(BLANK_STR + f"BAD | {obj['title']} | [{pos_start[i]},{pos_end[i]}]: {obj['mentions'][i][1]}:{obj['mentions'][i][2]}")
                    ex_bad_counter += 1
                    bad += 1
                if pos_start[i] == -1 and pos_end[i] == 0:
                    print(BLANK_STR + f"BAD | {obj['title']} | [{pos_start[i]},{pos_end[i]}]")
                    extreme_bad += 1
            if ex_bad_counter != 0:
                print(BLANK_STR + f"BAD | {obj['title']} | {ex_bad_counter} bad mentions")


            final_vertexSet = []
            vertexSet = {}
            vertex_ID_to_index = {}
            vertex_id = 0
            for m_idx, m in enumerate(obj['mentions']):
                if m[-1] not in vertexSet:
                    vertexSet[m[-1]] = []
                    vertex_ID_to_index[m[-1]] = vertex_id
                    vertex_id += 1
                vertexSet[m[-1]].append(
                    {
                        "type": m[4],
                        "sent_id": sent_ids[m_idx],
                        "name": m[3],
                        "pos": [pos_start[m_idx], pos_end[m_idx]],
                        "KB_ID": m[-1]
                    }
                )

            index_to_ID = {v:k for k,v in vertex_ID_to_index.items()}
            for i in range(len(vertexSet)):
                final_vertexSet.append(deepcopy(vertexSet[index_to_ID[i]]))

            final_labels = []
            for rel in obj['rels']:
                head_ID, tail_ID = rel[2], rel[3]
                final_labels.append({"h": vertex_ID_to_index[head_ID], "intrain": False, "evidence": [], "r": 1,
                                     "t": vertex_ID_to_index[tail_ID], "indev_train": False})

            out_ex = {
                'sents': sen_list,
                'title': obj['title'],  # actual is ID of the article
                'labels': final_labels,
                'vertexSet': final_vertexSet,
                'index': idx
            }
            idx += 1
            # if idx == 3:
            # 	break
            print()

            out_file.write(json.dumps(out_ex) + '\n')

        print(f"extreme_bad: {extreme_bad}")
        print(f"bad: {bad}")
        assert bad == 0

    assert idx == 500


def step2_mergeIDs_remove_redunt_rel(split):
    # merge IDs
    with open('%s/raw_%s.json' % (data_dir, split), 'r', encoding='utf8') as in_file, \
            open('%s/raw_v2_%s.json' % (data_dir, split), 'w', encoding='utf8') as out_file:
        for line in in_file:
            obj = json.loads(line.strip())
            ids = set()
            out_mentions = deepcopy(obj['mentions'])
            for m in obj['mentions']:
                assert len(m) == 6
                ids.add(m[-1])
            for m_idx, m in enumerate(obj['mentions']):
                for _id in ids:
                    if m[-1] in _id and m[-1] != _id:
                        out_mentions[m_idx][-1] = _id
                        break
            out_rels = deepcopy(obj['rels'])
            if obj['rels']:
                for r_idx, rel in enumerate(obj['rels']):
                    for _id in ids:
                        if rel[-1] in _id and rel[-1] != _id:
                            out_rels[r_idx][-1] = _id
                        if rel[-2] in _id and rel[-2] != _id:
                            out_rels[r_idx][-2] = _id
                            break
            else:
                print(obj['title'], '  has no rel instance')
            final_rels = set()
            for rel in out_rels:
                final_rels.add(tuple(rel))
            out_rels = []
            for rel in final_rels:
                out_rels.append(list(rel))

            out_ex = {
                'sents': obj['sents'],
                'title': obj['title'],
                'rels': out_rels,
                'mentions': out_mentions
            }

            out_file.write(json.dumps(out_ex) + '\n')


if __name__ == '__main__':

    for split in ['train', 'dev', 'test']:
        step1_to_json(split)
        step2_mergeIDs_remove_redunt_rel(split)
        step3(split)








