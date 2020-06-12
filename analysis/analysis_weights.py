import json
import numpy as np
from copy import deepcopy
from misc.metrics import num2_p_r_f


def softmax(arr):
    arr = np.exp(arr)
    return arr/float(sum(arr))


def return_indices(weights_on_mp, threshold):
    indices = []
    for mp_idx, value in enumerate(weights_on_mp):
        if value >= threshold:
            indices.append(mp_idx)
    # sorted_indices = np.argsort(weights_on_mp)[::-1]
    # for mp_idx in sorted_indices:
    #     if weights_on_mp[mp_idx] >= threshold:
    #         indices.append(mp_idx)
    # if len(indices) >= 2:
    #     indices = indices[:-1]
    return indices


def get_sent_idx(ex_idx, indices_of_max_values, ht_comb_indices, head_indices, tail_indices, idx_to_span_pos, start_end_pos_to_sent_id, idx_to_headi_tailj=None):
    sent_ids = []
    for idx in indices_of_max_values:
        h, t = ht_comb_indices[idx]
        if idx_to_headi_tailj is not None:
            assert idx_to_headi_tailj[idx] == (h, t)
        _h_start, _h_end = map(int, idx_to_span_pos[str(head_indices[h])].split("_"))
        _t_start, _t_end = map(int, idx_to_span_pos[str(tail_indices[t])].split("_"))
        h_sent_idx = start_end_pos_to_sent_id[f"{ex_idx}_{_h_start}_{_h_end}"]
        t_sent_idx = start_end_pos_to_sent_id[f"{ex_idx}_{_t_start}_{_t_end}"]
        sent_ids.append(h_sent_idx)
        sent_ids.append(t_sent_idx)
    return set(sent_ids)


is_dev = True
seed = 2
truth_dev = 'E:/workspace/repo/DocRED/code/prepro_data/dev_dev.json'
truth_dev = json.load(open(truth_dev, 'r', encoding='utf8'))
pred_file_path = r'E:\DASFAA2020\experiments\weights\all\Ours-attenders\repro-seed%d_dev_dev.json' % seed
pred_file = json.load(open(pred_file_path, 'r', encoding='utf8'))
weights_file = r'E:\DASFAA2020\experiments\weights\all' \
               r'\Ours-attenders\repro-seed%d_dev_dev_attn_weights.json' % seed
weights = json.load(open(weights_file, 'r', encoding='utf8'))
start_end_pos_to_sent_id = {}
for ex_idx, ex in enumerate(truth_dev):
    for vertex in ex['vertexSet']:
        for mention in vertex:
            key = f"{ex_idx}_{mention['pos'][0]}_{mention['pos'][1]}"  # idx_startIdx_endIdx  -->  send idx
            start_end_pos_to_sent_id[key] = mention['sent_id']


def get_evidences(ex_idx, triplet, idx_to_span_pos):
    num_H, num_T, num_HT = sum(triplet['head_mentions_indices_mask']), sum(triplet['tail_mentions_indices_mask']), sum(triplet['ht_comb_mask'])
    assert num_H*num_T == num_HT
    head_indices = triplet['head_mentions_indices']
    tail_indices = triplet['tail_mentions_indices']
    ht_comb_indices = triplet['ht_comb_indices']
    weights_on_mp = np.array(triplet['weights_on_mp'])
    weights_on_mp = np.mean(weights_on_mp, axis=0)
    weights_on_mp = weights_on_mp[:num_HT]

    threshold = float(1.0)/num_HT
    indices_of_max_values = return_indices(weights_on_mp, threshold)
    # indices_of_max_values = list(range(num_HT))
    # if len(indices_of_max_values) >= 3:
    #     indices_of_max_values = indices_of_max_values[:-2]
    _pred_evids = get_sent_idx(ex_idx, indices_of_max_values, ht_comb_indices, head_indices, tail_indices, idx_to_span_pos, start_end_pos_to_sent_id)
    return _pred_evids


def integration_attn():
    num_pred_evi = 0
    num_tot_evidences = 0
    num_correct_evidence = 0
    for ex_idx, ex in enumerate(truth_dev):
        # truth_evidences
        truth_evids = {}
        for label in ex['labels']:
            r = label['r']
            h_idx = label['h']
            t_idx = label['t']
            truth_evids[(ex_idx, r, h_idx, t_idx)] = set(label['evidence'])
            num_tot_evidences += len(label['evidence'])
        # get tokens
        toks = []
        for sent in ex['sents']:
            toks.extend(sent)
        toks = [f"{tok}_[{__idx}]" for __idx, tok in enumerate(toks)]
        # print(' '.join(toks))
        #
        weight = weights[ex_idx]
        assert weight['index'] == ex_idx
        entity_span_indices = weight['entity_span_indices']
        idx_to_span_pos = weight['idx_to_span_pos']
        pred_triplets = weight['pred_triplets']
        if ex_idx % 100 == 0:
            print(f"{'='*20} {ex_idx} {'='*20}")
        for pred_idx, triplet in enumerate(pred_triplets):
            num_H, num_T, num_HT = sum(triplet['head_mentions_indices_mask']), sum(triplet['tail_mentions_indices_mask']), sum(triplet['ht_comb_mask'])
            assert num_H*num_T == num_HT
            # if num_H > 1 or num_T > 1:
            if True:
                # head_indices = triplet['head_mentions_indices']
                # print("head mentions")
                # for h in range(num_H):
                #     _start, _end = map(int, idx_to_span_pos[str(head_indices[h])].split("_"))
                # print(toks[_start: _end])
                # tail_indices = triplet['tail_mentions_indices']
                # print("tail mentions")
                # for t in range(num_T):
                #     _start, _end = map(int, idx_to_span_pos[str(tail_indices[t])].split("_"))
                # print(toks[_start: _end])
                # print(f"Rel: {triplet['rel']}")
                # print("-"*20)
                # indices_of_max_values = list(range(num_HT))
                _pred_evids = get_evidences(ex_idx, triplet, idx_to_span_pos)
                num_pred_evi += len(_pred_evids)
                _key = (ex_idx, triplet['r_idx'], triplet['h_t_idx'][0], triplet['h_t_idx'][1])
                if _key in truth_evids:
                    # num_tot_evidences += len(truth_evids[_key])
                    num_correct_evidence += len(truth_evids[_key] & _pred_evids)

                if ex_idx % 100 == 0:
                    head_indices = triplet['head_mentions_indices']
                    tail_indices = triplet['tail_mentions_indices']
                    ht_comb_indices = triplet['ht_comb_indices']
                    weights_on_mp = np.array(triplet['weights_on_mp'])
                    weights_on_mp = np.mean(weights_on_mp, axis=0)
                    weights_on_mp = weights_on_mp[:num_HT]
                    preds = []
                    for __i in range(num_HT):
                        h, t = ht_comb_indices[__i]
                        _h_start, _h_end = map(int, idx_to_span_pos[str(head_indices[h])].split("_"))
                        _t_start, _t_end = map(int, idx_to_span_pos[str(tail_indices[t])].split("_"))
                        preds.append((' '.join(toks[_h_start: _h_end]),
                                      triplet['rel'],
                                      ' '.join(toks[_t_start: _t_end]),
                                      weights_on_mp[__i]))
                    # print("{:<30}   <=    {}     =>   {:<30}   : {:4.4f}".format(' '.join(toks[_h_start: _h_end]),
                    #                                                              triplet['rel'],
                    #                                                              ' '.join(toks[_t_start: _t_end]),
                    #                                                              weights_on_mp[__i]))
                    preds.sort(reverse=True, key=lambda x: x[3])
                    for p in preds:
                        print("{:<30}   <=    {}     =>   {:<30}   : {:4.4f}".format(p[0], p[1], p[2], p[3]))
                    print()

            # print("="*50)
        # if ex_idx == 3:
        #     break

    evi_p, evi_r, evi_f1 = num2_p_r_f(num_tot_evidences, num_pred_evi, num_correct_evidence)
    print("evi: Precision: {:.2f}, ({}/{}); "
          "Recall: {:.2f}, ({}/{}); "
          "F1: {:.2f}".format(evi_p * 100, num_correct_evidence, num_pred_evi,
                              evi_r * 100, num_correct_evidence, num_tot_evidences,
                              evi_f1 * 100))
    print(f"seed {seed}")
    print("{:.2f} {:.2f} {:.2f}".format(evi_p * 100, evi_r * 100, evi_f1 * 100))


# integration_attn()


def mutual_attn():
    num_pred_evi = 0
    num_tot_evidences = 0
    num_correct_evidence = 0
    for ex_idx, ex in enumerate(truth_dev):
        # truth_evidences
        truth_evids = {}
        for label in ex['labels']:
            r = label['r']
            h_idx = label['h']
            t_idx = label['t']
            truth_evids[(ex_idx, r, h_idx, t_idx)] = set(label['evidence'])
            num_tot_evidences += len(label['evidence'])
        # get tokens
        toks = []
        for sent in ex['sents']:
            toks.extend(sent)
        toks = [f"{tok}_[{__idx}]" for __idx, tok in enumerate(toks)]
        # print(' '.join(toks))
        #
        weight = weights[ex_idx]
        assert weight['index'] == ex_idx
        entity_span_indices = weight['entity_span_indices']
        idx_to_span_pos = weight['idx_to_span_pos']
        pred_triplets = weight['pred_triplets']
        for pred_idx, triplet in enumerate(pred_triplets):
            num_H, num_T, num_HT = sum(triplet['head_mentions_indices_mask']), sum(triplet['tail_mentions_indices_mask']), sum(triplet['ht_comb_mask'])
            assert num_H*num_T == num_HT
            # if num_H > 1 or num_T > 1:
            if True:
                head_indices = triplet['head_mentions_indices']
                # print("head mentions")
                # for h in range(num_H):
                #     _start, _end = map(int, idx_to_span_pos[str(head_indices[h])].split("_"))
                # print(toks[_start: _end])
                tail_indices = triplet['tail_mentions_indices']
                # print("tail mentions")
                # for t in range(num_T):
                #     _start, _end = map(int, idx_to_span_pos[str(tail_indices[t])].split("_"))
                # print(toks[_start: _end])
                # print(f"Rel: {triplet['rel']}")
                # print("-"*20)
                ht_comb_indices = triplet['ht_comb_indices']
                hAt_weights = np.array(triplet['hAt_weights'])
                tAh_weights = np.array(triplet['tAh_weights'])
                # weights_on_mp = np.array(triplet['weights_on_mp'])
                # weights_on_mp = np.mean(weights_on_mp, axis=0)
                # weights_on_mp = weights_on_mp[:num_HT]  # drop padding mention pair

                headi_tailj_to_idx = {}
                idx_to_headi_tailj = {}
                ij_idx = 0
                attn_weights = np.zeros(num_HT)
                for hi in range(num_H):
                    for tj in range(num_T):
                        attn_weights[ij_idx] = np.sqrt(hAt_weights[hi, tj] * tAh_weights[tj, hi])
                        key = (hi, tj)
                        headi_tailj_to_idx[key] = ij_idx
                        idx_to_headi_tailj[ij_idx] = key
                        ij_idx += 1

                threshold = float(1.0)/num_HT
                indices_of_max_values = return_indices(softmax(attn_weights), threshold)
                # indices_of_max_values = list(range(num_HT))
                _pred_evids = get_sent_idx(ex_idx, indices_of_max_values, ht_comb_indices, head_indices, tail_indices, idx_to_span_pos, start_end_pos_to_sent_id, idx_to_headi_tailj)
                num_pred_evi += len(_pred_evids)
                _key = (ex_idx, triplet['r_idx'], triplet['h_t_idx'][0], triplet['h_t_idx'][1])
                if _key in truth_evids:
                    # num_tot_evidences += len(truth_evids[_key])
                    num_correct_evidence += len(truth_evids[_key] & _pred_evids)

                # preds = []
                # for __i in range(num_HT):
                #     h, t = ht_comb_indices[__i]
                #     _h_start, _h_end = map(int, idx_to_span_pos[str(head_indices[h])].split("_"))
                #     _t_start, _t_end = map(int, idx_to_span_pos[str(tail_indices[t])].split("_"))
                #     preds.append((' '.join(toks[_h_start: _h_end]),
                #                   triplet['rel'],
                #                   ' '.join(toks[_t_start: _t_end]),
                #                   weights_on_mp[__i]))
                # print("{:<30}   <=    {}     =>   {:<30}   : {:4.4f}".format(' '.join(toks[_h_start: _h_end]),
                #                                                              triplet['rel'],
                #                                                              ' '.join(toks[_t_start: _t_end]),
                #                                                              weights_on_mp[__i]))
                # preds.sort(reverse=True, key=lambda x: x[3])
                # for p in preds:
                #     print("{:<30}   <=    {}     =>   {:<30}   : {:4.4f}".format(p[0], p[1], p[2], p[3]))

            # print("="*50)
        # if ex_idx == 3:
        #     break

    evi_p = 1.0 * num_correct_evidence / num_pred_evi if num_pred_evi > 0 else 0
    evi_r = 1.0 * num_correct_evidence / num_tot_evidences
    if evi_p + evi_r == 0:
        evi_f1 = 0
    else:
        evi_f1 = 2.0 * evi_p * evi_r / (evi_p + evi_r)
    print("evi: Precision: {:.2f}, ({}/{}); "
          "Recall: {:.2f}, ({}/{}); "
          "F1: {:.2f}".format(evi_p * 100, num_correct_evidence, num_pred_evi,
                              evi_r * 100, num_correct_evidence, num_tot_evidences,
                              evi_f1 * 100))
    print(f"seed {seed}, threshold: {threshold}")
    print("{:.2f} {:.2f} {:.2f}".format(evi_p * 100, evi_r * 100, evi_f1 * 100))


# integration_attn()


def gen_test_evidences():
    pred_evidences = {}
    num_pred_evi = 0
    num_tot_evidences = 0
    num_correct_evidence = 0
    for ex_idx, ex in enumerate(truth_dev):
        if is_dev:
            truth_evids = {}
            for label in ex['labels']:
                r = label['r']
                h_idx = label['h']
                t_idx = label['t']
                truth_evids[(ex_idx, r, h_idx, t_idx)] = set(label['evidence'])
                num_tot_evidences += len(label['evidence'])
        weight = weights[ex_idx]
        assert weight['index'] == ex_idx
        idx_to_span_pos = weight['idx_to_span_pos']
        pred_triplets = weight['pred_triplets']
        for pred_idx, triplet in enumerate(pred_triplets):
            _pred_evids = get_evidences(ex_idx, triplet, idx_to_span_pos)

            _key = f"{ex_idx}_{triplet['h_t_idx'][0]}_{triplet['h_t_idx'][1]}_{triplet['r_idx']}"
            pred_evidences[_key] = list(_pred_evids)

            if is_dev:
                num_pred_evi += len(_pred_evids)
                _key = (ex_idx, triplet['r_idx'], triplet['h_t_idx'][0], triplet['h_t_idx'][1])
                if _key in truth_evids:
                    num_correct_evidence += len(truth_evids[_key] & _pred_evids)
    if is_dev:
        evi_p, evi_r, evi_f1 = num2_p_r_f(num_tot_evidences, num_pred_evi, num_correct_evidence)
        print("{:.2f} {:.2f} {:.2f}".format(evi_p * 100, evi_r * 100, evi_f1 * 100))
    out_preds = []
    for triplet_idx, pred_triplet in enumerate(pred_file):
        out_ex = deepcopy(pred_triplet)
        _key = f"{pred_triplet['index']}_{pred_triplet['h_idx']}_{pred_triplet['t_idx']}_{pred_triplet['r_idx']}"
        out_ex['evidence'] = pred_evidences[_key]
        out_preds.append(out_ex)
    json.dump(out_preds, open(pred_file_path + '_evids.json', 'w', encoding='utf8'), ensure_ascii=False, indent=2)


# gen_test_evidences()






















