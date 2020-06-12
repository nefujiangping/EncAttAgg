import json
from typing import List, Dict, Tuple, Set


class Accuracy(object):
    def __init__(self) -> None:
        self.correct = 0
        self.total = 0

    def add(self, is_correct) -> None:
        self.total += 1
        if is_correct:
            self.correct += 1

    def get(self) -> float:
        if self.total == 0:
            return 0.0
        else:
            return float(self.correct) / self.total

    def clear(self) -> None:
        self.correct = 0
        self.total = 0


def num2_p_r_f(num_true_all, num_pred_all, pred_correct) -> Tuple[float, float, float]:
    # print(num_true_all, num_pred_all, pred_correct)
    precision = (1.0*pred_correct / num_pred_all) if num_pred_all > 0 else 0.
    recall = (1.0*pred_correct / num_true_all) if num_true_all > 0 else 0.
    f1 = (2.0*precision*recall / (precision + recall)) if (precision + recall) > 0 else 0.
    return precision, recall, f1


def __get__dev_true() -> Tuple[Set, Dict]:
    recall = 0
    dev_true_rel_facts = set()
    with open('/home/jp/workspace2/datasets/CDR_PubMed_with_KBID/raw_dev.json', 'r', encoding='utf8') as in_fi:
        for line in in_fi:
            ex = json.loads(line.strip())
            recall += len(ex['rels'])
            for rel in ex['rels']:
                dev_true_rel_facts.add((ex['title'], rel[2], rel[3]))  # (DOC_KB_ID, h_KB_ID, t_KB_ID)
    assert recall == 1012
    assert len(dev_true_rel_facts) == 1012
    dev_idx_to_IDs = {}
    for ex in json.load(open('/home/jp/workspace2/datasets/CDR_PubMed_with_KBID/dev_train.json', 'r', encoding='utf8'))[500:]:
        for vertex_idx, vertex in enumerate(ex['vertexSet']):
            key = f"{ex['title']}_{vertex_idx}"
            dev_idx_to_IDs[key] = vertex[0]['KB_ID'].split('|')
    return dev_true_rel_facts, dev_idx_to_IDs


def __get__test_true() -> Tuple[Set, Dict]:
    recall = 0
    test_true_rel_facts = set()
    with open('/home/jp/workspace2/datasets/CDR_PubMed_with_KBID/raw_test.json', 'r', encoding='utf8') as in_fi:
        for line in in_fi:
            ex = json.loads(line.strip())
            recall += len(ex['rels'])
            for rel in ex['rels']:
                test_true_rel_facts.add((ex['title'], rel[2], rel[3]))  # (DOC_idx, h_ID, t_ID)
    assert recall == 1066
    assert len(test_true_rel_facts) == 1066
    test_idx_to_IDs = {}
    for ex in json.load(open('/home/jp/workspace2/datasets/CDR_PubMed_with_KBID/dev_test.json', 'r', encoding='utf8')):
        for vertex_idx, vertex in enumerate(ex['vertexSet']):
            key = f"{ex['title']}_{vertex_idx}"
            test_idx_to_IDs[key] = vertex[0]['KB_ID'].split('|')
    return test_true_rel_facts, test_idx_to_IDs


def __p_r_f__(results_file, true_rel_facts, idx_to_IDs) -> Tuple[float, float, float]:
    preds = json.load(open(results_file))
    pred_rel_facts = set()
    for pred_idx, pred in enumerate(preds):
        doc_idx = pred['KB_ID']
        h_idx = pred['h_idx']
        t_idx = pred['t_idx']
        h_IDs = idx_to_IDs[f"{doc_idx}_{h_idx}"]
        t_IDs = idx_to_IDs[f"{doc_idx}_{t_idx}"]
        for hID in h_IDs:
            for tID in t_IDs:
                pred_rel_facts.add((str(doc_idx), hID, tID))
    json.dump(list(pred_rel_facts), open(results_file[:-5] + '_facts.json', 'w'), ensure_ascii=False, indent=2)
    precision, recall, f1 = num2_p_r_f(len(true_rel_facts), len(pred_rel_facts), len(true_rel_facts & pred_rel_facts))

    return precision, recall, f1


# For CDR
def p_r_f(results_file, split='dev') -> Tuple[float, float, float]:
    assert split in ['dev', 'test']
    if split == 'dev':
        true_rel_facts, idx_to_IDs = __get__dev_true()
        return __p_r_f__(results_file, true_rel_facts, idx_to_IDs)
    elif split == 'test':
        true_rel_facts, idx_to_IDs = __get__test_true()
        return __p_r_f__(results_file, true_rel_facts, idx_to_IDs)


def coref_vs_non_coref_performance(true_file, test_result: List) -> Dict:
    """

    :param true_file:
    :param test_result:
    :return:
        f1 = {
            'all_f1',
            'all_coref',
            'all_non-coref',
            'ign_f1',
            'ign_coref',
            'ign_non-coref'
        }
    """
    true_dict = {}
    has_coreferences = {}
    for idx, ins in enumerate(true_file):
        vertexSet = ins['vertexSet']
        true_dict[idx] = set()
        for label in ins['labels']:
            true_dict[idx].add((label['h'], label['r'], label['t']))  # (h_index, rel_idx, t_idx)
            key = f"{idx}_{label['h']}_{label['t']}"
            has_coreferences[key] = (len(vertexSet[label['h']]) > 1 or len(vertexSet[label['t']]) > 1)
        for na_triple in ins['na_triple']:
            key = f"{idx}_{na_triple[0]}_{na_triple[1]}"
            has_coreferences[key] = (len(vertexSet[na_triple[0]]) > 1 or len(vertexSet[na_triple[1]]) > 1)

    pred_dict = {}
    train_rel_facts = []
    for ins in test_result:
        if isinstance(ins, Dict):
            key_idx = 'index'
            h_idx = 'h_idx'
            t_idx = 't_idx'
            r_idx = 'r_idx'
            in_train = 'intrain'
        elif isinstance(ins, tuple):
            key_idx = -4
            h_idx = -3
            t_idx = -2
            r_idx = -1
            in_train = 2
        else:
            raise Exception('`test_result` neither `List[Dict]` nor `List[tuple]` ERROR.')
        if ins[key_idx] not in pred_dict:
            pred_dict[ins[key_idx]] = set()
            pred_dict[ins[key_idx]].add((ins[h_idx], ins[r_idx], ins[t_idx]))
        else:
            pred_dict[ins[key_idx]].add((ins[h_idx], ins[r_idx], ins[t_idx]))
        if ins[in_train]:
            train_rel_facts.append((ins[key_idx], ins[h_idx], ins[r_idx], ins[t_idx]))

    num_true_all = 0
    num_pred_all = 0
    pred_correct = 0
    types_num_true_all = [0]*4
    types_num_pred_all = [0]*4
    types_pred_correct = [0]*4
    for idx, labels in true_dict.items():
        if idx not in pred_dict:
            _pred = set()
        else:
            _pred = pred_dict[idx]
        correct_pred_triplets = labels & _pred
        pred_correct += len(correct_pred_triplets)
        num_true_all += len(labels)
        num_pred_all += len(_pred)
        for triplet in _pred:
            key = f"{idx}_{triplet[0]}_{triplet[2]}"
            if has_coreferences[key]:
                types_num_pred_all[0] += 1
            else:
                types_num_pred_all[1] += 1
            to_check = (idx, triplet[0], triplet[1], triplet[2])
            if not (to_check in train_rel_facts):
                if has_coreferences[key]:
                    types_num_pred_all[2] += 1
                else:
                    types_num_pred_all[3] += 1
        for triplet in labels:
            key = f"{idx}_{triplet[0]}_{triplet[2]}"
            if has_coreferences[key]:
                types_num_true_all[0] += 1
                types_num_true_all[2] += 1
            else:
                types_num_true_all[1] += 1
                types_num_true_all[3] += 1
        for triplet in correct_pred_triplets:
            key = f"{idx}_{triplet[0]}_{triplet[2]}"
            if has_coreferences[key]:
                types_pred_correct[0] += 1
            else:
                types_pred_correct[1] += 1
            to_check = (idx, triplet[0], triplet[1], triplet[2])
            if not (to_check in train_rel_facts):
                if has_coreferences[key]:
                    types_pred_correct[2] += 1
                else:
                    types_pred_correct[3] += 1
    f1 = {}
    _, _, f1['all_f1'] = num2_p_r_f(sum(types_num_true_all[:2]), sum(types_num_pred_all[:2]), sum(types_pred_correct[:2]))
    _types = ['all_coref', 'all_non-coref']
    for i in range(2):
        precision, recall, f1[_types[i]] = num2_p_r_f(types_num_true_all[i], types_num_pred_all[i], types_pred_correct[i])

    ign_p = (1.0*(types_pred_correct[2] + types_pred_correct[3]) / (types_num_pred_all[2] + types_num_pred_all[3])) if (types_num_pred_all[2] + types_num_pred_all[3]) > 0 else 0.
    ign_r = (1.0*(types_pred_correct[0] + types_pred_correct[1]) / (types_num_true_all[0] + types_num_true_all[1])) if (types_num_true_all[0] + types_num_true_all[1]) > 0 else 0.
    ign_f = (2*ign_p*ign_r/(ign_p+ign_r)) if (ign_p+ign_r) > 0. else 0.
    f1['ign_f1'] = ign_f

    p = (1.0*types_pred_correct[2] / types_num_pred_all[2]) if types_num_pred_all[2] > 0 else 0.
    r = (1.0*types_pred_correct[0] / types_num_true_all[0]) if types_num_true_all[0] > 0 else 0.
    f1['ign_coref'] = (2*p*r/(p+r)) if (p+r) > 0 else 0.

    p = (1.0*types_pred_correct[3] / types_num_pred_all[3]) if types_num_pred_all[3] > 0 else 0.
    r = (1.0*types_pred_correct[1] / types_num_true_all[1]) if types_num_true_all[1] > 0 else 0.
    f1['ign_non-coref'] = (2*p*r/(p+r)) if (p+r) > 0 else 0.
    return f1


def diff_dist_performance(true_file, test_result: List, ret_overall=False) -> Dict:
    """

    :param true_file:
    :param test_result:
    :param ret_overall: Whether return 'all_f1' and 'ign_f1'
    :return:
        f1 = {
            'all_f1',
            'all_dist:1-25'
            'all_dist:26+'
            'ign_f1'
            'ign_dist:1-25'
            'ign_dist:26+'
        }
    """
    def to_bin_idx(dist):
        ret = 0
        if 1 <= dist <= 25:
            ret = 0
        else:
            ret = 1
        return ret

    true_dict = {}
    dist_dict = {}
    for idx, ins in enumerate(true_file):
        true_dict[idx] = set()
        for label in ins['labels']:
            true_dict[idx].add((label['h'], label['r'], label['t']))  # (h_index, rel_idx, t_idx)
            key = f"{idx}_{label['h']}_{label['t']}"
            dist_dict[key] = ins['pair_dist'][label['h']][label['t']]
        for na_triple in ins['na_triple']:
            key = f"{idx}_{na_triple[0]}_{na_triple[1]}"
            dist_dict[key] = dist_dict[key] = ins['pair_dist'][na_triple[0]][na_triple[1]]

    pred_dict = {}
    train_rel_facts = []
    for ins in test_result:
        if isinstance(ins, Dict):
            key_idx = 'index'
            h_idx = 'h_idx'
            t_idx = 't_idx'
            r_idx = 'r_idx'
            in_train = 'intrain'
        elif isinstance(ins, tuple):
            key_idx = -4
            h_idx = -3
            t_idx = -2
            r_idx = -1
            in_train = 2
        else:
            raise Exception('`test_result` neither `List[Dict]` nor `List[tuple]` ERROR.')
        if ins[key_idx] not in pred_dict:
            pred_dict[ins[key_idx]] = set()
            pred_dict[ins[key_idx]].add((ins[h_idx], ins[r_idx], ins[t_idx]))
        else:
            pred_dict[ins[key_idx]].add((ins[h_idx], ins[r_idx], ins[t_idx]))
        if ins[in_train]:
            train_rel_facts.append((ins[key_idx], ins[h_idx], ins[r_idx], ins[t_idx]))

    num_true_all = 0
    num_pred_all = 0
    pred_correct = 0
    types_num_true_all = [0]*4
    types_num_pred_all = [0]*4
    types_pred_correct = [0]*4
    for idx, labels in true_dict.items():
        if idx not in pred_dict:
            _pred = set()
        else:
            _pred = pred_dict[idx]
        correct_pred_triplets = labels & _pred
        pred_correct += len(correct_pred_triplets)
        num_true_all += len(labels)
        num_pred_all += len(_pred)
        for triplet in _pred:
            key = f"{idx}_{triplet[0]}_{triplet[2]}"
            if to_bin_idx(dist_dict[key]) == 0:
                types_num_pred_all[0] += 1
            else:
                types_num_pred_all[1] += 1
            to_check = (idx, triplet[0], triplet[1], triplet[2])
            if not (to_check in train_rel_facts):
                if to_bin_idx(dist_dict[key]) == 0:
                    types_num_pred_all[2] += 1
                else:
                    types_num_pred_all[3] += 1
        for triplet in labels:
            key = f"{idx}_{triplet[0]}_{triplet[2]}"
            if to_bin_idx(dist_dict[key]) == 0:
                types_num_true_all[0] += 1
                types_num_true_all[2] += 1
            else:
                types_num_true_all[1] += 1
                types_num_true_all[3] += 1
        for triplet in correct_pred_triplets:
            key = f"{idx}_{triplet[0]}_{triplet[2]}"
            if to_bin_idx(dist_dict[key]) == 0:
                types_pred_correct[0] += 1
            else:
                types_pred_correct[1] += 1
            to_check = (idx, triplet[0], triplet[1], triplet[2])
            if not (to_check in train_rel_facts):
                if to_bin_idx(dist_dict[key]) == 0:
                    types_pred_correct[2] += 1
                else:
                    types_pred_correct[3] += 1
    f1 = {}
    if ret_overall:
        _, _, f1['all_f1'] = num2_p_r_f(sum(types_num_true_all[:2]), sum(types_num_pred_all[:2]), sum(types_pred_correct[:2]))
    _types = ['all_dist:1-25', 'all_dist:26+']
    for i in range(2):
        precision, recall, f1[_types[i]] = num2_p_r_f(types_num_true_all[i], types_num_pred_all[i], types_pred_correct[i])

    if ret_overall:
        ign_p = (1.0*(types_pred_correct[2] + types_pred_correct[3]) / (types_num_pred_all[2] + types_num_pred_all[3])) if (types_num_pred_all[2] + types_num_pred_all[3]) > 0 else 0.
        ign_r = (1.0*(types_pred_correct[0] + types_pred_correct[1]) / (types_num_true_all[0] + types_num_true_all[1])) if (types_num_true_all[0] + types_num_true_all[1]) > 0 else 0.
        ign_f = (2*ign_p*ign_r/(ign_p+ign_r)) if (ign_p+ign_r) > 0. else 0.
        f1['ign_f1'] = ign_f

    p = (1.0*types_pred_correct[2] / types_num_pred_all[2]) if types_num_pred_all[2] > 0 else 0.
    r = (1.0*types_pred_correct[0] / types_num_true_all[0]) if types_num_true_all[0] > 0 else 0.
    f1['ign_dist:1-25'] = (2*p*r/(p+r)) if (p+r) > 0 else 0.

    p = (1.0*types_pred_correct[3] / types_num_pred_all[3]) if types_num_pred_all[3] > 0 else 0.
    r = (1.0*types_pred_correct[1] / types_num_true_all[1]) if types_num_true_all[1] > 0 else 0.
    f1['ign_dist:26+'] = (2*p*r/(p+r)) if (p+r) > 0 else 0.
    return f1


if __name__ == '__main__':
    precision, recall, f1 = p_r_f('/home/jp/workspace2/DocRED/CDR_Ours_onAll-UpperCase-PubMed-seed0_dev_test.json', 'test')
    print(precision, recall, f1)
