import json
import numpy as np

data_dir = 'E:/workspace/repo/DocRED/code/prepro_data/'


def get_dist(h_vertex, t_vertex):
    min_d = 620
    for h_m in h_vertex:
        for t_m in t_vertex:
            d = abs(h_m['pos'][0] - t_m['pos'][0])
            min_d = min(min_d, d)
    return min_d


def number_of_pos_neg_triplets():
    from collections import defaultdict
    for split in ['train', 'dev', 'test']:
        counter = 0
        all_counter = 0
        for ex in json.load(open(data_dir+f"dev_{split}.json")):
            labels = defaultdict(list)
            for label in ex['labels']:
                labels[(label['h'], label['t'])].append(label['r'])
            counter += len(labels)
            all_counter += (len(ex['vertexSet'])*(len(ex['vertexSet'])-1))

        print(f"{split}: all: {all_counter}, pos: {counter}, neg: {all_counter-counter}")


def sent_avg_len():
    sent_lens = []
    for split in ['train', 'dev', 'test']:
        for ex in json.load(open(data_dir+f"dev_{split}.json")):
            sent_lens.extend([len(sent) for sent in ex['sents']])
    print(np.mean(sent_lens))  # 24.88


def coref_short_proportion():
    true_file = json.load(open(data_dir+f"dev_train.json"))
    all_counter = 0
    coref = [0]*2
    dist = [0]*2
    for idx, ins in enumerate(true_file):
        vertexSet = ins['vertexSet']
        for label in ins['labels']:
            key = f"{idx}_{label['h']}_{label['t']}"
            if len(vertexSet[label['h']]) > 1 or len(vertexSet[label['t']]) > 1:
                coref[0] += 1
            else:
                coref[1] += 1
            if 0 <= get_dist(vertexSet[label['h']], vertexSet[label['t']]) <= 25:
                dist[0] += 1
            else:
                dist[1] += 1
            all_counter += 1

    print(f"all: {all_counter}, {coref[0]/float(all_counter)}({coref[0]}/{all_counter})")
    print(f"all: {all_counter}, {dist[0]/float(all_counter)}({dist[0]}/{all_counter})")
    # dev
    # all: 12323, 0.5931185587925019(7309/12323) 60% have co-references
    # all: 12323, 0.6469203927615029(7972/12323) 65% short; 35% long distance
    # train
    # all: 38180, 0.5930330015715034(22642/38180) 60%
    # all: 38180, 0.639732844421163(24425/38180)  64%


def entity_type_combinations():
    h_t_entity_type_combs = set()
    for split in ['train', 'dev']:
        data = json.load(open(f"{data_dir}dev_{split}.json"))
        for ex in data:
            v = ex['vertexSet']
            for label in ex['labels']:
                h_idx, t_idx = label['h'], label['t']
                for h_m in v[h_idx]:
                    for t_m in v[t_idx]:
                        h_t_entity_type_combs.add((h_m['type'], t_m['type']))
    h_t_entity_type_combs = list(map(list, h_t_entity_type_combs))
    print(len(h_t_entity_type_combs))
    json.dump(h_t_entity_type_combs, open(f'{data_dir}DocRED_h_t_entity_type_combs_{len(h_t_entity_type_combs)}.json', 'w'), indent=2)


entity_type_combinations()
# num_vertex = []
# for split in ['train', 'dev', 'test']:
#     for ex in json.load(open(data_dir+f"dev_{split}.json")):
#         num_vertex.append(len(ex['vertexSet']))
#
# print(f"avg: {np.mean(num_vertex)}")

# number_of_pos_neg_triplets()
















