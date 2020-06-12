import re
import os
import numpy as np

"""
2019/12/14 13:18  | epoch 9  | Ign F1: 56.50, Precision: 59.20; Recall: 54.03
2019/12/14 13:19  | epoch 11 | Ign F1: 61.85, Precision: 56.52; Recall: 68.29
2019/12/14 0:47   | epoch 0  | Ign F1: 10.79, Precision: 74.70; Recall: 5.82
"""
line = "2019/12/14 13:19  | epoch 11 | Ign F1: 61.85, Precision: 56.52; Recall: 68.29"
num_maximum = 5
num_seeds = 5
num_epoch = 25

log_dir = 'E:/DASFAA2020/experiments/CDR_results/logs-filterTest'
# models = [
            # 'CDR_BaselineV2-Bert_onAll-PubMed',
            # 'CDR_bran-Bert_onAll-PubMed',
            # 'CDR_NONE-Bert_onAll-PubMed',
            # 'CDR_Ours-Bert_onAll-PubMed',
            # 'CDR_NONE_onAll-PubMed',
            # 'CDR_Ours_onAll-PubMed',
            # 'CDR_bran-noBert-onAll-PubMed',
            # 'CDR_NONE_onAll-noAVG_Param-PubMed',
            # 'CDR_Ours_onAll-noAVG_Param-PubMed',
            # 'CDR_bran-noBert-UpperCase-onAll-PubMed',
            # 'CDR_NONE_onAll-UpperCase-PubMed',
            # 'CDR_Ours_onAll-UpperCase-PubMed'
# ]
models = [
    # 'CDR_BaselineV2-Bert_onAll-filterTest',
    # 'CDR_bran-Bert-onAll-filterTest',
    # 'CDR_NONE-Bert_onAll-filterTest',
    # 'CDR_Ours-Bert_onAll-filterTest',
    # 'CDR_BaselineV2_onAll-filterTest-PubMed',
    # 'CDR_bran-onAll-filterTest-PubMed',
    # 'CDR_NONE_onAll-filterTest-PubMed',
    # 'CDR_Ours_1layer',
    # 'CDR_BERT_NONE_Ours'
    'CDR_PubMed_GCNN',
    'CDR_Bert_GCNN'
]
pattern = r'\d{4}/\d{1,2}/\d{1,2}\s\d{1,2}:\d{1,2}\s{2,}\|\sepoch\s(\d{1,2})\s{1,2}\|\sIgn\sF1:\s(\d{1,2}\.\d{2}),\sPrecision:\s(\d{1,3}\.\d{2});\sRecall:\s(\d{1,2}\.\d{2})'

for model_idx in range(len(models)):
    model_name = models[model_idx]
    # print(model_name)
    prfs_of_this_model = []
    for seed_idx in range(num_seeds):
    # for seed_idx in [0, 3, 4]:
        epoch_counter = 0
        prfs = []
        file_name = f"{model_name}-seed{seed_idx}"
        with open(os.path.join(log_dir, model_name, file_name)) as inf:
            for line in inf:
                line = line.strip()
                if line:
                    groups = re.match(pattern, line)
                    if groups is not None:
                        _epoch = int(groups.group(1))
                        _f1 = float(groups.group(2))
                        _precision = float(groups.group(3))
                        _recall = float(groups.group(4))
                        assert 0 <= _epoch < num_epoch
                        assert 0.0 <= _f1 <= 100.0
                        assert 0.0 <= _precision <= 100.0
                        assert 0.0 <= _recall <= 100.0
                        assert _epoch == epoch_counter, f"{line}\n{_epoch} {epoch_counter}"
                        if _precision + _recall > 0:
                            _computed_f1 = 2.0*_precision*_recall / (_precision + _recall)
                            if abs(_computed_f1 - _f1) >= 0.1:
                                print("ERROR 1===")
                                print(line)
                                print(_epoch, _f1, _precision, _recall)
                                print(model_name, file_name, epoch_counter)
                                print("===ERROR 1")
                        prfs.append([_epoch, _precision, _recall, _f1])
                        epoch_counter += 1
        # print(f"epoch_counter: {epoch_counter}")

        # for _ii in range(num_epoch):
        #     assert prfs[_ii][0] == _ii
            # print(prfs[_ii])
        prfs.sort(key=lambda item: item[3], reverse=True)
        print(prfs[:num_maximum])
        prfs_of_this_model.extend(prfs[:num_maximum])
        if epoch_counter != num_epoch:
            print("ERROR 2===")
            print(line)
            print(model_name, file_name, epoch_counter, num_epoch)
            print("===ERROR 2")

    assert len(prfs_of_this_model) == num_seeds*num_maximum
    prfs_of_this_model = np.array(prfs_of_this_model)[:, 1:]
    print(np.mean(prfs_of_this_model, axis=0))



