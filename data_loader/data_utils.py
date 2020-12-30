import json


ENTITY_START = 'S'
ENTITY_END = '/S'

marker_fmt = "{S_or_E}-{entity_type}"


def get_marker_to_token(model_name: str, meta_dir='./data'):
    """
    :param entity_types: list, 实体类型列表，如 ['组织', '飞机', '时间', '导弹', '机载计算机网络', '机载传感器', '座舱显控']
    :return: marker_to_bert_token: dict, 头尾实体标记 与 bert-base-chinese vocab token 的映射；
        key 为实体标记，value 为 Bert token
        Ref: https://mirrors.tuna.tsinghua.edu.cn/hugging-face-models/bert-base-chinese-vocab.txt
    """
    entity_types = json.load(open(f'{meta_dir}/entity_types.json'))
    unused_tokens = list(json.load(open(f'{meta_dir}/unused_tokens.json', 'r', encoding='utf8'))[model_name].keys())
    num_least = 2 * len(entity_types)
    if len(unused_tokens) < num_least:
        raise Exception(f"number of unused tokens (in `meta/unused_tokens.json`) is not enough, please add more. "
                        f"At least {num_least} unused tokens, now {len(unused_tokens)}")

    marker_to_bert_token = dict()
    cnt = 0
    for H_or_T in [ENTITY_START, ENTITY_END]:
        for et in entity_types:
            marker = marker_fmt.format(S_or_E=H_or_T, entity_type=et)
            marker_to_bert_token[marker] = unused_tokens[cnt]
            cnt += 1
    return marker_to_bert_token
