Data pre-processing
--------------

# DocRED
- Note that [transformers](https://github.com/huggingface/transformers/issues) is required.
- Download DocRED meta data to `prepro_data/DocRED`, from [TsinghuaCloud](https://cloud.tsinghua.edu.cn/d/99e1c0805eb64736af95/) or [GoogleDrive](https://drive.google.com/drive/folders/1Ri3LIILKKBi3aBJjUVCOBpGX5PpONHRK).
- Make sure that these 7 files `[train_distant.json, train_annotated.json, dev.json, test.json, rel2id.json, word2id.json, ner2id.json]` are in the directory of `prepro_data/DocRED`.
- Then run:
    ```shell
    python gen_data.py --in_path prepro_data/DocRED --out_path prepro_data/DocRED --pretrained_model_name_or_path bert-base-cased
    ```
