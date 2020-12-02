Data Pre-processing
--------------

# DocRED
1. Download metadata from [TsinghuaCloud](https://cloud.tsinghua.edu.cn/d/99e1c0805eb64736af95/) or [GoogleDrive](https://drive.google.com/drive/folders/1Ri3LIILKKBi3aBJjUVCOBpGX5PpONHRK) to `DATA_DIR`
2. Set `DATA_DIR`: `export DATA_DIR=/xxx/EncAttAgg/data/DocRED`
3. run `gen_data.py` (copied from repo [thunlp/DocRED](https://github.com/thunlp/DocRED/blob/master/code/gen_data.py)):
    ```shell script
    python3 DocRED/gen_data.py --in_path $DATA_DIR --out_path $DATA_DIR
    ```
4. Dump train/dev/test document representations to h5py files:
    - prepare env:
        + python 3 (tested on 3.7)
        + tensorflow-gpu (tested on 1.11 or 1.15)
    - clone [bert repo](https://github.com/google-research/bert) to `<bert-codes-root-dir>`
    - download [Bert Base Cased](https://storage.googleapis.com/bert_models/2018_10_18/cased_L-12_H-768_A-12.zip)
    - copy `DocRED/extract_features_DocRED.py` to `<bert-codes-root-dir>`, then run:
    ```shell script
        export CUDA_VISIBLE_DEVICES=4
        export BERT_BASE_DIR=/path/to/cased_L-12_H-768_A-12
        python extract_features_DocRED.py \
        --data_dir=$DATA_DIR \
        --vocab_file=$BERT_BASE_DIR/vocab.txt \
        --bert_config_file=$BERT_BASE_DIR/bert_config.json \
        --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
        --layers=-1 \
        --max_seq_length=512 \
        --batch_size=4
    ```
   - `768_dev_train.h5/768_dev_dev.h5/768_dev_test.h5` are dumped to `${DATA_DIR}`, and those files
   will be loaded for train/dev/test

# CDR
Raw data of CDR (CDR_DevelopmentSet.PubTator.txt.gz, CDR_TestSet.PubTator.txt.gz, CDR_TrainingSet.PubTator.txt.gz) are downloaded from repo [patverga/bran](https://github.com/patverga/bran/tree/master/data/cdr).

The pre-processed data are uploaded to [BaiDu Drive](https://pan.baidu.com/s/1uwm88j9wjSQ5Q9phx9UekQ) (CODE: sth4). Please download these data to directory `data/CDR`.
