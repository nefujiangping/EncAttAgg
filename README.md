EncAttAgg
---------
This is the source code for ICKG 2020 paper "[Improving Document-level Relation Extraction via Contextualizing Mention Representations and Weighting Mention Pairs](https://conferences.computer.org/ickg/pdfs/ICKG2020-66r9RP2mQIZywMjHhQVtDI/815600a305/815600a305.pdf)"

We propose an effective **Enc**oder-**Att**ender-**Agg**regator (EncAttAgg) model for ducument-level RE. This model introduced two attenders to tackle two problems: 1) We introduce a mutual attender layer to efficiently obtain the entity-pair-specific mention representations.
2) We introduce an integration attender to weight mention pairs of a target entity pair.

## Requirements
+ python 3.7.4
+ pytorch 1.3.0
+ [allennlp](https://allennlp.org/tutorials) 0.9.0
+ yaml
+ tensorboardX
## Datasets
+ [Chemical-Disease Relations dataset (CDR)](https://github.com/patverga/bran/tree/master/data/cdr) [1,2]. CDR consists of 1500 abstracts of PubMed, which is in the domain of biomedicine considering 2 entity types **Chemical** and **Disease** and one **Chemical-Induced Disease** relation type. It is split into three equally sized sets for training, development and testing.

+ [DocRED](https://github.com/thunlp/DocRED) [3]. DocRED is a large-scaled document-level dataset presented by Yao et al. for general purpose RE, which contains 5053 documents and is split into 3053, 1000 and 1000 for training, development and testing. The dataset contains 6 general entity types and 96 relation types.

## Usage
### train our model
```shell
python train.py --param_file ${param_file} --gpu 0,1 --exp_id EncAttAgg --randseed_no 0
```
### test our model
```shell
python test.py --param_file ${param_file} --gpu 0 --exp_id EncAttAgg
```

## Results and Hyper-parameters
Main results on DocRED/CDR and the corresponding hyper-parameters are shown below.
Please refer to the [paper](https://conferences.computer.org/ickg/pdfs/ICKG2020-66r9RP2mQIZywMjHhQVtDI/815600a305/815600a305.pdf) for more details of the experiments.

+ Results
![main_results](images/main_results.png)

+ Hyper-parameters
![hyperparams](images/hyperparams.jpg)

## References
1. Wei, Chih-Hsuan and Peng, Yifan and Leaman, R. and Davis, Allan Peter and Mattingly, C.J. and Li, J. and Wiegers, T.C. and lu, Zhiyong. Overview of the BioCreative V chemical disease relation (CDR) task
2. Li, Jiao and Sun, Yueping and Johnson, Robin J. and Sciaky, Daniela and Wei, Chih-Hsuan and Leaman, Robert and Davis, Allan Peter and Mattingly, Carolyn J. and Wiegers, Thomas C. and Lu, Zhiyong. BioCreative V CDR task corpus: a resource for chemical disease relation extraction
3. Yao, Yuan  and Ye, Deming  and Li, Peng  and Han, Xu  and Lin, Yankai  and Liu, Zhenghao  and Liu, Zhiyuan  and Huang, Lixin  and Zhou, Jie  and Sun, Maosong. DocRED: A Large-Scale Document-Level Relation Extraction Dataset.


