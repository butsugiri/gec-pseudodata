# pseudodata-for-gec

This is the official repository of following paper: 


```
An Empirical Study of Incorporating Pseudo Data into Grammatical Error Correction
Shun Kiyono, Jun Suzuki, Masato Mita, Tomoya Mizumoto, Kentaro Inui
2019 Conference on Empirical Methods in Natural Language Processing and 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP2019), 2019 
```

* [pdf](https://www.aclweb.org/anthology/D19-1119.pdf)
* [bib](https://www.aclweb.org/anthology/D19-1119.bib)

## Requirements

- Python 3.6 or higher
- PyTorch (version 1.0.1.post2 is recommended)
- [blingfire](https://github.com/microsoft/BlingFire) (for preprocessing - sentence splitting)
- [spaCy](https://spacy.io/) (for preprocessing - tokenization)
- [subword-nmt](https://github.com/rsennrich/subword-nmt) (for splitting the data into subwords)
- [fairseq](https://github.com/pytorch/fairseq) (I used commit ID: [3658fa3](https://github.com/pytorch/fairseq/commit/3658fa329b8cb987d951b2e38ec86c44b9e1fea5) for all experiments. I strongly recommend sticking with the same commit ID.)

## Resources

- [bpe code file (compatible with subword-nmt)](https://github.com/butsugiri/gec-pseudodata/blob/master/bpe/bpe_code.trg.dict_bpe8000)
    - Number of merge operation is set to `8000`.
- model files
    - [pretlarge (pre-train only)](https://gec-pseudo-data.s3-ap-northeast-1.amazonaws.com/ldc_giga.pret.checkpoint_last.pt)
    - [pretlarge+SSE (pre-train only)]( https://gec-pseudo-data.s3-ap-northeast-1.amazonaws.com/ldc_giga.spell_error.pretrain.checkpoint_last.pt )
    - [pretlarge (finetuned)]( https://gec-pseudo-data.s3-ap-northeast-1.amazonaws.com/ldc_giga.finetune.checkpoint_best.pt )
    - [pretlarge+SSE (finetuned)]( https://gec-pseudo-data.s3-ap-northeast-1.amazonaws.com/ldc_giga.spell_error.finetune.checkpoint_best.pt )
- [vocabulary files](https://github.com/butsugiri/gec-pseudodata/tree/master/vocab)
    - These files must be passed to `fairseq-preprocess` if you are to fine-tune our pre-trained model with your own data. Also, these are required for decoding from fine-tuned model.
- The outputs of models on Table 5 is available in [outputs](https://github.com/butsugiri/gec-pseudodata/tree/master/outputs).
    - ERRANT commit ID: [4d7f3c9](https://github.com/chrisjbryant/errant/commit/4d7f3c9d2ee9b3ed16106208b8b1a391fbfdd324)

## Reproducing the CoNLL2014/JFLEG/BEA-test Result

- Download test-set from appropriate places.
- Split source sentence into subwords using [this](https://github.com/butsugiri/gec-pseudodata/blob/master/bpe/bpe_code.trg.dict_bpe8000) bpe code file.
- Run following command: `output.txt` is the decoded result.

```decode.sh
#! /bin/sh
set -xe

cd /path/to/cloned/fairseq

# PATHs
CHECKPOINT="/path/to/downloaded/model.pt"  # avaiable at https://github.com/butsugiri/gec-pseudodata#resources
SRC_BPE="/path/to/src_file"  # this needs to be in subword
DATA_DIR="/path/to/vocab_dir"  # i.e., `vocab` dir in this repository

# Decoding
cat $SRC_BPE | python -u interactive.py ${DATA_DIR} \
    --path ${CHECKPOINT} \
    --source-lang src_bpe8000 \
    --target-lang trg_bpe8000 \
    --buffer-size 1024 \
    --batch-size 12 \
    --log-format simple \
    --beam 5 \
    --remove-bpe \
    | tee temp.txt

cat temp.txt | grep -e "^H" | cut -f1,3 | sed 's/^..//' | sort -n -k1  | cut -f2 > output.txt
rm temp.txt
```

The model `pretlarge+SSE (finetuned)` should achieve the score: `F0.5=62.03` .

## Generating Pseudo Data from Monolingual Corpus

### Preprocessing

- `ssplit_and_tokenize.py` applies sentence splitting and tokenization
- `remove_dirty_examples.py` removes noisy examples (details are described in the script)


### DirectNoise

- `cat monolingual_corpus.bpe | python count_unigram_freq.py > freq_file`
- `python normalize_unigram_freq.py --norm 100 < freq_file > norm_freq_file`
- `python generate_pseudo_samples.py -uf norm_freq_file -po 0.2 -pm 0.7 --single_mistake 0 --seed 2020 > proc_file`
- feed `proc_file` to `fairseq_preprocess`


## Citing

If you use resources in this repository, please cite our paper.
```reference.bib
@inproceedings{kiyono-etal-2019-empirical,
    title = "An Empirical Study of Incorporating Pseudo Data into Grammatical Error Correction",
    author = "Kiyono, Shun  and
      Suzuki, Jun  and
      Mita, Masato  and
      Mizumoto, Tomoya  and
      Inui, Kentaro",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)",
    month = nov,
    year = "2019",
    address = "Hong Kong, China",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/D19-1119",
    pages = "1236--1242",
    abstract = "The incorporation of pseudo data in the training of grammatical error correction models has been one of the main factors in improving the performance of such models. However, consensus is lacking on experimental configurations, namely, choosing how the pseudo data should be generated or used. In this study, these choices are investigated through extensive experiments, and state-of-the-art performance is achieved on the CoNLL-2014 test set (F0.5=65.0) and the official test set of the BEA-2019 shared task (F0.5=70.2) without making any modifications to the model architecture.",
}
```
