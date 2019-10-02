# pseudodata-for-gec

This is the official repository of following paper: 

```
An Empirical Study of Incorporating Pseudo Data into Grammatical Error Correction
Shun Kiyono, Jun Suzuki, Masato Mita, Tomoya Mizumoto, Kentaro Inui
2019 Conference on Empirical Methods in Natural Language Processing and 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP2019), 2019 
```

## Requirements

- PyTorch (version 1.0.1.post2 is recommended)
- [blingfire](https://github.com/microsoft/BlingFire) (for preprocessing - sentence splitting)
- [spaCy](https://spacy.io/) (for preprocessing - tokenization)
- [subword-nmt](https://github.com/rsennrich/subword-nmt) (for splitting the data into subwords)
- [fairseq](https://github.com/pytorch/fairseq) (I used commit ID: [3658fa3](https://github.com/pytorch/fairseq/commit/3658fa329b8cb987d951b2e38ec86c44b9e1fea5) for all experiments. I recommend sticking with the same commit ID.)

## Resources

- [bpe code file (compatible with subword-nmt)](https://github.com/butsugiri/gec-pseudodata/blob/master/bpe/bpe_code.trg.dict_bpe8000)
    - Number of merge operation is set to `8000`.
- model files
    - [pretlarge (pre-train only)](https://drive.google.com/open?id=1B_DRUyMiDxchFfTGBwG36BrLsa7RfZnb)
    - [pretlarge+SSE (pre-train only)](https://drive.google.com/open?id=1rhcu1MfTA2JhJoRqRXvG6qKkUfcAMr8d)
    - [pretlarge (finetuned)](https://drive.google.com/open?id=1-QHd0_RxJPnkwta9kgoT41DLWg4ogLt8)
    - [pretlarge+SSE (finetuned)](https://drive.google.com/open?id=1hNNfY3Rg6SfjLPRAezdGblprOo0NDUQY)
- [vocabulary files](https://github.com/butsugiri/gec-pseudodata/tree/master/vocab)
    - These files must be passed to `fairseq-preprocess` if you are to fine-tune our pre-trained model with your own data. Also, these are required for decoding from fine-tuned model.

## Reproducing the CoNLL2014/JFLEG/BEA-test Result
- Download test-set from appropriate places.
- Split source sentence into subwords using [this](https://github.com/butsugiri/gec-pseudodata/blob/master/bpe/bpe_code.trg.dict_bpe8000) bpe code file.
- Run following command

```decode.sh
#! /bin/sh
set -xe

cd /path/to/cloned/fairseq

# PATHs
CHECKPOINT="/path/to/downloaded/model.pt"
SRC_BPE="/path/to/src_file"
DATA_DIR="/path/to/vocab_dir"

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

### Backtrans (noisy)
- We have no plan to publicly release the code of Backtrans (noisy).


## Citing

If you use resources in this repository, please cite our paper.
```reference.bib
@InProceedings{emnlp-2019-gec,
    title = "An Empirical Study of Incorporating Pseudo Data into Grammatical Error Correction",
    author = "Kiyono, Shun and Suzuki, Jun  and Mita, Masato and Mizumoto, Tomoya and Inui, Kentaro",
    booktitle = "2019 Conference on Empirical Methods in Natural Language Processing and 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP2019)",
    year = "2019"
}
```
