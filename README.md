# gec-pseudodata

This is the official repository of following paper: 

```
An Empirical Study of Incorporating Pseudo Data into Grammatical Error Correction
```

## Requirements

- PyTorch (version 1.0.1.post2 is recommended)
- [blingfire](https://github.com/microsoft/BlingFire) (for preprocessing - ssplit)
- [spaCy](https://spacy.io/) (for preprocessing - tokenization)
- [subword-nmt](https://github.com/rsennrich/subword-nmt) (for splitting the data into subwords)
- [fairseq](https://github.com/pytorch/fairseq) (I used commit ID: `3658fa3` for all experiments. I recommend sticking with the same commit ID.)

## Resources

- [bpe code file (compatible with subword-nmt)]()
- model files
    - [pretlarge (pre-train only)]()
    - [pretlarge+SSE (pre-train only)]()
    - [pretlarge (finetuned)]()
    - [pretlarge+SSE (finetuned)]()
- vocabulary files
    - [source]()
    - [target]()

## Reproducing the CoNLL2014/JFLEG/BEA-test Result
- Download test-set from appropriate places.
- Split source sentence into subwords using [this]() bpe code file.
- Run following command
```
Here comes a command
```


## Generating Pseudo Data

### Preprocessing

### DirectNoise

### Backtrans (noisy)


