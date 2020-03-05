This readme is specific to TyDi QA's secondary Gold Passage (GoldP) task. See
the README.md in the parent directory for more general information.

The TyDi QA Gold Passage task is a simplified version of TyDi QA's primary task,
which is intended to be compatible with existing code that processes the
English-only SQuAD 1.1 dataset. It is simplified in the following ways:

*   only the gold answer passage is provided rather than the entire Wikipedia
    article;
*   unanswerable questions have been discarded, similar to MLQA and XQuAD;
*   we evaluate with the SQuAD 1.1 metrics like XQuAD; and
*   Thai and Japanese are removed since the lack of whitespace require
    significant modifications to standard SQuAD evaluation scripts.

While we expect this task to be significantly easier than the full task, it does
still have the following distinguishing qualities:

*   questions were written without seeing the answer; and
*   no translation nor modeling was used to create the dataset.

These leads to measurably less lexical overlap and which we believe makes the
task more related to the needs of QA users.

To run the baseline system, it should be possible to swap in the TyDiQA-GoldP
JSON files for your SQuAD 1.1 files into your existing code. We provide a
working baseline example using the public multilingual BERT checkpoint and
open-source code.

You will need to download:

*   BERT [https://github.com/google-research/bert]
*   The latest mBERT checkpoint
    [https://storage.googleapis.com/bert_models/2018_11_23/multi_cased_L-12_H-768_A-12.zip]
*   TyDi QA [http://github.com/google-research-datasets/tydiqa]
*   a standard SQuAD 1.1 evaluation script
    [https://github.com/allenai/bi-att-flow/blob/master/squad/evaluate-v1.1.py]

Then update the path variables in the following scripts in this directory and
run them:

```
./run_gold_passage_baseline.sh
./eval_gold_passage_baseline.sh
```

The aggregate F1 score should be computed by averaging all **non-English**
languages.

**When reporting results, please include the label 'TyDiQA-GoldP' in your
results table to avoid confusion with TyDi QA's primary tasks.**

The citation for TyDiQA-GoldP is:

```
@article{tydiqa,
title   = {TyDi QA: A Benchmark for Information-Seeking Question Answering in Typologically Diverse Languages},
author  = {Jonathan H. Clark and Eunsol Choi and Michael Collins and Dan Garrette and Tom Kwiatkowski and Vitaly Nikolaev and Jennimaria Palomaki}
year    = {2020},
journal = {Transactions of the Association for Computational Linguistics}
}
```

Information specific to the Gold Passage task can be found in Section 8 of the
[TACL article](https://storage.googleapis.com/tydiqa/tydiqa.pdf); the prior
sections of the paper describe the primary tasks (not the Gold Passage task).
