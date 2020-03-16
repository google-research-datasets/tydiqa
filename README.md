# TyDi QA: A Benchmark for Information-Seeking Question Answering in Typologically Diverse Languages

[**Tasks**](#the-tasks) | [**Download**](#download-the-dataset) |
[**Baselines**](#building-a-baseline-system) | [**Evaluation**](#evaluation) |
[**Leaderboard**](#leaderboard-submissions) |
[**Website/Glosses**](https://ai.google.com/research/tydiqa) |
[**Paper**](https://storage.googleapis.com/tydiqa/tydiqa.pdf) |
[**Slides**](https://storage.googleapis.com/tydiqa/tydiqa_slides_stanford.pdf) |
[**Changelog**](CHANGELOG.md)

This repository contains information about TyDi QA, code for evaluating results
on the dataset, implementations of baseline systems for the dataset, and some
advice for working with the dataset.

Want to keep up to date on updates and new releases? Join our low-traffic
[announcement email list](https://groups.google.com/forum/#!forum/tydiqa-announce).

# Introduction

TyDi QA is a question answering dataset covering 11 typologically diverse
languages with 204K question-answer pairs. The languages of TyDi QA are diverse
with regard to their typology -- the set of linguistic features that each
language expresses -- such that we expect models performing well on this set to
generalize across a large number of the languages in the world. It contains
language phenomena that would not be found in English-only corpora. To provide a
realistic information-seeking task and avoid priming effects, questions are
written by people who want to know the answer, but don’t know the answer yet,
(unlike SQuAD and its descendents) and the data is collected directly in each
language without the use of translation (unlike MLQA and XQuAD).

To see some examples from the dataset with linguistic glosses or for information
on TyDi QA's leaderboard, see the
[website](https://ai.google.com/research/tydiqa).

For a full description of the dataset, how it was collected, and the quality
measurements for the baseline system, see the
[TACL article](https://storage.googleapis.com/tydiqa/tydiqa.pdf).

# The Tasks

*   Primary tasks:
    *   **Passage selection task (SelectP):** Given a list of the passages in
        the article, return either (a) the index of the passage that answers the
        question or (b) NULL if no such passage exists.
    *   **Minimal answer span task (MinSpan):** Given the full text of an
        article, return one of (a) the start and end byte indices of the minimal
        span that completely answers the question; (b) YES or NO if the question
        requires a yes/no answer and we can draw a conclusion from the passage;
        (c) NULL if it is not possible to produce a minimal answer for this
        question.
*   Secondary task:
    *   **Gold passage task (GoldP):** Given a passage that is guaranteed to
        contain the answer, predict the single contiguous span of characters
        that answers the question. This is more similar to existing reading
        comprehension datasets (as opposed to the information-seeking task
        outlined above). This task is constructed with two goals in mind: (1)
        more directly comparing with prior work and (2) providing a simplified
        way for researchers to use TyDi QA by providing compatibility with
        existing code for SQuAD 1.1, XQuAD, and MLQA. Toward these goals, the
        gold passage task differs from the primary task in several ways:
        *   only the gold answer passage is provided rather than the entire
            Wikipedia article;
        *   unanswerable questions have been discarded, similar to MLQA and
            XQuAD;
        *   we evaluate with the SQuAD 1.1 metrics like XQuAD; and
        *   Thai and Japanese are removed since the lack of whitespace breaks
            some tools.

We of course encourage you to participate in the primary tasks as we believe
these are a fuller and more robust representative of information-seeking
question answering. However, we realize that not all researchers may be able to
jump directly into these tasks. If you are constrained by computational
resources or are tied to existing code that processes the SQuAD format, the gold
passage task may be a better way for you to get started.

**When reporting results for any TyDi QA tasks, please include the full task
descriptor using one of the strings: TyDiQA-SelectP, TyDiQA-MinSpan, or
TyDiQA-GoldP**. Please do NOT simply list 'TyDi QA' in your results table, since
we do have several flavors of the task, which are quite different from one
another and we want to avoid confusion.

# Download the Dataset

Once you've chosen which task to work on (above), you can download the data at
the following URLs.

For the primary tasks:

*   [https://storage.googleapis.com/tydiqa/v1.0/tydiqa-v1.0-dev.jsonl.gz](https://storage.googleapis.com/tydiqa/v1.0/tydiqa-v1.0-dev.jsonl.gz)
*   [https://storage.googleapis.com/tydiqa/v1.0/tydiqa-v1.0-train.jsonl.gz](https://storage.googleapis.com/tydiqa/v1.0/tydiqa-v1.0-train.jsonl.gz)

The primary task training set is about 1.6GB while the dev set is about 150MB.

For the gold passage task:

*   [https://storage.googleapis.com/tydiqa/v1.1/tydiqa-goldp-v1.1-dev.json](https://storage.googleapis.com/tydiqa/v1.1/tydiqa-goldp-v1.1-dev.json)
*   [https://storage.googleapis.com/tydiqa/v1.1/tydiqa-goldp-v1.1-train.json](https://storage.googleapis.com/tydiqa/v1.1/tydiqa-goldp-v1.1-train.json)
*   [https://storage.googleapis.com/tydiqa/v1.1/tydiqa-goldp-v1.1-dev.tgz](https://storage.googleapis.com/tydiqa/v1.1/tydiqa-goldp-v1.1-dev.tgz)

The gold passage training set is about 50MB and the dev set is about 10MB. The
extra tarball for the dev set contains JSON files that are split along language
boundaries; these are used for evaluation while the single large JSON dev file
makes it easier to run inference on the entire dev set in a single invocation.

# Building a Baseline System

## Primary Tasks (TyDiQA-SelectP and TyDiQA-MinSpan)

We provide a baseline system based on multilingual BERT in this repo. Please see
[baseline/README.md](baseline) for details on running and modifying that system.
You may also find this code useful even if you plan to build a system from
scratch as it is designed to be easily re-used.

## Gold Passage Task (TyDiQA-GoldP)

Because the gold passage task has been simplified to fit the constraints of the
SQuAD 1.1 setting, it can generally be swapped into any code that accepts SQuAD
1.1 JSON inputs by simply changing a few file paths in your code. We provide an
example of doing exactly this with the original/unmodified multilingual BERT
reference implementation. See
[gold_passage_baseline/README.md](gold_passage_baseline) for details.

# Evaluation

## Primary Task Evaluation

The predictions can be evaluated using a command like the following:

```
python3 tydi_eval.py \
  --gold_path=tydiqa-v1.0-dev.jsonl.gz \
  --predictions_path=your_model_predictions.jsonl
```

This script computes language-wise F1 scores and then averages over languages,
excluding English. Spans are compared based on predicted byte positions and
partial credit is assigned within spans based on F1 positional overlap. See the
description of evaluation in the TACL article for details.

Please see the [evaluation script](tydi_eval.py) for a description of the
prediction format that your model should output.

## Gold Passage Task Evaluation

For the gold passage task, we re-use the existing SQuAD 1.1 evaluation code to
allow maximal re-use of existing pipelines. An example of calling the code for
evaluation is in
[gold_passage_baseline/eval_gold_passage_baseline.sh](gold_passage_baseline/eval_gold_passage_baseline.sh).

```
cd gold_passage_baseline
vim eval_gold_passage_baseline.sh  # Edit path to `TYDIQA_GOLDP_DIR`
./eval_gold_passage_baseline.sh predictions.jsonl /tmp
```

Note that for dev and test evaluation, each language is evaluated separately and
the overall score is the average over languages, excluding English.

# Leaderboard Submissions

In addition to reporting results on the dev set in your own research articles,
we also encourage you to submit to our
[public leaderboard](https://ai.google.com/research/tydiqa), to create a record
of your experiments. We believe leaderboard submissions serve two main purposes:

(a) to create an existence proof that such a result is **possible** under
carefully isolated conditions (i.e. cheating, intentional or accidental is
difficult) so that the community knows such a score is possible; and (b) to
inform the community **how** the result was obtained. Toward this latter goal,
we request that you submit a description (e.g. paper draft) of your submission
and also answer a few "repoducibility questions" that let the community know if
it will be possible to reproduce and build on your result. These include:

1.  Is there a research paper describing the system you are submitting? (The
    community benefits far more from knowing how to achieve a result than the
    fact that it exists.)
2.  Is the source code for the system you are submitting publicly available?
    (Your results will be replicated and trusted more if the community can
    quickly and reliably reproduce your results).
3.  Was the system you are submitting trained on any additional public data?
4.  Was the system you are submitting trained on any NON-public data? (The
    community cannot reproduce results on non-public data.)
5.  Was the system you are submitting trained with, or does it use, any external
    APIs, data labelers, or data transformations (e.g. a translation API)? (The
    use of public APIs is not reproducible and creates a black box effect since
    the community does not know the details of the underlying model and data it
    was built on.)

For step-by-step instructions on submitting, see
[leaderboard.md](leaderboard.md).

In addition to submitting to the leaderboard we encourage you to make both your
source code and your Docker images public so that others can easily run
inference with your system. This opens up the possibility of others (such as
MT-focused researchers) building on top of (and citing!) your QA system.

# Analyze Your Results

We encourage those working with the data to not only report numeric results, but
also analyze the results at a linguistic level. Consider partnering with
linguists and/or native speakers of these languages to create glosses that
explain how your model is interacting with language. See the TACL article for
examples of glossed examples with explanations (Figures 2 - 7).

# Source Data

The articles for TyDi QA are drawn from single coherent snapshots of Wikipedia
from the Internet Archive to enable open-retrieval experiments. You can download
the original article data in Wikitext format from the following URLS:

*   https://archive.org/download/arwiki-20190201/arwiki-20190201-pages-articles-multistream.xml.bz2
*   https://archive.org/download/bnwiki-20190201/bnwiki-20190201-pages-articles-multistream.xml.bz2
*   https://archive.org/download/enwiki-20190201/enwiki-20190201-pages-articles-multistream.xml.bz2
*   https://archive.org/download/fiwiki-20190201/fiwiki-20190201-pages-articles-multistream.xml.bz2
*   https://archive.org/download/idwiki-20190201/idwiki-20190201-pages-articles-multistream.xml.bz2
*   https://archive.org/download/jawiki-20190201/jawiki-20190201-pages-articles-multistream.xml.bz2
*   https://archive.org/download/kowiki-20190201/kowiki-20190201-pages-articles-multistream.xml.bz2
*   https://archive.org/download/ruwiki-20190201/ruwiki-20190201-pages-articles-multistream.xml.bz2
*   https://archive.org/download/tewiki-20190201/tewiki-20190201-pages-articles-multistream.xml.bz2
*   https://archive.org/download/tlwiki-20190201/tlwiki-20190201-pages-articles-multistream.xml.bz2
*   https://archive.org/download/swwiki-20190201/swwiki-20190201-pages-articles-multistream.xml.bz2
*   https://archive.org/download/thwiki-20190101/thwiki-20190101-pages-articles-multistream.xml.bz2

# Citation

Please cite the
[TyDi QA TACL article](https://storage.googleapis.com/tydiqa/tydiqa.pdf) as:

```
@article{tydiqa,
title   = {TyDi QA: A Benchmark for Information-Seeking Question Answering in Typologically Diverse Languages},
author  = {Jonathan H. Clark and Eunsol Choi and Michael Collins and Dan Garrette and Tom Kwiatkowski and Vitaly Nikolaev and Jennimaria Palomaki}
year    = {2020},
journal = {Transactions of the Association for Computational Linguistics}
}
```

# Contact us

If you have a technical question regarding the dataset, code or publication,
please create an issue in this repository. This is the fastest way to reach us.
