# BERT Baseline for TyDi QA Primary Tasks

This repo contains code for training and evaluating a BERT baseline model on
TyDi QA.

The approach is nearly identical to the BERT baseline for the Natural Questions
as described in [https://arxiv.org/abs/1901.08634]. Initial quality measurements
for this system on TyDi QA are given in the
[TACL article](https://storage.googleapis.com/tydiqa/tydiqa.pdf).

## Hardware Requirements

This baseline fine tunes multilingual BERT (mBERT) and so has similar compute
and memory requirements. Unlike BERT-base, mBERT requires 16 GB of GPU RAM. If
you don't have this on your local GPU, there's two avenues you might consider:
(1) grab some cycles on a cloud provider -- a Tesla T4 can be had for around
$0.35/hr or (2) rewrite the model to use something a bit less resource intensive
(you can still do good science and apples-to-apples comparisons using less
resource-intensive models).

## Install

This code runs on Python3. You'll also need the following libraries -- you can
skip the pip install steps below if you already have these on your system:

```
sudo apt install python3-dev python3-pip
pip3 install --upgrade tensorflow-gpu
```

You'll probably also want a good GPU (or TPU) to efficiently run the model
computations.

Finally, download the latest multilingual BERT checkpoint, which will serve as a
starting point for fine tuning:

[https://storage.googleapis.com/bert_models/2018_11_23/multi_cased_L-12_H-768_A-12.zip](https://storage.googleapis.com/bert_models/2018_11_23/multi_cased_L-12_H-768_A-12.zip)

## Get the Data

To get the data, see the instructions in [../README.md](../README.md) in the
main directory of this repository.

## Prepare Data

To run the TensorFlow baseline, we first have to process the data from its
original JSONL format into tfrecord format. These steps use only CPU (no GPU).

*You may wish to develop on a much smaller subset of the data. Because these are
JSONL files, standard shell commands such as `head` will work fine for this
purpose.*

First, process the smaller dev set to make sure everything is working properly:

```
python3 prepare_tydi_data.py \
  --input_jsonl=tydiqa-v1.0-dev.jsonl.gz \
  --output_tfrecord=dev_samples/dev.tfrecord \
  --vocab_file=mbert_modified_vocab.txt \
  --is_training=false
```

The output of this step will be about 3.0GB. You'll see some fairly detailed
debug logging (from `debug.py`) for the first few examples.

Next, prepare the training samples:

```
python3 prepare_tydi_data.py \
  --input_jsonl=tydiqa-v1.0-train.jsonl.gz \
  --output_tfrecord=train_samples.tfrecord \
  --vocab_file=mbert_modified_vocab.txt \
  --record_count_file=train_samples_record_count.txt \
  --is_training=true
```

The output of this step will be about 1.7GB. Note this is smaller than the dev
set since we subsample negative examples during training, but must do inference
on entire articles for the dev set. This process will take significantly longer
since we process entire Wikipedia articles. This can take around 10 hours for
the training set on a single process. A bit of extra effort of splitting this
into multiple shards, running on many cores, and then combining record counts
may save you a significant amount of wall time if you plan to run this multiple
times. Otherwise, if you plan to run this once and modify only the modeling,
then running this overnight on a workstation should be fairly painless.

## Train (Fine-tuning mBERT)

Next, we fine tune on the TyDi QA training data starting from the multilingual
BERT checkpoint, preferably on GPU:

```
python3 run_tydi.py \
  --bert_config_file=mbert_dir/bert_config.json \
  --vocab_file=mbert_modified_vocab.txt \
  --init_checkpoint=mbert_dir/bert_model.ckpt \
  --train_records_file=train_samples/*.tfrecord \
  --record_count_file=train_samples_record_count.txt \
  --do_train \
  --output_dir=~/tydiqa_baseline_model
```

## Predict

Once the model is trained, we run inference on the dev set:

```
python3 run_tydi.py \
  --bert_config_file=mbert_dir/bert_config.json \
  --vocab_file=mbert_modified_vocab.txt \
  --init_checkpoint=~/tydiqa_baseline_model \
  --predict_file=tydiqa-v1.0-dev.jsonl.gz \
  --precomputed_predict_file=dev_samples/*.tfrecord \
  --do_predict \
  --output_dir=~/tydiqa_baseline_model/predict \
  --output_prediction_file=~/tydiqa_baseline_model/predict/pred.jsonl
```

NOTE: Make sure you correctly set the `--init_checkpoint` to point to your fine
tuned weights in this step rather than the original pretrained multilingual BERT
checkpoint.

## Evaluate

For evaluation, see the instructions in [../README.md](../README.md) in the main
directory of this repository for how to evaluate the primary tasks.

We encourage you to fine tune using multiple random seeds and average the
results over these replicas to reading too much into optimization noise.

## Modify and Repeat

Once you've successfully run the baseline system, you'll likely want to improve
on it and measure the effect of your improvements.

To help you get started in modifying the baseline system to incorporate your new
idea -- or incorporating parts of the baseline system's code into your own
system -- we provide an overview of how the code is organized:

1.  [data.py] - Responsible for deserializing the JSON and creating Pythonic
    data structures. *Usable by any ML framework / no TF dependencies*

2.  [tokenization.py] - Fork of BERT's tokenizer that tracks byte offsets.
    *Usable by any ML framework / no TF dependencies*

3.  [preproc.py] - Calls tokenization and munges JSON into a format usable by
    the model. *Usable by any ML framework / no TF dependencies*

4.  [tf_io.py] - Tensorflow-specific IO code (reads `tf.Example`s from TF
    records). *If you'd like to use your own favorite DL framework, you'd need
    to modify this; it's only about 200 lines.*

5.  [tydi_modeling.py] - The core TensorFlow model code. **If you want to
    replace BERT with your own latest and greatest, start here!** *Similarly, if
    you'd like to use your own favorite DL framework, this would be the only
    file that should require heavy modification; it's only about 200 lines.*

6.  [postproc.py] - Does postprocessing to find the answer, etc. Relevant only
    for inference (not used in training). *Usable by any ML framework with
    minimal edits. Has minimal tf dependencies (e.g. a few tensor
    post-processing functions).*

7.  [run_tydi.py] - The main driver script that uses all of the above and calls
    Tensorflow to do the main training and inference loops.

If you modify any of the preprocessing code, you may wish to enable verbose
logging and use the functions in `debug.py` to print the details of how the data
is being processed to confirm your code is working as expected on the various
languages in TyDi QA.

# Citation

The citation for TyDi QA is:

```
@article{tydiqa,
title   = {TyDi QA: A Benchmark for Information-Seeking Question Answering in Typologically Diverse Languages},
author  = {Jonathan H. Clark and Eunsol Choi and Michael Collins and Dan Garrette and Tom Kwiatkowski and Vitaly Nikolaev and Jennimaria Palomaki}
year    = {2020},
journal = {Transactions of the Association for Computational Linguistics}
}
```
