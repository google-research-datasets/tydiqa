#!/usr/bin/env ducttape
#
# This is a ducttape (hyper)workflow script. Ducttape is not required for
# running the TyDi QA baselines, but you may find it more convenient than
# running individual commands in the shell or writing fragile shell scripts.
# It will download all required data and software for you (with the exception
# of packages that must be installed system-wide), prepare the data, train a
# system, run predictions, and evaluate the system after you issue
# One Unix Command.
#
# To use this script:
# 1. Install dependencies (ducttape, Python3, TensorFlow)
# 2. Run `./tydiqa.tape`.
# 3. Wait awhile.
#
# ** Dependencies **
# ducttape: https://github.com/ExperimentWith/ducttape/files/1725708/ducttape.v0.4.binary.zip
# documentation and tutorial for ducttape: http://github.com/jhclark/ducttape
#
# And Python3 + Tensorflow:
# sudo apt install python3-dev python3-pip
# pip3 install --upgrade tensorflow-gpu

global {
  ducttape_experimental_packages=true
}

package tydiqa
  :: .versioner=git
  :: .repo="git://github.com/google-research-datasets/tydiqa"
  :: .ref=HEAD
{}

task download
  > train="tydiqa-v1.0-train.jsonl.gz"
  > dev="tydiqa-v1.0-dev.jsonl.gz"
  > mbert="multi_cased_L-12_H-768_A-12"
{
  wget "https://storage.googleapis.com/tydiqa/v1.0/tydiqa-v1.0-dev.jsonl.gz" .
  wget "https://storage.googleapis.com/tydiqa/v1.0/tydiqa-v1.0-train.jsonl.gz" .
  wget "https://storage.googleapis.com/bert_models/2018_11_23/multi_cased_L-12_H-768_A-12.zip"
  unzip "multi_cased_L-12_H-768_A-12.zip"
}

task prepare_train : tydiqa
                 < jsonl=$train@download
                 > tfrecord="train.tfrecord"
                 > count="count.txt"
{
  python3 "${tydiqa}/baseline/prepare_tydi_data.py" \
    --input_jsonl="${jsonl}" \
    --output_tfrecord="${tfrecord}" \
    --vocab_file="${tydiqa}/baseline/mbert_modified_vocab.txt" \
    --record_count_file="${count}" \
    --is_training=true
}

task prepare_dev : tydiqa
                 < jsonl=$dev@download
                 > tfrecord="dev.tfrecord"
{
  python3 "${tydiqa}/baseline/prepare_tydi_data.py" \
    --input_jsonl="${jsonl}" \
    --output_tfrecord="${tfrecord}" \
    --vocab_file="${tydiqa}/baseline/mbert_modified_vocab.txt" \
    --is_training=false
}

task train : tydiqa
  < mbert=@download
  < tfrecord=@prepare_train
  < count=@prepare_train
  > checkpoint=.
{
  python3 "${tydiqa}/baseline/run_tydi.py" \
    --vocab_file="${tydiqa}/baseline/mbert_modified_vocab.txt" \
    --bert_config_file="${mbert}/bert_config.json" \
    --init_checkpoint="${mbert}/bert_model.ckpt" \
    --train_records_file="${tfrecord}" \
    --record_count_file="${count}" \
    --do_train \
    --output_dir=.
}

task predict : tydiqa
  < mbert=@download
  < dev_jsonl=$dev@download
  < tfrecord=@prepare_dev
  < checkpoint=@train
  > predictions="pred.jsonl"
{
  python3 "${tydiqa}/baseline/run_tydi.py" \
    --bert_config_file="${mbert}/bert_config.json" \
    --vocab_file="${tydiqa}/baseline/mbert_modified_vocab.txt" \
    --init_checkpoint="${checkpoint}" \
    --predict_file="${dev_jsonl}" \
    --precomputed_predict_file="${tfrecord}"\
    --do_predict \
    --output_dir=. \
    --output_prediction_file="${predictions}"
}

task eval : tydiqa
  < predictions=@predict
  < gold=$dev@download
{
  python3 "${tydiqa}/tydi_eval.py" \
    --gold_path="${gold}" \
    --predictions_path="${predictions}"
}
