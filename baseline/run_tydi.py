# coding=utf-8
# Copyright 2020 The Google Research Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Lint as: python3
"""BERT-joint baseline for TyDi v1.0.

 This code is largely based on the Natural Questions baseline from
 https://github.com/google-research/language/blob/master/language/question_answering/bert_joint/run_nq.py.

 The model uses special tokens to dealing with offsets between the original
 document content and the wordpieces. Here are examples:
 [ContextId=N] [Q]
 The presence of these special tokens requires overwriting some of the [UNUSED]
 vocab ids of the public BERT wordpiece vocabulary, similar to NQ baseline.

Overview:
  1. data.py: Responsible for deserializing the JSON and creating Pythonic data
       structures
     [ Usable by any ML framework / no TF dependencies ]

  2. tokenization.py: Fork of BERT's tokenizer that tracks byte offsets.
     [ Usable by any ML framework / no TF dependencies ]

  3. preproc.py: Calls tokenization and munges JSON into a format usable by
       the model.
     [ Usable by any ML framework / no TF dependencies ]

  4. tf_io.py: Tensorflow-specific IO code (reads `tf.Example`s from
       TF records). If you'd like to use your own favorite DL framework, you'd
       need to modify this; it's only about 200 lines.

  4. tydi_modeling.py: The core TensorFlow model code. **If you want to replace
       BERT with your own latest and greatest, start here!** Similarly, if
       you'd like to use your own favorite DL framework, this would be
       the only file that should require heavy modification; it's only about
       200 lines.

  5. postproc.py: Does postprocessing to find the answer, etc. Relevant only
     for inference.
     [ Usable by any ML framework / minimal tf dependencies ]

  6. run_tydi.py: The actual main driver script that uses all of the above and
       calls Tensorflow to do the main training and inference loops.
"""

import collections
import json
import os

from absl import logging
from bert import modeling as bert_modeling
import tensorflow.compat.v1 as tf
import postproc
import preproc
import tf_io
import tydi_modeling

import tensorflow.contrib as tf_contrib

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written.")

flags.DEFINE_string("train_records_file", None,
                    "Precomputed tf records for training.")

flags.DEFINE_string(
    "record_count_file", None,
    "File containing number of precomputed training records "
    "(in terms of 'features', meaning slices of articles). "
    "This is used for computing how many steps to take in "
    "each fine tuning epoch.")

flags.DEFINE_integer(
    "candidate_beam", 30,
    "How many wordpiece offset to be considered as boundary at inference time.")

flags.DEFINE_string(
    "predict_file", None,
    "TyDi json for predictions. E.g., dev-v1.1.jsonl.gz or test-v1.1.jsonl.gz. "
    "Used only for `--do_predict`.")

flags.DEFINE_string(
    "precomputed_predict_file", None,
    "TyDi tf.Example records for predictions, created separately by "
    "`prepare_tydi_data.py` Used only for `--do_predict`.")

flags.DEFINE_string(
    "output_prediction_file", None,
    "Where to print predictions in TyDi prediction format, to be passed to"
    "tydi_eval.py.")

flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained mBERT model).")

flags.DEFINE_integer(
    "max_seq_length", 512,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_integer(
    "doc_stride", 128,
    "When splitting up a long document into chunks, how much stride to "
    "take between chunks.")

flags.DEFINE_integer(
    "max_question_length", 64,
    "The maximum number of tokens for the question. Questions longer than "
    "this will be truncated to this length.")

flags.DEFINE_bool("do_train", False, "Whether to run training.")

flags.DEFINE_bool("do_predict", False, "Whether to run prediction.")

flags.DEFINE_integer("train_batch_size", 16, "Total batch size for training.")

flags.DEFINE_integer("predict_batch_size", 8,
                     "Total batch size for predictions.")

flags.DEFINE_integer(
    "predict_file_shard_size", 1000,
    "The maximum number of examples to put into each temporary TF example file "
    "used as model input a prediction time.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_float("num_train_epochs", 3.0,
                   "Total number of training epochs to perform.")

flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_integer(
    "max_answer_length", 30,
    "The maximum length of an answer that can be generated. This is needed "
    "because the start and end predictions are not conditioned on one another.")

flags.DEFINE_float(
    "include_unknowns", -1.0,
    "If positive, probability of including answers of type `UNKNOWN`.")

flags.DEFINE_bool(
    "verbose_logging", False,
    "If true, all of the warnings related to data processing will be printed. "
    "A number of warnings are expected for a normal TyDi evaluation.")

flags.DEFINE_integer(
    "max_passages", 45, "Maximum number of passages to consider for a "
    "single article. If an article contains more than"
    "this, they will be discarded during training. "
    "BERT's WordPiece vocabulary must be modified to include "
    "these within the [unused*] vocab IDs.")

flags.DEFINE_integer(
    "max_position", 45,
    "Maximum passage position for which to generate special tokens.")

flags.DEFINE_bool(
    "fail_on_invalid", True,
    "Stop immediately on encountering an invalid example? "
    "If false, just print a warning and skip it.")


### TPU-specific flags:

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

tf.flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

tf.flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")


def validate_flags_or_throw(bert_config):
  """Validate the input FLAGS or throw an exception."""
  if not FLAGS.do_train and not FLAGS.do_predict:
    raise ValueError("At least one of `{do_train,do_predict}` must be True.")

  if FLAGS.do_train:
    if not FLAGS.train_records_file:
      raise ValueError("If `do_train` is True, then `train_records_file` "
                       "must be specified.")
    if not FLAGS.record_count_file:
      raise ValueError("If `do_train` is True, then `record_count_file` "
                       "must be specified.")

  if FLAGS.do_predict:
    if not FLAGS.predict_file:
      raise ValueError("If `do_predict` is True, "
                       "then `predict_file` must be specified.")
    if not FLAGS.output_prediction_file:
      raise ValueError("If `do_predict` is True, "
                       "then `output_prediction_file` must be specified.")

  if FLAGS.max_seq_length > bert_config.max_position_embeddings:
    raise ValueError(
        "Cannot use sequence length %d because the BERT model "
        "was only trained up to sequence length %d" %
        (FLAGS.max_seq_length, bert_config.max_position_embeddings))

  if FLAGS.max_seq_length <= FLAGS.max_question_length + 3:
    raise ValueError(
        "The max_seq_length (%d) must be greater than max_question_length "
        "(%d) + 3" % (FLAGS.max_seq_length, FLAGS.max_question_length))


def main(_):
  logging.set_verbosity(logging.INFO)
  bert_config = bert_modeling.BertConfig.from_json_file(FLAGS.bert_config_file)
  validate_flags_or_throw(bert_config)
  tf.gfile.MakeDirs(FLAGS.output_dir)

  tpu_cluster_resolver = None
  if FLAGS.use_tpu and FLAGS.tpu_name:
    tpu_cluster_resolver = tf_contrib.cluster_resolver.TPUClusterResolver(
        FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

  is_per_host = tf_contrib.tpu.InputPipelineConfig.PER_HOST_V2
  run_config = tf_contrib.tpu.RunConfig(
      cluster=tpu_cluster_resolver,
      master=FLAGS.master,
      model_dir=FLAGS.output_dir,
      save_checkpoints_steps=FLAGS.save_checkpoints_steps,
      tpu_config=tf_contrib.tpu.TPUConfig(
          iterations_per_loop=FLAGS.iterations_per_loop,
          per_host_input_for_training=is_per_host))

  num_train_steps = None
  num_warmup_steps = None
  if FLAGS.do_train:
    with tf.gfile.Open(FLAGS.record_count_file, "r") as f:
      num_train_features = int(f.read().strip())
    num_train_steps = int(num_train_features / FLAGS.train_batch_size *
                          FLAGS.num_train_epochs)
    logging.info("record_count_file: %s", FLAGS.record_count_file)
    logging.info("num_records (features): %d", num_train_features)
    logging.info("num_train_epochs: %d", FLAGS.num_train_epochs)
    logging.info("train_batch_size: %d", FLAGS.train_batch_size)
    logging.info("num_train_steps: %d", num_train_steps)

    num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

  model_fn = tydi_modeling.model_fn_builder(
      bert_config=bert_config,
      init_checkpoint=FLAGS.init_checkpoint,
      learning_rate=FLAGS.learning_rate,
      num_train_steps=num_train_steps,
      num_warmup_steps=num_warmup_steps,
      use_tpu=FLAGS.use_tpu,
      use_one_hot_embeddings=FLAGS.use_tpu)

  # If TPU is not available, this falls back to normal Estimator on CPU or GPU.
  estimator = tf_contrib.tpu.TPUEstimator(
      use_tpu=FLAGS.use_tpu,
      model_fn=model_fn,
      config=run_config,
      train_batch_size=FLAGS.train_batch_size,
      predict_batch_size=FLAGS.predict_batch_size)

  if FLAGS.do_train:
    logging.info("Running training on precomputed features")
    logging.info("  Num split examples = %d", num_train_features)
    logging.info("  Batch size = %d", FLAGS.train_batch_size)
    logging.info("  Num steps = %d", num_train_steps)
    train_filenames = tf.gfile.Glob(FLAGS.train_records_file)
    train_input_fn = tf_io.input_fn_builder(
        input_file=train_filenames,
        seq_length=FLAGS.max_seq_length,
        is_training=True,
        drop_remainder=True)
    estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)

  if FLAGS.do_predict:
    if not FLAGS.precomputed_predict_file:
      predict_examples_iter = preproc.read_tydi_examples(
          input_file=FLAGS.predict_file,
          is_training=False,
          max_passages=FLAGS.max_passages,
          max_position=FLAGS.max_position,
          fail_on_invalid=FLAGS.fail_on_invalid,
          open_fn=tf_io.gopen)
      shards_iter = write_tf_feature_files(predict_examples_iter)
    else:
      # Uses zeros for example and feature counts since they're unknown, and
      # we only use them for logging anyway.
      shards_iter = enumerate(
          ((f, 0, 0) for f in tf.gfile.Glob(FLAGS.precomputed_predict_file)), 1)

    # Accumulates all of the prediction results to be written to the output.
    full_tydi_pred_dict = {}
    total_num_examples = 0
    for shard_num, (shard_filename, shard_num_examples,
                    shard_num_features) in shards_iter:
      total_num_examples += shard_num_examples
      logging.info(
          "Shard %d: Running prediction for %s; %d examples, %d features.",
          shard_num, shard_filename, shard_num_examples, shard_num_features)

      # Runs the model on the shard and store the individual results.
      # If running predict on TPU, you will need to specify the number of steps.
      predict_input_fn = tf_io.input_fn_builder(
          input_file=[shard_filename],
          seq_length=FLAGS.max_seq_length,
          is_training=False,
          drop_remainder=False)
      all_results = []
      for result in estimator.predict(
          predict_input_fn, yield_single_examples=True):
        if len(all_results) % 10000 == 0:
          logging.info("Shard %d: Predicting for feature %d/%s", shard_num,
                       len(all_results), shard_num_features)
        unique_id = int(result["unique_ids"])
        start_logits = [float(x) for x in result["start_logits"].flat]
        end_logits = [float(x) for x in result["end_logits"].flat]
        answer_type_logits = [
            float(x) for x in result["answer_type_logits"].flat
        ]
        all_results.append(
            tydi_modeling.RawResult(
                unique_id=unique_id,
                start_logits=start_logits,
                end_logits=end_logits,
                answer_type_logits=answer_type_logits))

      # Reads the prediction candidates from the (entire) prediction input file.
      candidates_dict = read_candidates(FLAGS.predict_file)
      predict_features = [
          tf.train.Example.FromString(r)
          for r in tf.python_io.tf_record_iterator(shard_filename)
      ]
      logging.info("Shard %d: Post-processing predictions.", shard_num)
      logging.info("  Num candidate examples loaded (includes all shards): %d",
                   len(candidates_dict))
      logging.info("  Num candidate features loaded: %d", len(predict_features))
      logging.info("  Num prediction result features: %d", len(all_results))
      logging.info("  Num shard features: %d", shard_num_features)

      tydi_pred_dict = postproc.compute_pred_dict(
          candidates_dict,
          predict_features, [r._asdict() for r in all_results],
          candidate_beam=FLAGS.candidate_beam)

      logging.info("Shard %d: Post-processed predictions.", shard_num)
      logging.info("  Num shard examples: %d", shard_num_examples)
      logging.info("  Num post-processed results: %d", len(tydi_pred_dict))
      if shard_num_examples != len(tydi_pred_dict):
        logging.warning("  Num missing predictions: %d",
                        shard_num_examples - len(tydi_pred_dict))
      for key, value in tydi_pred_dict.items():
        if key in full_tydi_pred_dict:
          logging.warning("ERROR: '%s' already in full_tydi_pred_dict!", key)
        full_tydi_pred_dict[key] = value

    logging.info("Prediction finished for all shards.")
    logging.info("  Total input examples: %d", total_num_examples)
    logging.info("  Total output predictions: %d", len(full_tydi_pred_dict))

    with tf.gfile.Open(FLAGS.output_prediction_file, "w") as output_file:
      for prediction in full_tydi_pred_dict.values():
        output_file.write((json.dumps(prediction) + "\n").encode())


def write_tf_feature_files(tydi_examples_iter):
  """Converts TyDi examples to features and writes them to files."""
  logging.info("Converting examples started.")

  total_feature_count_frequencies = collections.defaultdict(int)
  total_num_examples = 0
  total_num_features = 0
  for shard_num, examples in enumerate(
      sharded_iterator(tydi_examples_iter, FLAGS.predict_file_shard_size), 1):
    features_writer = tf_io.FeatureWriter(
        filename=os.path.join(FLAGS.output_dir,
                              "features.tf_record-%03d" % shard_num),
        is_training=False)
    num_features_to_ids, shard_num_examples = (
        preproc.convert_examples_to_features(
            tydi_examples=examples,
            vocab_file=FLAGS.vocab_file,
            is_training=False,
            max_question_length=FLAGS.max_question_length,
            max_seq_length=FLAGS.max_seq_length,
            doc_stride=FLAGS.doc_stride,
            include_unknowns=FLAGS.include_unknowns,
            output_fn=features_writer.process_feature))
    features_writer.close()

    if shard_num_examples == 0:
      continue

    shard_num_features = 0
    for num_features, ids in num_features_to_ids.items():
      shard_num_features += (num_features * len(ids))
      total_feature_count_frequencies[num_features] += len(ids)
    total_num_examples += shard_num_examples
    total_num_features += shard_num_features
    logging.info("Shard %d: Converted %d input examples into %d features.",
                 shard_num, shard_num_examples, shard_num_features)
    logging.info("  Total so far: %d input examples, %d features.",
                 total_num_examples, total_num_features)
    yield (shard_num, (features_writer.filename, shard_num_examples,
                       shard_num_features))

  logging.info("Converting examples finished.")
  logging.info("  Total examples = %d", total_num_examples)
  logging.info("  Total features = %d", total_num_features)
  logging.info("  total_feature_count_frequencies = %s",
               sorted(total_feature_count_frequencies.items()))


def sharded_iterator(iterator, shard_size):
  """Returns an iterator of iterators of at most size `shard_size`."""
  exhaused = False
  while not exhaused:

    def shard():
      for i, item in enumerate(iterator, 1):
        yield item
        if i == shard_size:
          return
      nonlocal exhaused
      exhaused = True

    yield shard()


def read_candidates(input_pattern):
  """Read candidates from an input pattern."""
  input_paths = tf.gfile.Glob(input_pattern)
  final_dict = {}
  for input_path in input_paths:
    file_obj = tf_io.gopen(input_path)
    final_dict.update(postproc.read_candidates_from_one_split(file_obj))
  return final_dict


if __name__ == "__main__":
  flags.mark_flag_as_required("vocab_file")
  flags.mark_flag_as_required("bert_config_file")
  flags.mark_flag_as_required("output_dir")
  tf.app.run()
