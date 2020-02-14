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
"""Performs IO somewhat specific to TensorFlow.

This includes reading/writing `tf.Example`s to/from TF record files and opening
files via `tf.gfile`.
"""

import collections
import gzip

from absl import logging
import tensorflow.compat.v1 as tf

import data
import preproc
import tokenization


def input_fn_builder(input_file, seq_length, is_training, drop_remainder):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""

  # This needs to be kept in sync with `FeatureWriter`.
  name_to_features = {
      "unique_ids": tf.FixedLenFeature([], tf.int64),
      "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
      "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
      "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
  }

  if is_training:
    name_to_features["start_positions"] = tf.FixedLenFeature([], tf.int64)
    name_to_features["end_positions"] = tf.FixedLenFeature([], tf.int64)
    name_to_features["answer_types"] = tf.FixedLenFeature([], tf.int64)

  def _decode_record(record, name_to_features):
    """Decodes a record to a TensorFlow example."""
    example = tf.parse_single_example(record, name_to_features)

    # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
    # So cast all int64 to int32.
    for name in list(example.keys()):
      t = example[name]
      if t.dtype == tf.int64:
        t = tf.to_int32(t)
      example[name] = t

    return example

  def input_fn(params):
    """The actual input function."""
    batch_size = params["batch_size"]

    # For training, we want a lot of parallel reading and shuffling.
    # For eval, we want no shuffling and parallel reading doesn't matter.
    d = tf.data.TFRecordDataset(input_file)
    if is_training:
      d = d.repeat()
      d = d.shuffle(buffer_size=100)

    d = d.apply(
        tf.data.experimental.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
            batch_size=batch_size,
            drop_remainder=drop_remainder))

    return d

  return input_fn


class FeatureWriter(object):
  """Writes InputFeature to TF example file."""

  def __init__(self, filename, is_training):
    self.filename = filename
    self.is_training = is_training
    self.num_features = 0
    self._writer = tf.python_io.TFRecordWriter(filename)

  def process_feature(self, feature):
    """Write a InputFeature to the TFRecordWriter as a tf.train.Example."""
    self.num_features += 1

    def create_int_feature(values):
      feature = tf.train.Feature(
          int64_list=tf.train.Int64List(value=list(values)))
      return feature

    # This needs to be kept in sync with `input_fn_builder`.
    features = collections.OrderedDict()
    features["unique_ids"] = create_int_feature([feature.unique_id])
    features["input_ids"] = create_int_feature(feature.input_ids)
    features["input_mask"] = create_int_feature(feature.input_mask)
    features["segment_ids"] = create_int_feature(feature.segment_ids)
    features["language_id"] = create_int_feature([feature.language_id])

    if self.is_training:
      features["start_positions"] = create_int_feature([feature.start_position])
      features["end_positions"] = create_int_feature([feature.end_position])
      features["answer_types"] = create_int_feature([feature.answer_type])
    else:
      features["wp_start_offset"] = create_int_feature(feature.wp_start_offset)
      features["wp_end_offset"] = create_int_feature(feature.wp_end_offset)

    tf_example = tf.train.Example(features=tf.train.Features(feature=features))
    self._writer.write(tf_example.SerializeToString())

  def close(self):
    self._writer.close()


class CreateTFExampleFn(object):
  """Functor for creating TyDi tf.Examples to be written to a TFRecord file."""

  def __init__(self, is_training, max_question_length, max_seq_length,
               doc_stride, include_unknowns, vocab_file):
    self.is_training = is_training
    self.tokenizer = tokenization.TyDiTokenizer(vocab_file=vocab_file)
    self.max_question_length = max_question_length
    self.max_seq_length = max_seq_length
    self.doc_stride = doc_stride
    self.include_unknowns = include_unknowns
    self.vocab = self.tokenizer.vocab  # used by callers

  def process(self, entry, errors, debug_info=None):
    """Converts TyDi entries into serialized tf examples.

    Args:
      entry: "TyDi entries", dicts as returned by `create_entry_from_json`.
      errors: A list that this function appends to if errors are created. A
        non-empty list indicates problems.
      debug_info: A dict of information that may be useful during debugging.
        These elements should be used for logging and debugging only. For
        example, we log how the text was tokenized into WordPieces.

    Yields:
      `tf.train.Example` with the features needed for training or inference
      (depending on how `is_training` was set in the constructor).
    """
    if not debug_info:
      debug_info = {}
    tydi_example = data.to_tydi_example(entry, self.is_training)
    debug_info["tydi_example"] = tydi_example
    input_features = preproc.convert_single_example(
        tydi_example,
        tokenizer=self.tokenizer,
        is_training=self.is_training,
        max_question_length=self.max_question_length,
        max_seq_length=self.max_seq_length,
        doc_stride=self.doc_stride,
        include_unknowns=self.include_unknowns,
        errors=errors,
        debug_info=debug_info)
    for input_feature in input_features:
      input_feature.example_index = int(entry["id"])
      input_feature.unique_id = (
          input_feature.example_index + input_feature.doc_span_index)

      def create_int_feature(values):
        return tf.train.Feature(
            int64_list=tf.train.Int64List(value=list(values)))

      features = collections.OrderedDict()
      features["unique_ids"] = create_int_feature([input_feature.unique_id])
      features["example_index"] = create_int_feature(
          [input_feature.example_index])
      features["input_ids"] = create_int_feature(input_feature.input_ids)
      features["input_mask"] = create_int_feature(input_feature.input_mask)
      features["segment_ids"] = create_int_feature(input_feature.segment_ids)
      features["language_id"] = create_int_feature([input_feature.language_id])

      if self.is_training:
        features["start_positions"] = create_int_feature(
            [input_feature.start_position])
        features["end_positions"] = create_int_feature(
            [input_feature.end_position])
        features["answer_types"] = create_int_feature(
            [input_feature.answer_type])
      else:
        features["wp_start_offset"] = create_int_feature(
            input_feature.wp_start_offset)
        features["wp_end_offset"] = create_int_feature(
            input_feature.wp_end_offset)

      yield tf.train.Example(features=tf.train.Features(feature=features))


def gopen(path):
  """Opens a file object given a (possibly gzipped) `path`."""
  logging.info("*** Loading from: %s ***", path)
  if ".gz" in path:
    return gzip.GzipFile(fileobj=tf.gfile.Open(path, "rb"))  # pytype: disable=wrong-arg-types
  else:
    return tf.gfile.Open(path, "r")
