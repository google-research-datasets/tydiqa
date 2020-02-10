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
# coding=utf-8
"""Tests for `run_tydi.py`."""

import gzip
import json
import os

import tensorflow.compat.v1 as tf
import data
import prepare_tydi_data
import preproc
import tf_io
import tokenization

# For test_srcdir
flags = tf.flags
FLAGS = flags.FLAGS


class RunTyDiTest(tf.test.TestCase):

  EXAMPLE = {
      "annotations": [{
          "annotation_id": 11725509819744756779,
          "minimal_answer": {
              "plaintext_end_byte": 2,
              "plaintext_start_byte": 7
          },
          "passage_answer": {
              "candidate_index": 0
          },
          "yes_no_answer": "NONE"
      }],
      "document_url":
          "https://en.wikipedia.org/wiki/Feature%20length",
      "document_title":
          "Feature length",
      "example_id":
          1472936222771770808,
      "language":
          "english",
      "document_plaintext":
          "In motion [CLS] [Q] picture terminology,",
      "passage_answer_candidates": [{
          "html_end_byte": 2050,
          "html_start_byte": 232,
          "plaintext_end_byte": 30,
          "plaintext_start_byte": 0
      }],
      "question_text":
          "What is the average length of a feature-length motion picture?"
  }

  def setUp(self):
    super(RunTyDiTest, self).setUp()
    self.test_tmpdir = tf.test.get_temp_dir()
    self.test_file = os.path.join(self.test_tmpdir, "tydi-unittest.jsonl.gz")

  def write_examples(self, examples):
    tf.gfile.MakeDirs(self.test_tmpdir)
    path = os.path.join(self.test_tmpdir, "tydi-unittest.jsonl.gz")
    with gzip.GzipFile(fileobj=tf.gfile.Open(path, "w")) as output_file:  # pytype: disable=wrong-arg-types
      for e in examples:
        output_file.write((json.dumps(e) + "\n").encode())

  def make_tf_examples(self, example, is_training):
    passages = []
    spans = []
    token_maps = []
    vocab_file = self._get_vocab_file()
    tf_example_creator = tf_io.CreateTFExampleFn(
        is_training=is_training,
        max_question_length=64,
        max_seq_length=512,
        doc_stride=128,
        include_unknowns=1.0,
        vocab_file=vocab_file)
    for record in list(
        tf_example_creator.process(example, errors=[], debug_info={})):
      tfexample = tf.train.Example()
      tfexample.ParseFromString(record)
      tokens = []
      passages.append(" ".join(tokens).replace(" ##", ""))
      if is_training:
        start = tfexample.features.feature["start_positions"].int64_list.value[
            0]
        end = tfexample.features.feature["end_positions"].int64_list.value[0]
        spans.append(" ".join(tokens[start:end + 1]).replace(" ##", ""))
      else:
        token_maps.append(
            tfexample.features.feature["token_map"].int64_list.value)

    return passages, spans, token_maps

  def test_minimal_examples(self):
    num_examples = 10
    self.write_examples([self.EXAMPLE] * num_examples)
    path = os.path.join(self.test_tmpdir, "tydi-unittest.jsonl.gz")
    output_examples = prepare_tydi_data.read_entries(
        path, fail_on_invalid=False)
    self.assertEqual(num_examples, len(list(output_examples)))

  def test_example_metadata(self):
    self.write_examples([self.EXAMPLE])
    path = os.path.join(self.test_tmpdir, "tydi-unittest.jsonl.gz")
    _, _, output_example, _ = next(
        prepare_tydi_data.read_entries(path, fail_on_invalid=False))
    self.assertEqual(output_example["name"], "Feature length")
    self.assertEqual(output_example["id"], "1472936222771770808")
    self.assertEqual(
        output_example["question"]["input_text"],
        "What is the average length of a feature-length motion picture?")

  def _get_vocab_file(self):
    return os.path.join(FLAGS.test_srcdir,
                        ".//baseline",
                        "mbert_modified_vocab.txt")

  def test_offset_wp_mapping(self):
    """Test the mapping from wordpiece to plaintext offsets."""
    testdata = os.path.join(
        FLAGS.test_srcdir, ".//"
        "small_gold_annotation.jsonl")
    vocab_file = self._get_vocab_file()
    examples = preproc.read_tydi_examples(
        testdata,
        is_training=False,
        max_passages=45,
        max_position=45,
        fail_on_invalid=False,
        open_fn=tf_io.gopen)
    vocab_file = self._get_vocab_file()
    tokenizer = tokenization.TyDiTokenizer(vocab_file=vocab_file)
    for tydi_example in examples:
      wordpieces, start_offsets, end_offsets, offset_to_wp = (
          tokenizer.tokenize_with_offsets(tydi_example.contexts))

      # Check invariants.
      for i in start_offsets:
        if i > 0:
          self.assertLess(
              i, len(tydi_example.context_to_plaintext_offset),
              "Expected start offset {} to be in `context_to_plaintext_offset` "
              "byte_len(contexts)={} Context@{}='{}' Have={}".format(
                  i, data.byte_len(tydi_example.contexts), i,
                  data.byte_slice(
                      tydi_example.contexts, i, i + 100,
                      errors="ignore").encode("utf8"),
                  tydi_example.context_to_plaintext_offset))
      for i in end_offsets:
        if i > 0:
          self.assertLess(
              i, len(tydi_example.context_to_plaintext_offset),
              "Expected end offset {} to be in `context_to_plaintext_offset` "
              "byte_len(contexts)={} Have={}".format(
                  i, data.byte_len(tydi_example.contexts),
                  tydi_example.context_to_plaintext_offset))

      wp_start_offsets, wp_end_offsets = (
          preproc.create_mapping(start_offsets, end_offsets,
                                 tydi_example.context_to_plaintext_offset))
      wp_count = 0
      for wp_s, wp_e in zip(wp_start_offsets, wp_end_offsets):
        if wp_s >= 0 or wp_e >= 0 and wp_count < 20:
          wp_txt = wordpieces[wp_count]
          if isinstance(wp_txt, str):
            if "##" not in wp_txt and wp_txt != "[UNK]":
              self.assertEqual(tydi_example.plaintext[wp_s:wp_e + 1], wp_txt)
        wp_count += 1

      for offset in offset_to_wp:
        self.assertLess(offset, data.byte_len(tydi_example.contexts))
        self.assertGreaterEqual(offset, 0)
        matching_wp = offset_to_wp[offset]
        if matching_wp == -1:
          continue
        if wp_end_offsets[matching_wp] == -1:
          continue
        if wp_start_offsets[matching_wp] == -1:
          continue
        self.assertGreaterEqual(wp_end_offsets[matching_wp],
                                wp_start_offsets[matching_wp])

  def test_tokenizer_simple(self):
    vocab_file = self._get_vocab_file()
    tokenizer = tokenization.TyDiTokenizer(vocab_file=vocab_file)
    text = "[CLS] [ContextId=0] This is a test."
    tokens, _, _, _ = tokenizer.tokenize_with_offsets(text)

    # Create reverse vocab lookup.
    reverse_vocab_table = {
        word_id: word for word, word_id in tokenizer.vocab.items()
    }
    output_tokens = [reverse_vocab_table[i] for i in tokens]
    self.assertEqual(output_tokens,
                     ["[CLS]", "[ContextId=0]", "This", "is", "a", "test", "."])

  def test_tokenizer_korean(self):
    vocab_file = self._get_vocab_file()
    tokenizer = tokenization.TyDiTokenizer(
        vocab_file=vocab_file, fail_on_mismatch=True)
    text = "[Q] 작가는 만화를 그리기 시작했나요?"
    tokens, _, _, _ = tokenizer.tokenize_with_offsets(text)

    # Create reverse vocab lookup.
    reverse_vocab_table = {
        word_id: word for word, word_id in tokenizer.vocab.items()
    }
    output_tokens = [reverse_vocab_table[i] for i in tokens]
    self.assertEqual(output_tokens, [
        "[Q]", u"\uc791", u"##\uac00\ub294", u"\ub9cc", u"##\ud654\ub97c",
        u"\uadf8", u"##\ub9ac", u"##\uae30", u"\uc2dc", u"##\uc791",
        u"##\ud588", u"##\ub098", u"##\uc694", "?"
    ])

  def test_tokenizer(self):
    testdata = os.path.join(
        FLAGS.test_srcdir, ".//"
        "small_gold_annotation.jsonl")
    test_examples = preproc.read_tydi_examples(
        testdata,
        is_training=True,
        max_passages=45,
        max_position=45,
        fail_on_invalid=False,
        open_fn=tf_io.gopen)
    vocab_file = self._get_vocab_file()
    tokenizer = tokenization.TyDiTokenizer(vocab_file=vocab_file)
    for tydi_example in test_examples:
      features = preproc.convert_single_example(
          tydi_example,
          tokenizer,
          is_training=True,
          max_question_length=64,
          max_seq_length=512,
          doc_stride=128,
          include_unknowns=1.0,
          errors=[],
          debug_info={})
      self.assertEqual(len(set([f.language_id for f in features])), 1)
      for feature in features:
        if feature.end_position <= 0:
          self.assertEqual(feature.start_position, 0)

  def test_tokenizer_val(self):
    testdata = os.path.join(
        FLAGS.test_srcdir, ".//"
        "small_gold_annotation.jsonl")
    train_examples = preproc.read_tydi_examples(
        testdata,
        is_training=True,
        max_passages=45,
        max_position=45,
        fail_on_invalid=False,
        open_fn=tf_io.gopen)
    dev_examples = preproc.read_tydi_examples(
        testdata,
        is_training=False,
        max_passages=45,
        max_position=45,
        fail_on_invalid=False,
        open_fn=tf_io.gopen)
    vocab_file = self._get_vocab_file()
    tokenizer = tokenization.TyDiTokenizer(vocab_file=vocab_file)
    for tr_ex, dev_ex in zip(train_examples, dev_examples):
      train_feats = preproc.convert_single_example(
          tr_ex,
          tokenizer,
          is_training=True,
          max_question_length=64,
          max_seq_length=512,
          doc_stride=128,
          include_unknowns=1.0,
          errors=[],
          debug_info={})
      dev_feats = preproc.convert_single_example(
          dev_ex,
          tokenizer,
          is_training=False,
          max_question_length=64,
          max_seq_length=512,
          doc_stride=128,
          include_unknowns=1.0,
          errors=[],
          debug_info={})
      for train_f, dev_f in zip(train_feats, dev_feats):
        if train_f.answer_text:
          st_ = train_f.start_position
          ed_ = train_f.end_position
          st_offset = dev_f.wp_start_offset[st_]
          end_offset = dev_f.wp_end_offset[ed_]
          self.assertGreaterEqual(end_offset, st_offset)

  def test_byte_str(self):
    self.assertEqual(data.byte_str("작"), b"\xec\x9e\x91")
    self.assertEqual(data.byte_str("[Q]"), b"[Q]")

  def test_byte_len(self):
    self.assertEqual(data.byte_len("작"), 3)
    self.assertEqual(data.byte_len("[Q]"), 3)

  def test_byte_slice(self):
    # 작 -- 3 UTF-8 bytes
    s = "[Q] 작가는 만화를 그리기 시작했나요?"
    q = data.byte_slice(s, 0, 3)
    self.assertEqual(q, "[Q]")

    one_char = data.byte_slice(s, 4, 7)
    self.assertEqual(one_char, "작")


if __name__ == "__main__":
  tf.test.main()
