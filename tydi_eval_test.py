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
"""Testing code for tydi_eval."""

import tensorflow.compat.v1 as tf
import tydiqa.eval_utils as util
import tydiqa.tydi_eval as ev


class TyDiEvalTest(tf.test.TestCase):
  """Testing codes for tydi_eval."""

  def _get_null_span(self):
    return util.Span(-1, -1)

  def _get_tydi_label(self, passage_index, minimal_span, language='eng', eid=0):
    return util.TyDiLabel(
        example_id=eid,
        passage_answer_index=passage_index,
        minimal_answer_span=minimal_span,
        question_text='',
        plaintext='',
        passage_score=0,
        minimal_score=0,
        language=language,
        yes_no_answer='none')

  def _get_tydi_label_with_yes_no(self, passage_index, yes_no_answer,
                                  language='eng', eid=0):
    assert yes_no_answer != 'none'
    return util.TyDiLabel(
        example_id=eid,
        passage_answer_index=passage_index,
        minimal_answer_span=self._get_null_span(),
        question_text='',
        plaintext='',
        passage_score=0,
        minimal_score=0,
        language=language,
        yes_no_answer=yes_no_answer)

  def _get_span(self, start, end):
    return util.Span(start, end)

  def testPassageStat(self):
    """Test instance level passage answer f1."""
    # Test cases when there is no long answer.
    gold_passage_indexes = [0, 0, -1, -1]
    gold_label_list = [
        self._get_tydi_label(gold_passage_index, self._get_null_span())
        for gold_passage_index in gold_passage_indexes]
    pred_label = self._get_tydi_label(0, self._get_null_span())
    gold_has_answer, pred_has_answer, is_correct, _ = ev.score_passage_answer(
        gold_label_list, pred_label, 1)

    self.assertEqual(gold_has_answer, True)
    self.assertEqual(pred_has_answer, True)
    self.assertEqual(is_correct, True)

    # Test cases when there is a long answer.
    gold_passage_indexes = [1, 2, -1]
    gold_label_list = [
        self._get_tydi_label(gold_passage_index, self._get_null_span())
        for gold_passage_index in gold_passage_indexes]

    pred_label = self._get_tydi_label(4,
                                      self._get_null_span())
    gold_has_answer, pred_has_answer, is_correct, _ = ev.score_passage_answer(
        gold_label_list, pred_label, 1)
    self.assertEqual(gold_has_answer, True)
    self.assertEqual(pred_has_answer, True)
    self.assertEqual(is_correct, False)

  def testMinimalStat(self):
    """Test instance level minimal answer p, r, f1."""
    long_span = self._get_span(0, 10)

    # Test case assumes having 5 way annotations.
    # Test case when there is no gold short answer.
    gold_spans = [(1, 3), (-1, -1), (-1, -1), (-1, -1), (-1, -1)]
    gold_label_list = [self._get_tydi_label(
        long_span, self._get_span(a, b)) for a, b in gold_spans]
    pred_label = self._get_tydi_label_with_yes_no(long_span, 'yes')
    gold_has_answer, pred_has_answer, (p, r, f1), _ = ev.score_minimal_answer(
        gold_label_list, pred_label, 1)
    self.assertEqual(gold_has_answer, True)
    self.assertEqual(pred_has_answer, True)
    self.assertEqual((p, r, f1), (0, 0, 0))

    # This test case assumes having 5 way annotations.
    # Test case when there is gold short answer.
    gold_spans = [(39, 50), (38, 50), (34, 50), (-1, -1), (-1, -1)]
    gold_label_list = [self._get_tydi_label(
        long_span, self._get_span(a, b)) for a, b in gold_spans]
    pred_label = self._get_tydi_label(long_span, self._get_span(30, 40))
    gold_has_answer, pred_has_answer, (p, r, f1), _ = ev.score_minimal_answer(
        gold_label_list, pred_label, 1)
    self.assertEqual(gold_has_answer, True)
    self.assertEqual(pred_has_answer, True)
    self.assertEqual((p, r), (6/10., 6/16.))

    # When there is no overlap.
    pred_label = self._get_tydi_label(long_span, self._get_span(30, 34))
    gold_has_answer, pred_has_answer, (p, r, f1), _ = ev.score_minimal_answer(
        gold_label_list, pred_label, 1)
    self.assertEqual(gold_has_answer, True)
    self.assertEqual(pred_has_answer, True)
    self.assertEqual((p, r), (0., 0.))

    # When there is complete overlap.
    pred_label = self._get_tydi_label(long_span, self._get_span(39, 50))
    gold_has_answer, pred_has_answer, (p, r, f1), _ = ev.score_minimal_answer(
        gold_label_list, pred_label, 1)
    self.assertEqual(gold_has_answer, True)
    self.assertEqual(pred_has_answer, True)
    self.assertEqual((p, r), (1., 1.))

    # This test case assumes having 3 way annotations.
    # Test case when there is gold short answer.
    gold_spans = [(39, 50), (-1, -1), (-1, -1)]
    gold_label_list = [self._get_tydi_label(
        long_span, self._get_span(a, b)) for a, b in gold_spans]
    pred_label = self._get_tydi_label(long_span, self._get_span(30, 40))
    gold_has_answer, pred_has_answer, (p, r, f1), _ = ev.score_minimal_answer(
        gold_label_list, pred_label, 1)
    self.assertEqual(gold_has_answer, True)
    self.assertEqual(pred_has_answer, True)
    self.assertEqual((p, r), (1/10., 1/11.))

    # When there is no overlap.
    pred_label = self._get_tydi_label(long_span, self._get_span(30, 34))
    gold_has_answer, pred_has_answer, (p, r, f1), _ = ev.score_minimal_answer(
        gold_label_list, pred_label, 1)
    self.assertEqual(gold_has_answer, True)
    self.assertEqual(pred_has_answer, True)
    self.assertEqual((p, r), (0., 0.))

    # When there is complete overlap.
    pred_label = self._get_tydi_label(long_span, self._get_span(39, 50))
    gold_has_answer, pred_has_answer, (p, r, f1), _ = ev.score_minimal_answer(
        gold_label_list, pred_label, 1)
    self.assertEqual(gold_has_answer, True)
    self.assertEqual(pred_has_answer, True)
    self.assertEqual((p, r), (1., 1.))
