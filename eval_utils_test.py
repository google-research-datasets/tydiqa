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
"""Testing code for eval_utils."""

import tensorflow.compat.v1 as tf
import tydiqa.eval_utils as util


class EvalUtilsTest(tf.test.TestCase):
  """Testing codes for eval_utils."""

  def testSpan(self):
    """Test inconsistent null spans."""
    self.assertRaises(ValueError, util.Span, -1, 1)
    self.assertRaises(ValueError, util.Span, 1, -1)

  def testNullSpan(self):
    """Test null spans."""
    self.assertTrue(util.Span(-1, -1).is_null_span())
    self.assertFalse(util.Span(0, 1).is_null_span())

  def testSpanEqual(self):
    """Test span equals."""
    span_a = util.Span(100, 102)
    span_b = util.Span(100, 102)
    self.assertTrue(util.nonnull_span_equal(span_a, span_b))

    span_a = util.Span(100, 102)
    span_b = util.Span(22, 23)
    self.assertFalse(util.nonnull_span_equal(span_a, span_b))

  def testSpanPartialMatch(self):
    """Test span equals."""
    # exact match.
    gold_span = util.Span(100, 102)
    pred_span = util.Span(100, 102)
    pre, rec, f1 = util.compute_partial_match_scores(gold_span, pred_span)
    self.assertEqual((1., 1., 1.), (pre, rec, f1))

    # pred earlier than gold, no overlap
    gold_span = util.Span(100, 102)
    pred_span = util.Span(78, 100)
    pre, rec, f1 = util.compute_partial_match_scores(gold_span, pred_span)
    self.assertEqual((0.0, 0.0, 0.0), (pre, rec, f1))

    # gold earlier than pred, no overlap
    gold_span = util.Span(1, 42)
    pred_span = util.Span(78, 100)
    pre, rec, f1 = util.compute_partial_match_scores(gold_span, pred_span)
    self.assertEqual((0.0, 0.0, 0.0), (pre, rec, f1))

    # partial overlap, gold inside pred.
    gold_span = util.Span(100, 102)
    pred_span = util.Span(100, 104)
    pre, rec, f1 = util.compute_partial_match_scores(gold_span, pred_span)
    self.assertEqual((0.5, 1.), (pre, rec))

    # partial overlap, gold comes before pred.
    gold_span = util.Span(90, 104)
    pred_span = util.Span(100, 112)
    pre, rec, f1 = util.compute_partial_match_scores(gold_span, pred_span)
    self.assertEqual((4./12, 4./14), (pre, rec))

    # partial overlap, gold fully inside pred.
    gold_span = util.Span(101, 102)
    pred_span = util.Span(100, 104)
    pre, rec, f1 = util.compute_partial_match_scores(gold_span, pred_span)
    self.assertEqual((0.25, 1), (pre, rec))

    # partial overlap, pred fully inside gold.
    gold_span = util.Span(100, 104)
    pred_span = util.Span(101, 102)
    pre, rec, f1 = util.compute_partial_match_scores(gold_span, pred_span)
    self.assertEqual((1, 0.25), (pre, rec))


if __name__ == '__main__':
  tf.test.main()
