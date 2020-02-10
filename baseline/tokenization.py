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
"""A fork of BERT's tokenizer that tracks byte offsets.

This module does not depend on TensorFlow and should be re-usable within your
favorite ML/DL framework.
"""

import collections

from absl import logging
from bert import tokenization as bert_tokenization
import data

SubToken = collections.namedtuple(
    "SubToken",
    [
        # The normalized version of the token, including '##' markers, etc.
        "normalized",
        # The original version of the token, which can be used for computing
        # byte offsets within a document.
        "orig",
        # Is this a 'good' token that should be fed to the model? (Stripped
        # whitespace, etc. that still needs to be kept around for byte-tracking
        # purposes.)
        "is_good"
    ])


def whitespace_tokenize(subtokens):
  """An implementation of BERT's whitespace tokenizer that preserves space."""
  return split_subtokens_on(
      subtokens, lambda char: char.isspace(), are_good=True)


def split_subtokens_on(subtokens, should_isolate_func, are_good):
  """Nondestructively splits subtokens using a character-wise predicate.

  Args:
    subtokens: List of `Subtoken`s.
    should_isolate_func: Function that takes a char and returns a boolean. True
      means that the character should be isolated in the output.
    are_good: This boolean indicates whether or not each isolated character is
      "good", controlling whether or not it will get fed to the model or simply
      dropped. This is stored in `is_good` in `SubToken`.

  Returns:
    List of `SubToken`s.
  """
  output = []
  result_subtoken = []
  for subtoken, orig_subtoken, is_good in subtokens:
    assert subtoken == orig_subtoken

    # Don't bother running predicate on bad tokens (including potentially
    # invalid Unicode).
    if not is_good:
      output.append(SubToken(subtoken, subtoken, is_good=False))
      continue

    for char in subtoken:
      if should_isolate_func(char):
        if result_subtoken:
          result_subtoken_str = "".join(result_subtoken)
          output.append(
              SubToken(result_subtoken_str, result_subtoken_str, is_good=True))
          result_subtoken = []
        output.append(SubToken(char, char, is_good=are_good))
      else:
        result_subtoken.append(char)
  if result_subtoken:
    result_subtoken_str = "".join(result_subtoken)
    output.append(
        SubToken(result_subtoken_str, result_subtoken_str, is_good=True))
  return output


class NonDestructiveFullTokenizer(object):
  """Runs end-to-end tokenziation."""

  def __init__(self, vocab_file):
    self.vocab = bert_tokenization.load_vocab(vocab_file)
    self.inv_vocab = {v: k for k, v in self.vocab.items()}
    self.basic_tokenizer = NonDestructiveBasicTokenizer(vocab=self.vocab)
    self.wordpiece_tokenizer = NonDestructiveWordpieceTokenizer(
        vocab=self.vocab)

  def tokenize(self, text):
    """Tokenizes a piece of `text` and returns a list of `SubToken`s."""
    split_tokens = []  # list of `SubToken`s.
    for token, orig_token, is_good_token in self.basic_tokenizer.tokenize(text):
      if not is_good_token:
        split_tokens.append(SubToken(token, orig_token, is_good=False))
        continue

      # Preserve special tokens such as '[Q]' and '[SEP]'.
      if bert_tokenization.preserve_token(token, self.vocab):
        split_tokens.append(SubToken(token, orig_token, is_good=True))
        continue

      # For everything else, send the text-like tokens that have survived
      # whitespace and puncutation splitting through a wordpiece tokenizer.
      for sub_token in self.wordpiece_tokenizer.tokenize(
          [SubToken(token, orig_token, is_good_token)]):
        # `sub_token` has type `SubToken`.
        split_tokens.append(sub_token)

    return split_tokens


class NonDestructiveBasicTokenizer(bert_tokenization.BasicTokenizer):
  """An implementation of BERT's BasicTokenizer that preserves space."""

  def __init__(self, vocab=tuple()):
    """Constructs a `NonDestructiveBasicTokenizer`.

    Lower casing (and accent removal) are not supported.

    Args:
      vocab: A container of tokens to not mutate during tokenization.
    """
    self.vocab = vocab

  def tokenize(self, text):
    """Tokenizes a piece of `text` and returns a list of `SubToken`s."""
    text = bert_tokenization.convert_to_unicode(text)

    # Begin with the entire input as a single string.
    subtokens = [SubToken(text, text, is_good=True)]
    del text  # unused after this point
    subtokens = self._clean_text(subtokens)

    # This was added on November 1st, 2018 for the multilingual and Chinese
    # models. This is also applied to the English models now, but it doesn't
    # matter since the English models were not trained on any Chinese data
    # and generally don't have any Chinese data in them (there are Chinese
    # characters in the vocabulary because Wikipedia does have some Chinese
    # words in the English Wikipedia.).
    subtokens = self._tokenize_chinese_chars(subtokens)

    # Split punctuation, preserving special tokens.
    subtokens = whitespace_tokenize(subtokens)

    split_subtokens = []
    for subtoken, orig_subtoken, is_good in subtokens:
      assert subtoken == orig_subtoken

      if not is_good:
        split_subtokens.append(SubToken(subtoken, subtoken, is_good=False))
        continue

      if bert_tokenization.preserve_token(subtoken, self.vocab):
        split_subtokens.append(SubToken(subtoken, subtoken, is_good=True))
        continue

      split_subtokens.extend(
          self._run_split_on_punc([SubToken(subtoken, subtoken, is_good=True)]))
    return split_subtokens

  def _run_split_on_punc(self, subtokens):
    """Splits punctuation within a list of `SubToken`s."""
    return split_subtokens_on(subtokens, self._is_punctuation, are_good=True)

  def _is_punctuation(self, char):
    return bert_tokenization._is_punctuation(char)  # pylint: disable=protected-access

  def _is_control(self, char):
    return bert_tokenization._is_control(char)  # pylint: disable=protected-access

  def _is_chinese_char(self, cp):
    return bert_tokenization.BasicTokenizer._is_chinese_char(  # pylint: disable=protected-access
        self, cp),

  def _tokenize_chinese_chars(self, subtokens):
    """Adds whitespace around any CJK character."""
    return split_subtokens_on(
        subtokens, lambda char: self._is_chinese_char(ord(char)), are_good=True)

  def _clean_text(self, subtokens):
    """Performs invalid character handling and whitespace cleanup on text.

    We never remove characters, but instead just isolate them and mark them as
    not being actual inputs to the model so that downstream code can accurately
    track byte offsets.

    Args:
      subtokens: List of input `SubToken`s.

    Returns:
      List of `SubToken`s.
    """

    def should_isolate(char):
      cp = ord(char)
      return cp == 0 or cp == 0xfffd or self._is_control(char)

    return split_subtokens_on(subtokens, should_isolate, are_good=False)


class NonDestructiveWordpieceTokenizer(object):
  """Runs WordPiece tokenziation."""

  def __init__(self, vocab, unk_token="[UNK]", max_input_chars_per_word=200):
    self.vocab = vocab
    self.unk_token = unk_token
    self.max_input_chars_per_word = max_input_chars_per_word

  def tokenize(self, subtokens):
    """Tokenizes a piece of text into its word pieces.

    This uses a greedy longest-match-first algorithm to perform tokenization
    using the given vocabulary.

    For example:
      input = "unaffable"
      output = ["un", "##aff", "##able"]

    Args:
      subtokens: List of `SubToken`s.

    Returns:
      List of `SubToken`s.
    """

    output_tokens = []
    for token, orig_token, is_good in subtokens:

      if not is_good:
        output_tokens.append(SubToken(token, orig_token, is_good=False))
        continue

      # Due to the use of Python3, this tokenization algorithm is subtly
      # different than the original BERT implementation: Instead of slicing
      # at byte boundaries and checking for membership in the vocabulary, we
      # only slice at character boundaries (and check for character-wise length,
      # not byte-wise length). In practice, this shouldn't make much difference
      # other than allowing some longer words to be checked and to prevent
      # invalid Unicode strings being checked against the vocabulary.
      token_char_len = len(token)
      if token_char_len > self.max_input_chars_per_word:
        output_tokens.append(SubToken(self.unk_token, token, is_good=True))
        continue

      is_unk = False
      start = 0
      sub_tokens = []
      while start < token_char_len:
        end = token_char_len
        cur_substr = None
        while start < end:
          orig_substr = token[start:end]
          substr = orig_substr
          if start > 0:
            substr = "##" + substr
          if substr in self.vocab:
            cur_substr = substr
            break
          end -= 1
        if cur_substr is None:
          is_unk = True
          break
        sub_tokens.append(SubToken(cur_substr, orig_substr, is_good=True))
        start = end

      if is_unk:
        output_tokens.append(SubToken(self.unk_token, token, is_good=True))
      else:
        output_tokens.extend(sub_tokens)
    return output_tokens


class TyDiTokenizer(object):
  """A BERT-compatible tokenizer that keeps byte indices."""

  def __init__(self, vocab_file, fail_on_mismatch=False):
    self.vocab = bert_tokenization.load_vocab(vocab_file)
    self.tokenizer = NonDestructiveFullTokenizer(vocab_file=vocab_file)
    self.fail_on_mismatch = fail_on_mismatch

  def tokenize(self, text):
    wordpieces, _, _, _ = self.tokenize_with_offsets(text)
    return wordpieces

  def tokenize_with_offsets(self, text):
    """Tokenize question or context with BERT tokenizer.

    Args:
      text: text string to be tokenized.

    Returns:
      tuple:
        wordpieces_out: List[int]
        start_offsets_out: List[int]
        end_offsets_out: List[int]
        offset_to_wp_out: Dict[int, int]
    """

    # First, tokenize on whitespace so that we can preserve special tokens
    # such as '[CLS]' and '[ContextId=0]' (`BertTokenizer` would split these
    # otherwise).
    whitespace_tokens = text.split(" ")

    berttok_wordpieces = []  # type List[List[int]]
    berttok_starts = []  # type List[List[int]]
    berttok_limits = []  # type List[List[int]]

    mismatched_tokens = []
    mismatch_bytes = 0
    num_tokens = len(whitespace_tokens)

    unk_token = "[UNK]"
    unk_id = self.vocab[unk_token]

    for token in whitespace_tokens:
      internal_byte_offset = 0
      subtokens = self.tokenizer.tokenize(token)
      subtoken_ids = [
          self.vocab.get(subtoken, unk_id) for subtoken, _, _ in subtokens
      ]
      subtoken_lengths = [
          data.byte_len(orig_subtoken) for _, orig_subtoken, _ in subtokens
      ]

      actual_token_length = data.byte_len(token)
      actual_subtokens_length = sum(subtoken_lengths)

      if actual_token_length != actual_subtokens_length:
        mismatched_tokens.append(token)
        mismatch_bytes += abs(actual_token_length - actual_subtokens_length)

        if self.fail_on_mismatch:
          raise ValueError(
              "Mismatched token. token='{}' (len={}) subtokens='{}' (len={})"
              .format(
                  token, actual_token_length,
                  " ".join(orig_subtoken for _, orig_subtoken, _ in subtokens),
                  actual_subtokens_length))

      inside_berttok_wordpieces = []
      inside_berttok_starts = []
      inside_berttok_limits = []
      for subtoken_id, subtoken_len in zip(subtoken_ids, subtoken_lengths):
        inside_berttok_wordpieces.append(subtoken_id)
        inside_berttok_starts.append(internal_byte_offset)
        inside_berttok_limits.append(internal_byte_offset + subtoken_len)
        # Track byte-wise offset inside token. We do *not* need to account
        # for splitting on spaces here since that's accounted *outside* of
        # each `token`. This should be exactly correct as long as BERT's
        # tokenizer doesn't change the number of bytes in a token during
        # tokenization; we check for this condition in
        # `num_mismatched_tokens`.
        internal_byte_offset += subtoken_len
      berttok_wordpieces.append(inside_berttok_wordpieces)
      berttok_starts.append(inside_berttok_starts)
      berttok_limits.append(inside_berttok_limits)
    if mismatched_tokens:
      logging.info("Have %d mismatched tokens of %d (%d bytes off): %s",
                   len(mismatched_tokens), num_tokens, mismatch_bytes,
                   " ".join(mismatched_tokens))

    # Finally, we post-process the result to ensure
    # that we don't split special tokens, taking care to preserve the mapping
    # from `text` bytes to wordpiece indices.
    wordpieces_out = []
    start_offsets_out = []
    end_offsets_out = []
    offset_to_wp_out = {}

    curr_offset = 0
    token_count = 0
    # `token`:str are the whitespace-delimited tokens from `tokenize`.
    # `wps`:List[int] are the wordpiece ids from `BertTokenizer` within each
    #   `token`.
    # `wp_starts`:List[int] are the byte starts for each wordpiece.
    # `wp_limits`:List[int] are the byte limits for each wordpiece.
    for token, wps, wp_starts, wp_limits in zip(whitespace_tokens,
                                                berttok_wordpieces,
                                                berttok_starts, berttok_limits):
      # If it is a special token (e.g. [UNK]), don't tokenize into wordpieces.
      if self.is_special_token(token):
        vocab_id = self.get_vocab_id(token)
        # Iterate over the text byte offsets covered by this token and
        # associate each with this wordpice index.
        wp_index = len(wordpieces_out)
        for j in range(0, data.byte_len(token)):
          offset_to_wp_out[j + curr_offset] = wp_index
        if vocab_id > -1:
          wordpieces_out.append(vocab_id)
        else:
          vocab_id = self.get_vocab_id("[UNK]")
          wordpieces_out.append(vocab_id)
        start_offsets_out.append(curr_offset)
        end_offsets_out.append(curr_offset + data.byte_len(token) - 1)
      else:
        # Not a special token, so keep the wordpieces.
        # `i`: index of the current wordpiece *within* the whitespace `token`.
        # `wp_start`: byte-wise start of this token within whitespace `token`.
        # `wp_limit`: byte-wise end index of this token within whitespace
        #             `token`.
        for i, (wp_start, wp_limit) in enumerate(zip(wp_starts, wp_limits)):
          # `j`: byte offset *within* the current whitespace token.
          for j in range(0, data.byte_len(token)):
            if j >= wp_start and j < wp_limit:
              offset_to_wp_out[j + curr_offset] = len(wordpieces_out) + i
        wordpieces_out.extend(wps)
        start_offsets_out.extend([k + curr_offset for k in wp_starts])
        end_offsets_out.extend([k + curr_offset - 1 for k in wp_limits])
      curr_offset += data.byte_len(token)

      # Splitting space is one byte as defined in `tokenize` function.
      token_count += 1
      if token_count < len(whitespace_tokens):
        offset_to_wp_out[curr_offset] = -1
        curr_offset += 1
    assert len(start_offsets_out) == len(wordpieces_out)
    assert len(start_offsets_out) == len(end_offsets_out)
    return wordpieces_out, start_offsets_out, end_offsets_out, offset_to_wp_out

  def is_special_token(self, token):
    """Is this a special token reserved for BERT or TyDi QA modeling?"""

    # NOTE: These must also be in the system's vocabulary file, which by default
    # is `mbert_modified_vocab.txt`, which is the original mBERT vocabulary
    # with some special tokens specific to our system added in the reserved
    # (unused) vocabulary space.
    special_tokens = set([
        "[CLS]", "[SEP]", "[PAD]", "[Q]", "[YES]", "[NO]", "[NoLongAnswer]",
        "[NoShortAnswer]", "[SA]", "[/SA]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"
    ])
    if token in special_tokens:
      return True
    if token.startswith("[Paragraph=") or token.startswith("[ContextId="):
      return True
    return False

  def get_vocab_id(self, special_token):
    """Gets the vocab id of a `special_token`."""
    if special_token in self.vocab:
      return self.vocab[special_token]
    else:
      raise "Unrecognized special token: '{}'".format(special_token)

  def _flatten_inner(self, seq):
    """Creates a 2D nested list from 3D, squeezing the inner dims."""
    result = []
    for subseq in seq:
      # `subseq` is seq[i], a 2D list.
      inner = []  # `inner` will remain a 1D list.
      for subsubseq in subseq:
        # `subsubseq` is seq[i][j], a 1D list.
        inner.extend(subsubseq)
      result.append(inner)
    return result
