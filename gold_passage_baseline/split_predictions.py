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
"""Splits a prediction file containing all languages into multiple files.

In order to evaluate the TyDiQA-GoldP task, each language must be evaluated
separately. However, much existing code expects a single training set and a
single evaluation set, so we provide this script to help with splitting
post hoc.

This script requires Python 3.
"""

import collections
import json
import os

from absl import app
from absl import flags
from absl import logging

FLAGS = flags.FLAGS

flags.DEFINE_string("input_json", None, "SQuAD-format predicions.json file.")
flags.mark_flag_as_required("input_json")

flags.DEFINE_string(
    "output_dir", None,
    "Output directory where individual language prediction files will be "
    "written.")
flags.mark_flag_as_required("output_dir")

flags.DEFINE_string(
    "lang_output_json_pattern", "tydiqa-goldp-dev-predictions-%s.json",
    "Per-language output file pattern. The language name will "
    "be inserted into the '%s' and files will be written in `output_dir`. ")


def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  lang_list = set([
      "english", "arabic", "bengali", "finnish", "indonesian", "swahili",
      "korean", "russian", "telugu"
  ])

  data_by_lang = collections.defaultdict(dict)

  with open(FLAGS.input_json, "r") as f:
    json_dict = json.load(f)
  for example_id, answer in json_dict.items():
    cols = example_id.split("-")
    if len(cols) < 2:
      raise ValueError("Example ID '%s' does not start with a valid language." %
                       example_id)
    lang = cols[0]
    if lang not in lang_list:
      raise ValueError(
          "Example ID '%s' does not start with a valid language: '%s'" %
          (example_id, lang))
    data_by_lang[lang][example_id] = answer

  for lang, data in data_by_lang.items():
    if "%s" not in FLAGS.lang_output_json_pattern:
      raise ValueError(
          "Expected placeholder '%s' in `lang_output_json_pattern`.")
    lang_for_filename = lang
    if lang == "english":
      # Make sure people don't accidentally include English in their
      # overall scores.
      lang_for_filename = "english-DO-NOT-AVERAGE"
    filename = FLAGS.lang_output_json_pattern % lang_for_filename
    path = os.path.join(FLAGS.output_dir, filename)
    logging.info("Writing %d %s answers to %s", len(data), lang, path)
    with open(path, "w") as f:
      json.dump(data, f, indent=4)


if __name__ == "__main__":
  app.run(main)
