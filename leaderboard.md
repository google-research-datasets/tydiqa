# Leaderboard Submissions

These instructions assume you have a working system for the TyDi QA primary taks
that you wish to submit to the TyDi QA leaderboard. To get up to this point,
generally you would:

1.  Run the provided baseline system to sanity check that you can reproduce the
    baseline results. [baseline/README.md]
2.  Modify the baseline system or your existing code to improve it.
    [baseline/README.md]
3.  Evaluate locally on the dev set to measure the quality of your system.
    [README.md]
4.  Package your code and fine-tuned model parameters into a Docker container
    and make it public via Google Cloud Storage, available for free
    (instructions in this file).
5.  Submit a trial run to the TyDi QA leaderboard system to make sure your code
    runs. You have unlimited attempts to do this.
6.  Submit an official run to the TyDi QA leaderboard system. You may only do
    this once a week.

## Creating a Docker Container with Your Submission

First, make sure that you have set up a profile as instructed on the
[TyDi QA site](https://google-research-datasets.github.io/tydiqa).

You must submit your model as a Docker image. You are allowed to use whatever
software dependencies you want, but those dependencies must be included in your
Docker image. Let's say your model is a Tensorflow model. For this, you can use
the official tensorflow Docker container as the base container image:

```dockerfile
FROM tensorflow/tensorflow:latest
ADD tydiqa_model /tydiqa_model/
```

The first line of this Dockerfile says to use the official Tensorflow Docker
image as the starting point of the image. The second line says to add the
contents of a directory called `tydiqa_model` to a folder called `/tydiqa_model`
inside the image. Read the [Docker manual](https://docs.docker.com/) for more
details on how to use Dockerfiles.

The folder `/tydi_model` is expected to contain a script called `submission.sh`.
The TyDi QA test set is in a number of gzipped jsonl files with exactly the same
format as the released development set. During evaluation, the
`/tydiqa_model/submission.sh` script contained in your Docker image will be
called with an argument `input_path` that matches the files containing the test
set. Another argument `output_path` tells your code where to write predictions
for each of the input examples. For a complete description of the prediction
format, please see the [evaluation script](tydi_eval.py).

Below, we give an example `submission.sh` that works with the
[`run_tydi.py`](https://github.com/google-research-datasets/tydiqa/tree/master/baseline/run_tydi.py)
executable released as part of the TyDi QA primary tasks baselines.

```shell
#!/bin/bash
#
# submission.sh: The script to be launched in the Docker image.
#
# Usage: submission.sh <input_path> <output_path>
#   input_path: File pattern (e.g. <input dir>/tydiqa-test-??.jsonl.gz).
#   output_path: Path to JSON file containing predictions (e.g. predictions.json).
#
# Sample usage:
#   submission.sh input_path output_path

INPUT_PATH=$1
OUTPUT_PATH=$2

# YOUR CODE HERE!
#
# For example, to run the baseline system from:
#  https://github.com/google-research-datasets/tydiqa/tree/master/baseline/run_tydi.py)

python -m tydiqa.baseline.run_tydi
  --predict_file=${INPUT_PATH} \
  --output_prediction_file=${OUTPUT_PATH} \
  --init_checkpoint=<path_to_model_params_within_docker_image> \
  --bert_config_file=<path_to_mbert_config_json_within_docker_image>
```

When you upload your Docker image to the
[TyDi QA leaderboard](https://google-research-datasets.github.io/tydiqa),
`/tydiqa_model/submission.sh` will be called with `input_path` and `output_path`
arguments that point to the test data input, and the output file that will be
fed to the evaluation script, respectively.

Remember that each team is only allowed to make one submission per week to the
TyDi QA leaderboard. But you are allowed to run as many times as you like on the
small sample that we provide so that you can test your uploaded Docker image.

## Uploading to a Google Cloud Storage Container

| IMPORTANT: Once you have created your Docker image and uploaded it to the  |
: TyDi QA competition site, you must grant our service account read access.  :
: Otherwise your Docker images will be private and we won't be able to run   :
: them.                                                                      :
| :------------------------------------------------------------------------- |
| 1. Go to the storage tab in the gcloud console.                            |
| 2. Locate the artifacts bucket. It should have a name like                 |
: `artifacts.<project-name >.appspot.com`                                    :
| 3. Click the dropdown for the bucket and select "Edit Bucket Permissions". |
| 4. Grant Storage Object Viewer permissions to the following user:          |
: `mljam-compute@mljam-205019.iam.gserviceaccount.com`                       :
