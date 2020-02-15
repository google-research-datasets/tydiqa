# Leaderboard Submissions

These instructions assume you have a working system for the TyDi QA primary
tasks that you wish to submit to the
[TyDi QA leaderboard](https://ai.google.com/research/tydiqa/).
To get up to this point, generally you would:

1.  Run the provided baseline system to sanity check that you can reproduce the
    baseline results. [baseline/README.md](baseline/README.md)
2.  Modify the baseline system or your existing code to improve it.
    [baseline/README.md](baseline/README.md)
3.  Evaluate locally on the dev set to measure the quality of your system.
    [README.md](README.md)
4.  Package your code and fine-tuned model parameters into a Docker container
    and make it public via Google Cloud Storage, available for free
    (instructions in this file).
5.  Submit a "Test attempt" to the TyDi QA leaderboard system to make sure your
    code runs. You may do this as many times as you like.
6.  Submit an "Official attempt" to the TyDi QA leaderboard system. You may only
    do this once per week.

## Submission Requirements

You must set up a profile as instructed on the
[TyDi QA site](https://ai.google.com/research/tydiqa/participate).

Your submission must be packaged as a Docker container that includes all of your
code and model files, and contains a file called `/tydiqa_model/submission.sh`
that runs your model on an input file. (See more details below.)
You are allowed to use whatever software dependencies you want, but those
dependencies must be included in your Docker image.

Your submission will be run on a GCloud instance of type
[n1-highmem-16](https://cloud.google.com/compute/docs/machine-types#n1_high-memory_machine_types)
that has access to a single Nvidia Tesla P100 GPU.

An "Offical attempt" of your submission must complete in under 24 hours.
A "Test attempt" must complete in under 20 minutes.

## Creating a Submission

This section will walk you through the steps to create and submit a system that
for the TyDi QA leaderboard competition. We will use the [baseline](baseline)
code as a running example.

Prerequesites:
* Sign up for a Google Cloud Platform account at http://cloud.google.com and
  create a Project.
* Install Cloud SDK via https://cloud.google.com/sdk. This is needed to interact
  with GCP using command-line tools such as `gcloud`.
* Install [Docker](https://www.docker.com/) via `sudo apt install docker.io` or
  an equivalent command for your platform. This is needed to build a Docker
  image for your model.
* Authenticate to GCloud with `gcloud auth login`.

First, let's assume you have a directory called [`baseline`](baseline) that
contains (at least) the following files:

```bash
$ ls baseline/
bert/
postproc.py
preproc.py
run_tydi.py
tf_io.py
tiny_dev_no_annotations.jsonl.gz
tydi_modeling.py
```

To create your Docker container, you will need to create a file in the current
directory called `Dockerfile` that lists all software dependencies and source
directories that should be included.
(See the [Docker manual](https://docs.docker.com/) for more details on how to
use Dockerfiles.)
For example, the following example `Dockerfile` does two things:
* Includes a version of Tensorflow that can run the baseline system on GPU.
* Adds the `baseline` directory, giving it the internal path `/tydiqa_model`.
  The name `/tydiqa_model` is required since our automated service will need to
  run your system via a file called `/tydiqa_model/submission.sh` (see below).

```dockerfile
FROM tensorflow/tensorflow:1.15.2-gpu-py3
ADD baseline /tydiqa_model/
```

Since your entire system must be packaged as a single Docker image, you'll need
to copy over any model data files you need:

```bash
$ mkdir baseline/model
$ cp /path/to/trained/model/bert_config.json baseline/
$ cp /path/to/trained/model/model.ckpt-5744.* baseline/
$ cp /path/to/trained/model/vocab.txt baseline/
```

Create a file in the `baseline/` directory called `submission.sh` that will
serve as the entry point for your system.
When you upload your submission via the
[TyDi QA website](https://ai.google.com/research/tydiqa/participate),
this is the file that will be run by our automated evaluation pipeline.
It will be called with two arguments:
* `input_path`, which will be a a gzipped file of json lines similar
  to the dev data, but without any "annotations" entries (so please ensure that
  your code does not expect them).
  See [tiny_dev_no_annotations.jsonl.gz](tiny_dev_no_annotations.jsonl.gz)
  as an example.
* `output_path`: the location of a file into which your code should write a file
  of predictions that will be subsequently passed to the evaluation script,
  [tydi_eval.py](tydi_eval.py).
  See the [tydi_eval.py](tydi_eval.py) source code for a complete description
  of the expected prediction format, or see
  [sample_prediction.jsonl](sample_prediction.jsonl) as an example.

```shell
#!/bin/bash
#
# submission.sh: The script that runs your system.
#
# Usage: submission.sh <input_path> <output_path>
#   input_path: File pattern (e.g. <input dir>/tydiqa-test-??.jsonl.gz).
#   output_path: Path to JSONL file containing predictions (e.g. predictions.jsonl).
#
# Sample usage:
#   submission.sh input_path output_path

INPUT_PATH=$1
OUTPUT_PATH=$2

# CALL YOUR CODE HERE! For example:
python3 /tydiqa_model/run_tydi.py \
  --predict_file=${INPUT_PATH} \
  --output_prediction_file=${OUTPUT_PATH} \
  --output_dir=/tydiqa_model/output \
  --bert_config_file=/tydiqa_model/model/bert_config.json \
  --init_checkpoint=/tydiqa_model/model/model.ckpt-5744 \
  --vocab_file=/tydiqa_model/model/vocab.txt \
  --do_predict
```

Build your Docker image:

```bash
$ docker_model_build_id=$(sudo docker build . | tail -n1 | cut -d " " -f 3)
$ echo $docker_model_build_id
da18d5be9d95
```

Optionally, test the Docker container locally to ensure that it works:

```bash
sudo docker run "${docker_model_build_id}" bash /tydiqa_model/submission.sh \
  /tydiqa_model/tiny_dev_no_annotations.jsonl.gz \
  /tydiqa_model/predictions.jsonl
```

Upload your docker image to GCloud. If your GCloud project's ID is
`my-project-1234`, then this would be:

```bash
gcloud config set project my-project-1234
gcloud builds submit --tag "gcr.io/my-project-1234/your-tydiqa-submission" .
```

**IMPORTANT:** Grant our evaluation service account read access to your Docker
images:
1. Visit the [Storage Browser](https://console.cloud.google.com/storage/browser)
   GCloud console.
2. Locate the "artifacts" bucket in the list; it should have a name like
   `artifacts.<project-name>.appspot.com`
3. Click the dropdown on the far right side of the bucket's row, and select
   "Edit Bucket Permissions".
4. Grant "Storage Object Viewer" permissions to the following user:
   `mljam-compute@mljam-205019.iam.gserviceaccount.com`

Visit the
[TyDi QA submission page](https://ai.google.com/research/tydiqa/participate)
and use the "Submit an Attempt" form to submit a "Test attempt".
The "Google Container Registry image name" will be the name you used above,
`gcr.io/my-project-1234/your-tydiqa-submission`. After you submit the form,
reload the page to see your attempt appear in your "Dashboard". The "Status"
column will say "pending" while the system is running, and will change to
either "success" or "error" when it ends.

Once you confirm that your "Test attempt" is successful, you can submit an
"Offical attempt" using the same form.
Remember that each team is only allowed to submit one "Official attempt" per
week to the leaderboard system, but you may submit as many "Test attempts" as
you like.
