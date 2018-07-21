# [im2latex](https://untrix.github.io/i2l/)

This project holds source code of a neural network model (an encoder-decoder style neural sequence generator with soft visual attention) solving the im2latex request for research of [OpenAI](https://openai.com/) with the best BLEU score (89%) reported at the time of this writing.

## Introduction
Please visit the [project website](https://untrix.github.io/i2l/) for a full description of what this research work is all about. The website includes links to the [research paper](https://arxiv.org/abs/1802.05415) that this code implements, as well as the dataset, visuals etc.

>**Before you start with the code, be sure to [read the paper](https://arxiv.org/abs/1802.05415)** first. This code supports the paper, not the other way around! The main objective for providing this code is to enable AI researchers to iterate atop this work.

## Platform

1. This souce code is written in the python 2.7 eco-system and uses Tensorflow 1.3 (the GPU version), Keras, Pandas, Jupyter Notebook, scipy, h5py and nltk to name a few. This list is not complete, so you'll probably need to import additional python packages as you work with this code. I used [anaconda](https://www.anaconda.com/) to setup a virtual python environment for this project and highly recommend it.
1. All the experiments were run on Linux (Ubuntu 16.04). All the preprocessing and postprocessing code should work on a Mac as well, but training requires GPUs.

## Hardware

Training was carried out on two Nvidia GeForce 1080Ti cards in [parallel](https://github.com/petewarden/tensorflow_makefile/blob/master/tensorflow/models/image/cifar10/cifar10_multi_gpu_train.py) and all commandline options here assume that. However the code is written to work on any number of cards in parallel, so you should be able to use just 1 or more cards. That said, this is a research project, not a finished product; therefore be prepared to poke at the code should you run into issues. 

You should ideally use two or more GeForce 1080Ti type cards (3584 CUDA cores and 11GB memory) to fully train the model in 2-2.5 days. One 1080Ti will take about 4-5 days to reach the accuracy specified in the paper. If you want to do serious research on a model of this size, I'd recommend 4 GPUs if you can afford it, because research requires 100s of runs (even 1000s). BTW, here's the [parts-list](https://pcpartpicker.com/user/Sumeet0/saved/#view=gFbvVn) of the hardware that I created for this project.

<a href="https://untrix.github.io/i2l/hardware.html"><img src="https://untrix.github.io/i2l/images/IMG_0008%20(cropped).JPG" width="100"/></a>
<a href="https://untrix.github.io/i2l/hardware.html"><img src="https://untrix.github.io/i2l/images/IMG_0007%20(cropped).JPG" width="100"/></a>
<a href="https://untrix.github.io/i2l/hardware.html"><img src="https://untrix.github.io/i2l/images/IMG_0006%20(cropped).JPG" width="100"/></a>
<a href="https://untrix.github.io/i2l/hardware.html"><img src="https://untrix.github.io/i2l/images/IMG_0004.JPG" width="100"/></a>
<!-- <a href="https://untrix.github.io/i2l/hardware.html"><img src="https://untrix.github.io/i2l/images/IMG_0005%20(cropped).JPG" width="100"/></a> -->
<!-- <a href="https://untrix.github.io/i2l/hardware.html"><img src="https://untrix.github.io/i2l/images/IMG_0003%20(cropped).JPG" width="100"/></a>
<a href="https://untrix.github.io/i2l/hardware.html"><img src="https://untrix.github.io/i2l/images/IMG_0002%20(cropped).JPG" width="100"/></a>
<a href="https://untrix.github.io/i2l/hardware.html"><img src="https://untrix.github.io/i2l/images/IMG_0001%20(cropped).JPG" width="100"/></a> -->

Below are instructions on how to train and evaluate this model. If you train it on my preprocessed dataset (see below), then you should expect to get results very similar to the ones in the [paper](https://arxiv.org/abs/1802.05415).

## Fork this repo

Fork and clone this repo. The instructions that follow assume that you've cloned this repo into `$REPO_DIR=~/im2latex`.

## Dataset Processing

LaTeX formulas need to be normalized, tokenized, vocabulary extracted and everything placed in a format that the code expects. You can either download all the processed data (800+ MB) from our [website](https://untrix.github.io/i2l/) or process it yourself. If you want to process the data yourself (why? maybe because you want to tokenize it in your own way and create your own vocabulary, or perhaps you want to use your own formula list) then go through the five preprocessing steps under src/preprocessing. There are five jupyter notebooks with inbuilt instructions that normalize, render, tokenize, filter and format the dataset. Run those notebooks and produce the data-files that the pipeline produces. By default the pipeline produces data-bins of size 56 and stores them in a sub-directory `training_56`.

Ensure that all data produced or downloaded above is placed under $REPO_DIR/data. So, for e.g. if your data is in $DATA_DIR you could just create a link: `ln -s $DATA_DIR $REPO_DIR/data`.

## Hyperparams

The code is very flexible in that one can create several model variants by simply changing the configuration params/hyperparams. That's why there are many params (about 86 top-level ones and more if you count the nested ones) and they need to be managed. All the commonly changed params are either set in the main script `$REPO_DIR/src/run.py`, or supplied to it from the command-line. `$REPO_DIR/src/model/hyper_params.py` declares all parameters used throughout the project (including those in `run.py`) and you can also set param values there. However, as a safeguard the code won't allow you to set different values in two places; therefore it is most convenient to set a param in only one location in the code.

## Train the Model
In the file run.py, set num_gpus to the number of gpus you want to train on.

### Train I2L-STRIPS

    `cd $REPO_DIR`
    # Open run.py in an editor and change the parameter REGROUP_IMAGE to `(4,1)`
    `./run.py -a 0.0001 -e -1 -b 28 -p -v -1 -i 2 --r-lambda 0.00005 --raw-data-folder ../data/training_56 --logdir-tag STRIPS`

This will run indefinitely, creating data and printing logs inside a newly created "logdir" (timestamp + logdir-tag) created under ./tb_metrics - e.g. `logdir=./tb_metrics/2017-12-21 02-20-10 PST STRIPS/`. Run logs will be written to file called training.log - for e.g. `./tb_metrics/2017-12-21 02-20-10 PST STRIPS/store/training.log`. 

Tensorboard event files are created under the logdir, and Tensorboard charts are the way to view training progress. A helpful script for running tensorboard is `$REPO_DIR/bin/runTensorboard.bash`.

Lots of other files are created under logdir or under "storedir" (=logdir/store or logdir/store_2 or logdir/store_3 etc.):

1. Training and validation predictions and some other data is dumped into `storedir/training_*.h5` and `storedir/training_*.h5`files. These can be visualized and processed using notebooks under $REPO_DIR/tools.
2. All hyperparams and arguments are dumped into `storedir/*.pkl` files.
3. Model snapshots are dumped under logdir. You can resume your model from a snapshot/checkpoint. If you stop training and resume from a snapshot, then a new storedir is created under logdir - e.g. `logdir/store_2`, then `logdir_store_3` and so on. You can resume as many times as you'd like.
4. Validation cycle is run periodically based on an algorithm inside `$REPO_DIR/src/train_multi_gpu.py`. But you can also manually run it by hitting control-C (i.e. sending SIGINT to run.py). A snapshot is first created, then the validation cycle starts running and tensorboard events are emitted for it. After a validation epoch is completed, the training cycle resumes.
4. The training runs indefinitely and can be stopped or influenced by sending the following signals to run.py:
    * SIGTERM: Dump snapshot, run one validation cycle (epoch) and then stop.
    * SIGINT (control-C): Dump snapshot, run a validation cycle, then resume training.
    * SIGUSR1: Stop training.
    * The signal's action is performed at the next opportunity - usually after the current training step is completed. If you send the same signal again before its action is taken, then the signal simply gets "undone". You can use this feature to recover if you sent the signal accidentally. For e.g. if you hit control-C by mistake, you can undo that action by immediately hitting control-C again.
5. Tensorboard metrics and top-level hyperparms across various runs can be compared using: [$REPO_DIR/src/tools/eval_runs.ipynb](https://github.com/untrix/im2latex/blob/master/src/tools/eval_runs.ipynb).
6. Attention scan visualization is available via the notebooks: [$REPO_DIR/src/tools/visualize.ipynb](https://github.com/untrix/im2latex/blob/master/src/tools/visualize.ipynb) and [$REPO_DIR/src/tools/disp_alpha.ipynb](https://github.com/untrix/im2latex/blob/master/src/tools/disp_alpha.ipynb). Also, be sure to checkout more functionality available in `$REPO_DIR/src/commons/viz_commons.py`.
7. You can diff hyperparams and args of various runs via the notebook: [$REPO_DIR/src/tools/diff_params.ipynb](https://github.com/untrix/im2latex/blob/master/src/tools/diff_params.ipynb).
8. More visualization is available using the notebook [$REPO_DIR/src/tools/disp.ipynb](https://github.com/untrix/im2latex/blob/master/src/tools/disp.ipynb) and functions in `$REPO_DIR/src/commons/pub_commons.py`.
9. Data extraction examples are available in [$REPO_DIR/src/tools/publishing.ipynb](https://github.com/untrix/im2latex/blob/master/src/tools/publishing.ipynb), [$REPO_DIR/src/tools/sample_preds.ipynb](https://github.com/untrix/im2latex/blob/master/src/tools/sample_preds.ipynb) and [$REPO_DIR/src/tools/sample_strs.ipynb](https://github.com/untrix/im2latex/blob/master/src/tools/sample_strs.ipynb)


### Train I2L-NOPOOL

    `cd $REPO_DIR`
    # Open run.py in an editor and change the parameter REGROUP_IMAGE to `None`
    `./run.py -a 0.0001 -e -1 -b 28 -v -1 -i 2 --r-lambda 0.00005 --raw-data-folder ../data/training_56 --logdir-tag NOPOOL`

## Evaluate Model

The model is evaluated by running an evaluation cycle on a snapshot. Evaluation cycle loads the model from a snapshot, runs one epoch of the desired dataset (validation or test dataset or even training dataset if you so desire) and produces various metrics which can be viewed in Tensorboard and processed and visualized using the scripts and notebooks mentioned before.

1. Decide which step you would like to evaluate. The step must be one on which a validation cycle was run during training because only those steps have snapshots - say step 00168100.
2. Create a new logdir under the original logdir - say test_runs/step_00168100:

    `cd $REPO_DIR/tb_metrics/2017-12-21 02-20-10 PST STRIPS/`  
    `mkdir -p test_runs/step_00168100`

3. Copy the snapshot and checkpoint files into the new logdir:

    `cp snapshot-00168100* test_runs/step_00168100`  
    `cp checkpoints_list test_runs/step_00168100`

4. Edit the checkpoints_list file such that it only has the desired step in it:

    `cd test_runs/step_00168100`  
    `# Edit checkpoints_list`  
    `more checkpoints_list`  

        model_checkpoint_path: "snapshot-00168100"
        all_model_checkpoint_paths: "snapshot-00168100"

5. Execute the script

    `cd $REPO_DIR`  
    `./run.py -a 0.0001 -e -1 -b 28 -v -1 -i 2 --r-lambda 0.00005 --raw-data-folder ../data/training_56 --restore ./tb_metrics/2017-12-21 02-20-10 PST STRIPS/test_runs/step_00168100 --test --save-all-eval`

        NOTE: Ensure that after training, you didn't change any hyper-parameters inside your code that alter the model materially from the snapshot.
    This will create the same files as the training-cycle except that owing to the `--save-all-eval` option the entire datasets's predictions will be dumped to file test_168100.h5 (instead of just one batch's predictions which is the case with training). Use the same scripts and notebooks mentioned before to view and process the results - or just look at the tensorboard charts. Note: providing the flag `--validate` instead of `--test` does the same thing except that it changes the names in various places to have the string 'test' instead of 'validation'.

### Evaluate Image Match (Optional)
The above steps will give you corpus BLEU score (testing/bleu2) and edit distance among other metrics. If you also want the visual match metric, then execute instructions in the notebook [$REPO_DIR/src/postprocessing/evaluate_images.ipynb](https://github.com/untrix/im2latex/blob/master/src/postprocessing/evaluate_images.ipynb).

### Running a Trained Model
We do not provide a graph/code to deploy a model. However, we have cobbled together an way to quickly run a model snapshot against a list of images. Follow the notebook [src/preprocessing/prepare_inferencing_dataset.ipynb](https://github.com/untrix/im2latex/blob/master/src/preprocessing/prepare_inferencing_dataset.ipynb) to create a special dataset for this purpose and then the notebook [src/tools/visualize.ipynb](https://github.com/untrix/im2latex/blob/master/src/tools/visualize.ipynb) for extracting the predictions (from underlying h5py files).
