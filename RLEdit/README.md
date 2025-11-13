# RLEdit

Official repo for our paper `Reinforced Lifelong Editing for Language Models`.

## Initial Setup

Create a virtual environment and install the dependencies.

```shell
conda create -n rledit python==3.11
conda activate rledit
pip install -r requirements.txt
```

Download LLMs, and put their paths into `config/model/xxx.yaml`

## Quick Start

You can modify the GPU device number you want to use in the `model.py` file.

`run.sh` is a simple example that performs a 400x20 editing task on the ZSRE dataset using Llama-3-8B.

```shell
sh run.sh
```

## Configuration

The `run.sh` script is as follows:

```shell
python main.py dataset=fever model=llama-3-instruct editor=rledit num_seq=20
```

Below are the explanations for each argument:

* `dataset`: Dataset used for editing and testing.
* `model`: LLM used for editing and testing.
* `editor`: Method for editing and testing. Available methods include `rledit`, `mend`, and `malmen`. All of these methods are evaluated using lifelong editing tasks.
* `num_seq`: Number of editing batches. For example, in a 400x20 editing task, this value is 400.

Other configurable settings can be found in `config` folder., such as:

* `n_edits` in `config/dataset/zsre.yaml`: Number of knowledge samples edited per batch. For example, in a 400x20 editing task, this value is 20.
* `reg_coef` in `config/editor/rledit.yaml`: The regularization coefficient.
* `time_decay` in `config/editor/rledit.yaml`: Memory backtracking decay factor.
* `edit_modules` in `config/model/llama-3-instruct.yaml`: Layer indices of the LLM to be edited.

## Acknowledgement

Our code is based on [malmen](https://github.com/ChenmienTan/malmen). Thanks to their clear and understandable code!
