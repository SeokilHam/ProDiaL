# Parameter Efficient Mamba Tuning via Projector-targeted Diagonal-centric Linear Transformation

**Authors:** Seokil Ham, Hee-Seon Kim, Sangmin Woo, Changick Kim

**Paper:** https://arxiv.org/abs/2411.15224

## Installation

Requirements:
- Linux
- NVIDIA GPU
- PyTorch 1.12+
- CUDA 11.6+

```sh
conda create -n ProDiaL python=3.9
pip install -r requirements.txt
```

## Usage

### 1. Set Hyperparameters in ProDiaL

``` python
CUDA_VISIBLE_DEVICES=0 python train.py \
    --model_path="state-spaces/mamba-130m" \
    --tokenizer_path="EleutherAI/gpt-neox-20b" \
    --instruction_datasets="[hellaswag]" \
    --output_dir="outputs" \
    --random_seed=42 \
    --sequence_max_length=512 \
    --save_steps=1 \
    --batch_size=4 \
    --cache_dir="huggingface" \
    --num_epochs=21 \
    --weight_decay=0.01 \
    --learning_rate=1e-4 \
    --dropout_rate=0.1 \
    --logging_steps=100 \
    --config_path="configs/130m" \
    --r_b1=768 \
    --r_b2=1536 \
    --off_diagonal_rank=16 \
```

### Run bash file

```
bash train_hellaswag.sh
```


## Evaluations

```sh
conda create -n eval_ProDiaL python=3.9
pip install lm-eval==0.4.2
pip install causal_conv1d-1.5.0.post8
pip install mamba-ssm==1.2.0.post1
```


Run evaluation with (more documentation at the [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/tree/big-refactor) repo):
``` sh
python evals/lm_harness_eval.py --model mamba_ssm --tasks hellaswag --device cuda --batch_size 256 --seed 42 --model_args pretrained=state-spaces/mamba-130m
```

For evaluating multiple checkpoints at once,
``` sh
bash eval_hellaswag.sh
```


## Citation

```
@article{ham2024parameter,
  title={Parameter Efficient Mamba Tuning via Projector-targeted Diagonal-centric Linear Transformation},
  author={Ham, Seokil and Kim, Hee-Seon and Woo, Sangmin and Kim, Changick},
  journal={arXiv preprint arXiv:2411.15224},
  year={2024}
}
```

## Reference
This codebase was partially adapted from the following repositories:

- (https://github.com/state-spaces/mamba) (Apache 2.0 License)
- (https://github.com/sangHa0411/Llama-Instruction-Tuning) (MIT License)

We thank the authors for open-sourcing their work.
