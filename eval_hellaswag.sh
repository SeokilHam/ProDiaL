#!/bin/bash
#SBATCH -p part1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH -e stderr_%j.txt
#SBATCH --gres=gpu:1

# Insturction-Tuning command example
export CUDA_VISIBLE_DEVICES=0
python evals/lm_harness_eval.py --model mamba_ssm --tasks hellaswag --device cuda --batch_size 256 --seed 42 --trust_remote_code --model_args pretrained="outputs/checkpoint-10000/"

python evals/lm_harness_eval.py --model mamba_ssm --tasks hellaswag --device cuda --batch_size 256 --seed 42 --trust_remote_code --model_args pretrained="outputs/checkpoint-20000/"

python evals/lm_harness_eval.py --model mamba_ssm --tasks hellaswag --device cuda --batch_size 256 --seed 42 --trust_remote_code --model_args pretrained="outputs/checkpoint-30000/"

python evals/lm_harness_eval.py --model mamba_ssm --tasks hellaswag --device cuda --batch_size 256 --seed 42 --trust_remote_code --model_args pretrained="outputs/checkpoint-40000/"

python evals/lm_harness_eval.py --model mamba_ssm --tasks hellaswag --device cuda --batch_size 256 --seed 42 --trust_remote_code --model_args pretrained="outputs/checkpoint-50000/"

python evals/lm_harness_eval.py --model mamba_ssm --tasks hellaswag --device cuda --batch_size 256 --seed 42 --trust_remote_code --model_args pretrained="outputs/checkpoint-60000/"

python evals/lm_harness_eval.py --model mamba_ssm --tasks hellaswag --device cuda --batch_size 256 --seed 42 --trust_remote_code --model_args pretrained="outputs/checkpoint-70000/"

python evals/lm_harness_eval.py --model mamba_ssm --tasks hellaswag --device cuda --batch_size 256 --seed 42 --trust_remote_code --model_args pretrained="outputs/checkpoint-80000/"

python evals/lm_harness_eval.py --model mamba_ssm --tasks hellaswag --device cuda --batch_size 256 --seed 42 --trust_remote_code --model_args pretrained="outputs/checkpoint-90000/"

python evals/lm_harness_eval.py --model mamba_ssm --tasks hellaswag --device cuda --batch_size 256 --seed 42 --trust_remote_code --model_args pretrained="outputs/checkpoint-100000/"

python evals/lm_harness_eval.py --model mamba_ssm --tasks hellaswag --device cuda --batch_size 256 --seed 42 --trust_remote_code --model_args pretrained="outputs/checkpoint-110000/"

python evals/lm_harness_eval.py --model mamba_ssm --tasks hellaswag --device cuda --batch_size 256 --seed 42 --trust_remote_code --model_args pretrained="outputs/checkpoint-120000/"

python evals/lm_harness_eval.py --model mamba_ssm --tasks hellaswag --device cuda --batch_size 256 --seed 42 --trust_remote_code --model_args pretrained="outputs/checkpoint-130000/"

python evals/lm_harness_eval.py --model mamba_ssm --tasks hellaswag --device cuda --batch_size 256 --seed 42 --trust_remote_code --model_args pretrained="outputs/checkpoint-140000/"

python evals/lm_harness_eval.py --model mamba_ssm --tasks hellaswag --device cuda --batch_size 256 --seed 42 --trust_remote_code --model_args pretrained="outputs/checkpoint-150000/"

python evals/lm_harness_eval.py --model mamba_ssm --tasks hellaswag --device cuda --batch_size 256 --seed 42 --trust_remote_code --model_args pretrained="outputs/checkpoint-160000/"

python evals/lm_harness_eval.py --model mamba_ssm --tasks hellaswag --device cuda --batch_size 256 --seed 42 --trust_remote_code --model_args pretrained="outputs/checkpoint-170000/"

python evals/lm_harness_eval.py --model mamba_ssm --tasks hellaswag --device cuda --batch_size 256 --seed 42 --trust_remote_code --model_args pretrained="outputs/checkpoint-180000/"

python evals/lm_harness_eval.py --model mamba_ssm --tasks hellaswag --device cuda --batch_size 256 --seed 42 --trust_remote_code --model_args pretrained="outputs/checkpoint-190000/"

python evals/lm_harness_eval.py --model mamba_ssm --tasks hellaswag --device cuda --batch_size 256 --seed 42 --trust_remote_code --model_args pretrained="outputs/checkpoint-200000/"

# python evals/lm_harness_eval.py --model mamba_ssm --tasks hellaswag --device cuda --batch_size 256 --seed 42 --trust_remote_code --model_args pretrained="outputs/checkpoint-210000/"

# python evals/lm_harness_eval.py --model mamba_ssm --tasks hellaswag --device cuda --batch_size 256 --seed 42 --trust_remote_code --model_args pretrained="outputs/checkpoint-220000/"

# python evals/lm_harness_eval.py --model mamba_ssm --tasks hellaswag --device cuda --batch_size 256 --seed 42 --trust_remote_code --model_args pretrained="outputs/checkpoint-230000/"

# python evals/lm_harness_eval.py --model mamba_ssm --tasks hellaswag --device cuda --batch_size 256 --seed 42 --trust_remote_code --model_args pretrained="outputs/checkpoint-240000/"

# python evals/lm_harness_eval.py --model mamba_ssm --tasks hellaswag --device cuda --batch_size 256 --seed 42 --trust_remote_code --model_args pretrained="outputs/checkpoint-250000/"

# python evals/lm_harness_eval.py --model mamba_ssm --tasks hellaswag --device cuda --batch_size 256 --seed 42 --trust_remote_code --model_args pretrained="outputs/checkpoint-260000/"

# python evals/lm_harness_eval.py --model mamba_ssm --tasks hellaswag --device cuda --batch_size 256 --seed 42 --trust_remote_code --model_args pretrained="outputs/checkpoint-270000/"

# python evals/lm_harness_eval.py --model mamba_ssm --tasks hellaswag --device cuda --batch_size 256 --seed 42 --trust_remote_code --model_args pretrained="outputs/checkpoint-280000/"

# python evals/lm_harness_eval.py --model mamba_ssm --tasks hellaswag --device cuda --batch_size 256 --seed 42 --trust_remote_code --model_args pretrained="outputs/checkpoint-290000/"

# python evals/lm_harness_eval.py --model mamba_ssm --tasks hellaswag --device cuda --batch_size 256 --seed 42 --trust_remote_code --model_args pretrained="outputs/checkpoint-300000/"
