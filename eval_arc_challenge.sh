#!/bin/bash
#SBATCH -p part1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH -e stderr_%j.txt
#SBATCH --gres=gpu:1

# Insturction-Tuning command example
export CUDA_VISIBLE_DEVICES=0
python evals/lm_harness_eval.py --model mamba_ssm --tasks arc_challenge --device cuda --batch_size 256 --seed 42 --model_args pretrained="outputs/checkpoint-5000"

python evals/lm_harness_eval.py --model mamba_ssm --tasks arc_challenge --device cuda --batch_size 256 --seed 42 --model_args pretrained="outputs/checkpoint-10000/"

python evals/lm_harness_eval.py --model mamba_ssm --tasks arc_challenge --device cuda --batch_size 256 --seed 42 --model_args pretrained="outputs/checkpoint-15000/"

python evals/lm_harness_eval.py --model mamba_ssm --tasks arc_challenge --device cuda --batch_size 256 --seed 42 --model_args pretrained="outputs/checkpoint-20000/"

python evals/lm_harness_eval.py --model mamba_ssm --tasks arc_challenge --device cuda --batch_size 256 --seed 42 --model_args pretrained="outputs/checkpoint-25000/"

python evals/lm_harness_eval.py --model mamba_ssm --tasks arc_challenge --device cuda --batch_size 256 --seed 42 --model_args pretrained="outputs/checkpoint-30000/"

python evals/lm_harness_eval.py --model mamba_ssm --tasks arc_challenge --device cuda --batch_size 256 --seed 42 --model_args pretrained="outputs/checkpoint-35000/"

python evals/lm_harness_eval.py --model mamba_ssm --tasks arc_challenge --device cuda --batch_size 256 --seed 42 --model_args pretrained="outputs/checkpoint-40000/"

python evals/lm_harness_eval.py --model mamba_ssm --tasks arc_challenge --device cuda --batch_size 256 --seed 42 --model_args pretrained="outputs/checkpoint-45000/"

python evals/lm_harness_eval.py --model mamba_ssm --tasks arc_challenge --device cuda --batch_size 256 --seed 42 --model_args pretrained="outputs/checkpoint-50000/"

python evals/lm_harness_eval.py --model mamba_ssm --tasks arc_challenge --device cuda --batch_size 256 --seed 42 --model_args pretrained="outputs/checkpoint-55000/"

python evals/lm_harness_eval.py --model mamba_ssm --tasks arc_challenge --device cuda --batch_size 256 --seed 42 --model_args pretrained="outputs/checkpoint-60000/"

python evals/lm_harness_eval.py --model mamba_ssm --tasks arc_challenge --device cuda --batch_size 256 --seed 42 --model_args pretrained="outputs/checkpoint-65000/"

python evals/lm_harness_eval.py --model mamba_ssm --tasks arc_challenge --device cuda --batch_size 256 --seed 42 --model_args pretrained="outputs/checkpoint-70000/"

python evals/lm_harness_eval.py --model mamba_ssm --tasks arc_challenge --device cuda --batch_size 256 --seed 42 --model_args pretrained="outputs/checkpoint-75000/"

python evals/lm_harness_eval.py --model mamba_ssm --tasks arc_challenge --device cuda --batch_size 256 --seed 42 --model_args pretrained="outputs/checkpoint-80000/"

python evals/lm_harness_eval.py --model mamba_ssm --tasks arc_challenge --device cuda --batch_size 256 --seed 42 --model_args pretrained="outputs/checkpoint-85000/"

python evals/lm_harness_eval.py --model mamba_ssm --tasks arc_challenge --device cuda --batch_size 256 --seed 42 --model_args pretrained="outputs/checkpoint-90000/"

python evals/lm_harness_eval.py --model mamba_ssm --tasks arc_challenge --device cuda --batch_size 256 --seed 42 --model_args pretrained="outputs/checkpoint-95000/"

python evals/lm_harness_eval.py --model mamba_ssm --tasks arc_challenge --device cuda --batch_size 256 --seed 42 --model_args pretrained="outputs/checkpoint-100000/"
