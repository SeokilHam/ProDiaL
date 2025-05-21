#!/bin/bash
#SBATCH -p part1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH -e stderr_%j.txt
#SBATCH --gres=gpu:1

# Insturction-Tuning command example
export CUDA_VISIBLE_DEVICES=0
python evals/lm_harness_eval.py --model mamba_ssm --tasks winogrande --device cuda --batch_size 256 --seed 42 --model_args pretrained="outputs/checkpoint-1000/"

python evals/lm_harness_eval.py --model mamba_ssm --tasks winogrande --device cuda --batch_size 256 --seed 42 --model_args pretrained="outputs/checkpoint-2000/"

python evals/lm_harness_eval.py --model mamba_ssm --tasks winogrande --device cuda --batch_size 256 --seed 42 --model_args pretrained="outputs/checkpoint-3000/"

python evals/lm_harness_eval.py --model mamba_ssm --tasks winogrande --device cuda --batch_size 256 --seed 42 --model_args pretrained="outputs/checkpoint-4000/"

python evals/lm_harness_eval.py --model mamba_ssm --tasks winogrande --device cuda --batch_size 256 --seed 42 --model_args pretrained="outputs/checkpoint-5000/"

python evals/lm_harness_eval.py --model mamba_ssm --tasks winogrande --device cuda --batch_size 256 --seed 42 --model_args pretrained="outputs/checkpoint-6000/"

python evals/lm_harness_eval.py --model mamba_ssm --tasks winogrande --device cuda --batch_size 256 --seed 42 --model_args pretrained="outputs/checkpoint-7000/"

python evals/lm_harness_eval.py --model mamba_ssm --tasks winogrande --device cuda --batch_size 256 --seed 42 --model_args pretrained="outputs/checkpoint-8000/"

python evals/lm_harness_eval.py --model mamba_ssm --tasks winogrande --device cuda --batch_size 256 --seed 42 --model_args pretrained="outputs/checkpoint-9000/"

python evals/lm_harness_eval.py --model mamba_ssm --tasks winogrande --device cuda --batch_size 256 --seed 42 --model_args pretrained="outputs/checkpoint-10000/"

python evals/lm_harness_eval.py --model mamba_ssm --tasks winogrande --device cuda --batch_size 256 --seed 42 --model_args pretrained="outputs/checkpoint-11000/"

python evals/lm_harness_eval.py --model mamba_ssm --tasks winogrande --device cuda --batch_size 256 --seed 42 --model_args pretrained="outputs/checkpoint-12000/"

python evals/lm_harness_eval.py --model mamba_ssm --tasks winogrande --device cuda --batch_size 256 --seed 42 --model_args pretrained="outputs/checkpoint-13000/"

python evals/lm_harness_eval.py --model mamba_ssm --tasks winogrande --device cuda --batch_size 256 --seed 42 --model_args pretrained="outputs/checkpoint-14000/"

python evals/lm_harness_eval.py --model mamba_ssm --tasks winogrande --device cuda --batch_size 256 --seed 42 --model_args pretrained="outputs/checkpoint-15000/"

python evals/lm_harness_eval.py --model mamba_ssm --tasks winogrande --device cuda --batch_size 256 --seed 42 --model_args pretrained="outputs/checkpoint-16000/"

python evals/lm_harness_eval.py --model mamba_ssm --tasks winogrande --device cuda --batch_size 256 --seed 42 --model_args pretrained="outputs/checkpoint-17000/"

python evals/lm_harness_eval.py --model mamba_ssm --tasks winogrande --device cuda --batch_size 256 --seed 42 --model_args pretrained="outputs/checkpoint-18000/"

python evals/lm_harness_eval.py --model mamba_ssm --tasks winogrande --device cuda --batch_size 256 --seed 42 --model_args pretrained="outputs/checkpoint-19000/"

python evals/lm_harness_eval.py --model mamba_ssm --tasks winogrande --device cuda --batch_size 256 --seed 42 --model_args pretrained="outputs/checkpoint-20000/"

python evals/lm_harness_eval.py --model mamba_ssm --tasks winogrande --device cuda --batch_size 256 --seed 42 --model_args pretrained="outputs/checkpoint-21000/"

python evals/lm_harness_eval.py --model mamba_ssm --tasks winogrande --device cuda --batch_size 256 --seed 42 --model_args pretrained="outputs/checkpoint-22000/"

python evals/lm_harness_eval.py --model mamba_ssm --tasks winogrande --device cuda --batch_size 256 --seed 42 --model_args pretrained="outputs/checkpoint-23000/"

python evals/lm_harness_eval.py --model mamba_ssm --tasks winogrande --device cuda --batch_size 256 --seed 42 --model_args pretrained="outputs/checkpoint-24000/"

python evals/lm_harness_eval.py --model mamba_ssm --tasks winogrande --device cuda --batch_size 256 --seed 42 --model_args pretrained="outputs/checkpoint-25000/"

python evals/lm_harness_eval.py --model mamba_ssm --tasks winogrande --device cuda --batch_size 256 --seed 42 --model_args pretrained="outputs/checkpoint-26000/"

python evals/lm_harness_eval.py --model mamba_ssm --tasks winogrande --device cuda --batch_size 256 --seed 42 --model_args pretrained="outputs/checkpoint-27000/"

python evals/lm_harness_eval.py --model mamba_ssm --tasks winogrande --device cuda --batch_size 256 --seed 42 --model_args pretrained="outputs/checkpoint-28000/"

python evals/lm_harness_eval.py --model mamba_ssm --tasks winogrande --device cuda --batch_size 256 --seed 42 --model_args pretrained="outputs/checkpoint-29000/"

python evals/lm_harness_eval.py --model mamba_ssm --tasks winogrande --device cuda --batch_size 256 --seed 42 --model_args pretrained="outputs/checkpoint-30000/"

# python evals/lm_harness_eval.py --model mamba_ssm --tasks winogrande --device cuda --batch_size 256 --seed 42 --model_args pretrained="outputs/checkpoint-31000/"

# python evals/lm_harness_eval.py --model mamba_ssm --tasks winogrande --device cuda --batch_size 256 --seed 42 --model_args pretrained="outputs/checkpoint-32000/"

# python evals/lm_harness_eval.py --model mamba_ssm --tasks winogrande --device cuda --batch_size 256 --seed 42 --model_args pretrained="outputs/checkpoint-33000/"

# python evals/lm_harness_eval.py --model mamba_ssm --tasks winogrande --device cuda --batch_size 256 --seed 42 --model_args pretrained="outputs/checkpoint-34000/"

# python evals/lm_harness_eval.py --model mamba_ssm --tasks winogrande --device cuda --batch_size 256 --seed 42 --model_args pretrained="outputs/checkpoint-35000/"

# python evals/lm_harness_eval.py --model mamba_ssm --tasks winogrande --device cuda --batch_size 256 --seed 42 --model_args pretrained="outputs/checkpoint-36000/"

# python evals/lm_harness_eval.py --model mamba_ssm --tasks winogrande --device cuda --batch_size 256 --seed 42 --model_args pretrained="outputs/checkpoint-37000/"

# python evals/lm_harness_eval.py --model mamba_ssm --tasks winogrande --device cuda --batch_size 256 --seed 42 --model_args pretrained="outputs/checkpoint-38000/"

# python evals/lm_harness_eval.py --model mamba_ssm --tasks winogrande --device cuda --batch_size 256 --seed 42 --model_args pretrained="outputs/checkpoint-39000/"

# python evals/lm_harness_eval.py --model mamba_ssm --tasks winogrande --device cuda --batch_size 256 --seed 42 --model_args pretrained="outputs/checkpoint-40000/"

# python evals/lm_harness_eval.py --model mamba_ssm --tasks winogrande --device cuda --batch_size 256 --seed 42 --model_args pretrained="outputs/checkpoint-41000/"

# python evals/lm_harness_eval.py --model mamba_ssm --tasks winogrande --device cuda --batch_size 256 --seed 42 --model_args pretrained="outputs/checkpoint-42000/"

# python evals/lm_harness_eval.py --model mamba_ssm --tasks winogrande --device cuda --batch_size 256 --seed 42 --model_args pretrained="outputs/checkpoint-43000/"

# python evals/lm_harness_eval.py --model mamba_ssm --tasks winogrande --device cuda --batch_size 256 --seed 42 --model_args pretrained="outputs/checkpoint-44000/"

# python evals/lm_harness_eval.py --model mamba_ssm --tasks winogrande --device cuda --batch_size 256 --seed 42 --model_args pretrained="outputs/checkpoint-45000/"

# python evals/lm_harness_eval.py --model mamba_ssm --tasks winogrande --device cuda --batch_size 256 --seed 42 --model_args pretrained="outputs/checkpoint-46000/"

# python evals/lm_harness_eval.py --model mamba_ssm --tasks winogrande --device cuda --batch_size 256 --seed 42 --model_args pretrained="outputs/checkpoint-47000/"

# python evals/lm_harness_eval.py --model mamba_ssm --tasks winogrande --device cuda --batch_size 256 --seed 42 --model_args pretrained="outputs/checkpoint-48000/"

# python evals/lm_harness_eval.py --model mamba_ssm --tasks winogrande --device cuda --batch_size 256 --seed 42 --model_args pretrained="outputs/checkpoint-49000/"

# python evals/lm_harness_eval.py --model mamba_ssm --tasks winogrande --device cuda --batch_size 256 --seed 42 --model_args pretrained="outputs/checkpoint-50000/"