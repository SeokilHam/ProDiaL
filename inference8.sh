#!/bin/bash
#SBATCH -p part1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH -e stderr_%j.txt
#SBATCH --gres=gpu:1

# Insturction-Tuning command example
export CUDA_VISIBLE_DEVICES=5
python evals/lm_harness_eval.py --model mamba_ssm --tasks piqa --device cuda --batch_size 256 --seed 42 --model_args pretrained="/mnt/server5_hard2/seokil/mamba_output/mamba-ProDiaL64-256-4/piqa/130m/5e-5/checkpoint-5000/"

python evals/lm_harness_eval.py --model mamba_ssm --tasks piqa --device cuda --batch_size 256 --seed 42 --model_args pretrained="/mnt/server5_hard2/seokil/mamba_output/mamba-ProDiaL64-256-4/piqa/130m/5e-5/checkpoint-10000/"

python evals/lm_harness_eval.py --model mamba_ssm --tasks piqa --device cuda --batch_size 256 --seed 42 --model_args pretrained="/mnt/server5_hard2/seokil/mamba_output/mamba-ProDiaL64-256-4/piqa/130m/5e-5/checkpoint-15000/"

python evals/lm_harness_eval.py --model mamba_ssm --tasks piqa --device cuda --batch_size 256 --seed 42 --model_args pretrained="/mnt/server5_hard2/seokil/mamba_output/mamba-ProDiaL64-256-4/piqa/130m/5e-5/checkpoint-20000/"

python evals/lm_harness_eval.py --model mamba_ssm --tasks piqa --device cuda --batch_size 256 --seed 42 --model_args pretrained="/mnt/server5_hard2/seokil/mamba_output/mamba-ProDiaL64-256-4/piqa/130m/5e-5/checkpoint-25000/"

python evals/lm_harness_eval.py --model mamba_ssm --tasks piqa --device cuda --batch_size 256 --seed 42 --model_args pretrained="/mnt/server5_hard2/seokil/mamba_output/mamba-ProDiaL64-256-4/piqa/130m/5e-5/checkpoint-30000/"

python evals/lm_harness_eval.py --model mamba_ssm --tasks piqa --device cuda --batch_size 256 --seed 42 --model_args pretrained="/mnt/server5_hard2/seokil/mamba_output/mamba-ProDiaL64-256-4/piqa/130m/5e-5/checkpoint-35000/"

python evals/lm_harness_eval.py --model mamba_ssm --tasks piqa --device cuda --batch_size 256 --seed 42 --model_args pretrained="/mnt/server5_hard2/seokil/mamba_output/mamba-ProDiaL64-256-4/piqa/130m/5e-5/checkpoint-40000/"

python evals/lm_harness_eval.py --model mamba_ssm --tasks piqa --device cuda --batch_size 256 --seed 42 --model_args pretrained="/mnt/server5_hard2/seokil/mamba_output/mamba-ProDiaL64-256-4/piqa/130m/5e-5/checkpoint-45000/"

python evals/lm_harness_eval.py --model mamba_ssm --tasks piqa --device cuda --batch_size 256 --seed 42 --model_args pretrained="/mnt/server5_hard2/seokil/mamba_output/mamba-ProDiaL64-256-4/piqa/130m/5e-5/checkpoint-50000/"

python evals/lm_harness_eval.py --model mamba_ssm --tasks piqa --device cuda --batch_size 256 --seed 42 --model_args pretrained="/mnt/server5_hard2/seokil/mamba_output/mamba-ProDiaL64-256-4/piqa/130m/5e-5/checkpoint-55000/"

python evals/lm_harness_eval.py --model mamba_ssm --tasks piqa --device cuda --batch_size 256 --seed 42 --model_args pretrained="/mnt/server5_hard2/seokil/mamba_output/mamba-ProDiaL64-256-4/piqa/130m/5e-5/checkpoint-60000/"

python evals/lm_harness_eval.py --model mamba_ssm --tasks piqa --device cuda --batch_size 256 --seed 42 --model_args pretrained="/mnt/server5_hard2/seokil/mamba_output/mamba-ProDiaL64-256-4/piqa/130m/5e-5/checkpoint-65000/"

python evals/lm_harness_eval.py --model mamba_ssm --tasks piqa --device cuda --batch_size 256 --seed 42 --model_args pretrained="/mnt/server5_hard2/seokil/mamba_output/mamba-ProDiaL64-256-4/piqa/130m/5e-5/checkpoint-70000/"

python evals/lm_harness_eval.py --model mamba_ssm --tasks piqa --device cuda --batch_size 256 --seed 42 --model_args pretrained="/mnt/server5_hard2/seokil/mamba_output/mamba-ProDiaL64-256-4/piqa/130m/5e-5/checkpoint-75000/"

python evals/lm_harness_eval.py --model mamba_ssm --tasks piqa --device cuda --batch_size 256 --seed 42 --model_args pretrained="/mnt/server5_hard2/seokil/mamba_output/mamba-ProDiaL64-256-4/piqa/130m/5e-5/checkpoint-80000/"

python evals/lm_harness_eval.py --model mamba_ssm --tasks piqa --device cuda --batch_size 256 --seed 42 --model_args pretrained="/mnt/server5_hard2/seokil/mamba_output/mamba-ProDiaL64-256-4/piqa/130m/5e-5/checkpoint-85000/"

python evals/lm_harness_eval.py --model mamba_ssm --tasks piqa --device cuda --batch_size 256 --seed 42 --model_args pretrained="/mnt/server5_hard2/seokil/mamba_output/mamba-ProDiaL64-256-4/piqa/130m/5e-5/checkpoint-90000/"

python evals/lm_harness_eval.py --model mamba_ssm --tasks piqa --device cuda --batch_size 256 --seed 42 --model_args pretrained="/mnt/server5_hard2/seokil/mamba_output/mamba-ProDiaL64-256-4/piqa/130m/5e-5/checkpoint-95000/"

python evals/lm_harness_eval.py --model mamba_ssm --tasks piqa --device cuda --batch_size 256 --seed 42 --model_args pretrained="/mnt/server5_hard2/seokil/mamba_output/mamba-ProDiaL64-256-4/piqa/130m/5e-5/checkpoint-100000/"

