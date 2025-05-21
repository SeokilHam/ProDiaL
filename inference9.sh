#!/bin/bash
#SBATCH -p part1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH -e stderr_%j.txt
#SBATCH --gres=gpu:1

# Insturction-Tuning command example
export CUDA_VISIBLE_DEVICES=5
python evals/lm_harness_eval.py --model mamba_ssm --tasks openbookqa --device cuda --batch_size 256 --seed 42 --model_args pretrained="/mnt/server5_hard2/seokil/mamba_output/mamba-out_ProDiaL1536-8/obqa/130m/1e-4/checkpoint-2000/"

python evals/lm_harness_eval.py --model mamba_ssm --tasks openbookqa --device cuda --batch_size 256 --seed 42 --model_args pretrained="/mnt/server5_hard2/seokil/mamba_output/mamba-out_ProDiaL1536-8/obqa/130m/1e-4/checkpoint-4000/"

python evals/lm_harness_eval.py --model mamba_ssm --tasks openbookqa --device cuda --batch_size 256 --seed 42 --model_args pretrained="/mnt/server5_hard2/seokil/mamba_output/mamba-out_ProDiaL1536-8/obqa/130m/1e-4/checkpoint-6000/"

python evals/lm_harness_eval.py --model mamba_ssm --tasks openbookqa --device cuda --batch_size 256 --seed 42 --model_args pretrained="/mnt/server5_hard2/seokil/mamba_output/mamba-out_ProDiaL1536-8/obqa/130m/1e-4/checkpoint-8000/"

python evals/lm_harness_eval.py --model mamba_ssm --tasks openbookqa --device cuda --batch_size 256 --seed 42 --model_args pretrained="/mnt/server5_hard2/seokil/mamba_output/mamba-out_ProDiaL1536-8/obqa/130m/1e-4/checkpoint-10000/"

python evals/lm_harness_eval.py --model mamba_ssm --tasks openbookqa --device cuda --batch_size 256 --seed 42 --model_args pretrained="/mnt/server5_hard2/seokil/mamba_output/mamba-out_ProDiaL1536-8/obqa/130m/1e-4/checkpoint-12000/"

python evals/lm_harness_eval.py --model mamba_ssm --tasks openbookqa --device cuda --batch_size 256 --seed 42 --model_args pretrained="/mnt/server5_hard2/seokil/mamba_output/mamba-out_ProDiaL1536-8/obqa/130m/1e-4/checkpoint-14000/"

python evals/lm_harness_eval.py --model mamba_ssm --tasks openbookqa --device cuda --batch_size 256 --seed 42 --model_args pretrained="/mnt/server5_hard2/seokil/mamba_output/mamba-out_ProDiaL1536-8/obqa/130m/1e-4/checkpoint-16000/"

python evals/lm_harness_eval.py --model mamba_ssm --tasks openbookqa --device cuda --batch_size 256 --seed 42 --model_args pretrained="/mnt/server5_hard2/seokil/mamba_output/mamba-out_ProDiaL1536-8/obqa/130m/1e-4/checkpoint-18000/"

python evals/lm_harness_eval.py --model mamba_ssm --tasks openbookqa --device cuda --batch_size 256 --seed 42 --model_args pretrained="/mnt/server5_hard2/seokil/mamba_output/mamba-out_ProDiaL1536-8/obqa/130m/1e-4/checkpoint-20000/"

python evals/lm_harness_eval.py --model mamba_ssm --tasks openbookqa --device cuda --batch_size 256 --seed 42 --model_args pretrained="/mnt/server5_hard2/seokil/mamba_output/mamba-out_ProDiaL1536-8/obqa/130m/1e-4/checkpoint-22000/"

python evals/lm_harness_eval.py --model mamba_ssm --tasks openbookqa --device cuda --batch_size 256 --seed 42 --model_args pretrained="/mnt/server5_hard2/seokil/mamba_output/mamba-out_ProDiaL1536-8/obqa/130m/1e-4/checkpoint-24000/"

python evals/lm_harness_eval.py --model mamba_ssm --tasks openbookqa --device cuda --batch_size 256 --seed 42 --model_args pretrained="/mnt/server5_hard2/seokil/mamba_output/mamba-out_ProDiaL1536-8/obqa/130m/1e-4/checkpoint-26000/"

python evals/lm_harness_eval.py --model mamba_ssm --tasks openbookqa --device cuda --batch_size 256 --seed 42 --model_args pretrained="/mnt/server5_hard2/seokil/mamba_output/mamba-out_ProDiaL1536-8/obqa/130m/1e-4/checkpoint-28000/"

python evals/lm_harness_eval.py --model mamba_ssm --tasks openbookqa --device cuda --batch_size 256 --seed 42 --model_args pretrained="/mnt/server5_hard2/seokil/mamba_output/mamba-out_ProDiaL1536-8/obqa/130m/1e-4/checkpoint-30000/"

python evals/lm_harness_eval.py --model mamba_ssm --tasks openbookqa --device cuda --batch_size 256 --seed 42 --model_args pretrained="/mnt/server5_hard2/seokil/mamba_output/mamba-out_ProDiaL1536-8/obqa/130m/1e-4/checkpoint-32000/"

python evals/lm_harness_eval.py --model mamba_ssm --tasks openbookqa --device cuda --batch_size 256 --seed 42 --model_args pretrained="/mnt/server5_hard2/seokil/mamba_output/mamba-out_ProDiaL1536-8/obqa/130m/1e-4/checkpoint-34000/"

python evals/lm_harness_eval.py --model mamba_ssm --tasks openbookqa --device cuda --batch_size 256 --seed 42 --model_args pretrained="/mnt/server5_hard2/seokil/mamba_output/mamba-out_ProDiaL1536-8/obqa/130m/1e-4/checkpoint-36000/"

python evals/lm_harness_eval.py --model mamba_ssm --tasks openbookqa --device cuda --batch_size 256 --seed 42 --model_args pretrained="/mnt/server5_hard2/seokil/mamba_output/mamba-out_ProDiaL1536-8/obqa/130m/1e-4/checkpoint-38000/"

python evals/lm_harness_eval.py --model mamba_ssm --tasks openbookqa --device cuda --batch_size 256 --seed 42 --model_args pretrained="/mnt/server5_hard2/seokil/mamba_output/mamba-out_ProDiaL1536-8/obqa/130m/1e-4/checkpoint-40000/"

python evals/lm_harness_eval.py --model mamba_ssm --tasks openbookqa --device cuda --batch_size 256 --seed 42 --model_args pretrained="/mnt/server5_hard2/seokil/mamba_output/mamba-out_ProDiaL1536-8/obqa/130m/1e-4/checkpoint-42000/"

python evals/lm_harness_eval.py --model mamba_ssm --tasks openbookqa --device cuda --batch_size 256 --seed 42 --model_args pretrained="/mnt/server5_hard2/seokil/mamba_output/mamba-out_ProDiaL1536-8/obqa/130m/1e-4/checkpoint-44000/"

python evals/lm_harness_eval.py --model mamba_ssm --tasks openbookqa --device cuda --batch_size 256 --seed 42 --model_args pretrained="/mnt/server5_hard2/seokil/mamba_output/mamba-out_ProDiaL1536-8/obqa/130m/1e-4/checkpoint-46000/"

python evals/lm_harness_eval.py --model mamba_ssm --tasks openbookqa --device cuda --batch_size 256 --seed 42 --model_args pretrained="/mnt/server5_hard2/seokil/mamba_output/mamba-out_ProDiaL1536-8/obqa/130m/1e-4/checkpoint-48000/"

python evals/lm_harness_eval.py --model mamba_ssm --tasks openbookqa --device cuda --batch_size 256 --seed 42 --model_args pretrained="/mnt/server5_hard2/seokil/mamba_output/mamba-out_ProDiaL1536-8/obqa/130m/1e-4/checkpoint-50000/"