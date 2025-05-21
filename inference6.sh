#!/bin/bash
#SBATCH -p part1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH -e stderr_%j.txt
#SBATCH --gres=gpu:1

# Insturction-Tuning command example
export CUDA_VISIBLE_DEVICES=6
python evals/lm_harness_eval.py --model mamba_ssm --tasks hellaswag --device cuda --batch_size 256 --seed 50 --trust_remote_code --model_args pretrained="/mnt/server5_hard2/seokil/mamba_output/mamba-ProDiaL16/hellaswag/130m/seed50/1e-4/checkpoint-10000/"

python evals/lm_harness_eval.py --model mamba_ssm --tasks hellaswag --device cuda --batch_size 256 --seed 50 --trust_remote_code --model_args pretrained="/mnt/server5_hard2/seokil/mamba_output/mamba-ProDiaL16/hellaswag/130m/seed50/1e-4/checkpoint-20000/"

python evals/lm_harness_eval.py --model mamba_ssm --tasks hellaswag --device cuda --batch_size 256 --seed 50 --trust_remote_code --model_args pretrained="/mnt/server5_hard2/seokil/mamba_output/mamba-ProDiaL16/hellaswag/130m/seed50/1e-4/checkpoint-30000/"

python evals/lm_harness_eval.py --model mamba_ssm --tasks hellaswag --device cuda --batch_size 256 --seed 50 --trust_remote_code --model_args pretrained="/mnt/server5_hard2/seokil/mamba_output/mamba-ProDiaL16/hellaswag/130m/seed50/1e-4/checkpoint-40000/"

python evals/lm_harness_eval.py --model mamba_ssm --tasks hellaswag --device cuda --batch_size 256 --seed 50 --trust_remote_code --model_args pretrained="/mnt/server5_hard2/seokil/mamba_output/mamba-ProDiaL16/hellaswag/130m/seed50/1e-4/checkpoint-50000/"

python evals/lm_harness_eval.py --model mamba_ssm --tasks hellaswag --device cuda --batch_size 256 --seed 50 --trust_remote_code --model_args pretrained="/mnt/server5_hard2/seokil/mamba_output/mamba-ProDiaL16/hellaswag/130m/seed50/1e-4/checkpoint-60000/"

python evals/lm_harness_eval.py --model mamba_ssm --tasks hellaswag --device cuda --batch_size 256 --seed 50 --trust_remote_code --model_args pretrained="/mnt/server5_hard2/seokil/mamba_output/mamba-ProDiaL16/hellaswag/130m/seed50/1e-4/checkpoint-70000/"

python evals/lm_harness_eval.py --model mamba_ssm --tasks hellaswag --device cuda --batch_size 256 --seed 50 --trust_remote_code --model_args pretrained="/mnt/server5_hard2/seokil/mamba_output/mamba-ProDiaL16/hellaswag/130m/seed50/1e-4/checkpoint-80000/"

python evals/lm_harness_eval.py --model mamba_ssm --tasks hellaswag --device cuda --batch_size 256 --seed 50 --trust_remote_code --model_args pretrained="/mnt/server5_hard2/seokil/mamba_output/mamba-ProDiaL16/hellaswag/130m/seed50/1e-4/checkpoint-90000/"

python evals/lm_harness_eval.py --model mamba_ssm --tasks hellaswag --device cuda --batch_size 256 --seed 50 --trust_remote_code --model_args pretrained="/mnt/server5_hard2/seokil/mamba_output/mamba-ProDiaL16/hellaswag/130m/seed50/1e-4/checkpoint-100000/"

python evals/lm_harness_eval.py --model mamba_ssm --tasks hellaswag --device cuda --batch_size 256 --seed 50 --trust_remote_code --model_args pretrained="/mnt/server5_hard2/seokil/mamba_output/mamba-ProDiaL16/hellaswag/130m/seed50/1e-4/checkpoint-110000/"

python evals/lm_harness_eval.py --model mamba_ssm --tasks hellaswag --device cuda --batch_size 256 --seed 50 --trust_remote_code --model_args pretrained="/mnt/server5_hard2/seokil/mamba_output/mamba-ProDiaL16/hellaswag/130m/seed50/1e-4/checkpoint-120000/"

python evals/lm_harness_eval.py --model mamba_ssm --tasks hellaswag --device cuda --batch_size 256 --seed 50 --trust_remote_code --model_args pretrained="/mnt/server5_hard2/seokil/mamba_output/mamba-ProDiaL16/hellaswag/130m/seed50/1e-4/checkpoint-130000/"

python evals/lm_harness_eval.py --model mamba_ssm --tasks hellaswag --device cuda --batch_size 256 --seed 50 --trust_remote_code --model_args pretrained="/mnt/server5_hard2/seokil/mamba_output/mamba-ProDiaL16/hellaswag/130m/seed50/1e-4/checkpoint-140000/"

python evals/lm_harness_eval.py --model mamba_ssm --tasks hellaswag --device cuda --batch_size 256 --seed 50 --trust_remote_code --model_args pretrained="/mnt/server5_hard2/seokil/mamba_output/mamba-ProDiaL16/hellaswag/130m/seed50/1e-4/checkpoint-150000/"

python evals/lm_harness_eval.py --model mamba_ssm --tasks hellaswag --device cuda --batch_size 256 --seed 50 --trust_remote_code --model_args pretrained="/mnt/server5_hard2/seokil/mamba_output/mamba-ProDiaL16/hellaswag/130m/seed50/1e-4/checkpoint-160000/"

python evals/lm_harness_eval.py --model mamba_ssm --tasks hellaswag --device cuda --batch_size 256 --seed 50 --trust_remote_code --model_args pretrained="/mnt/server5_hard2/seokil/mamba_output/mamba-ProDiaL16/hellaswag/130m/seed50/1e-4/checkpoint-170000/"

python evals/lm_harness_eval.py --model mamba_ssm --tasks hellaswag --device cuda --batch_size 256 --seed 50 --trust_remote_code --model_args pretrained="/mnt/server5_hard2/seokil/mamba_output/mamba-ProDiaL16/hellaswag/130m/seed50/1e-4/checkpoint-180000/"

python evals/lm_harness_eval.py --model mamba_ssm --tasks hellaswag --device cuda --batch_size 256 --seed 50 --trust_remote_code --model_args pretrained="/mnt/server5_hard2/seokil/mamba_output/mamba-ProDiaL16/hellaswag/130m/seed50/1e-4/checkpoint-190000/"

python evals/lm_harness_eval.py --model mamba_ssm --tasks hellaswag --device cuda --batch_size 256 --seed 50 --trust_remote_code --model_args pretrained="/mnt/server5_hard2/seokil/mamba_output/mamba-ProDiaL16/hellaswag/130m/seed50/1e-4/checkpoint-200000/"

# python evals/lm_harness_eval.py --model mamba_ssm --tasks hellaswag --device cuda --batch_size 256 --seed 50 --trust_remote_code --model_args pretrained="/mnt/server5_hard2/seokil/mamba_output/mamba-ProDiaL16/hellaswag/130m/seed50/1e-4/checkpoint-210000/"

# python evals/lm_harness_eval.py --model mamba_ssm --tasks hellaswag --device cuda --batch_size 256 --seed 50 --trust_remote_code --model_args pretrained="/mnt/server5_hard2/seokil/mamba_output/mamba-ProDiaL16/hellaswag/130m/seed50/1e-4/checkpoint-220000/"

# python evals/lm_harness_eval.py --model mamba_ssm --tasks hellaswag --device cuda --batch_size 256 --seed 50 --trust_remote_code --model_args pretrained="/mnt/server5_hard2/seokil/mamba_output/mamba-ProDiaL16/hellaswag/130m/seed50/1e-4/checkpoint-230000/"

# python evals/lm_harness_eval.py --model mamba_ssm --tasks hellaswag --device cuda --batch_size 256 --seed 50 --trust_remote_code --model_args pretrained="/mnt/server5_hard2/seokil/mamba_output/mamba-ProDiaL16/hellaswag/130m/seed50/1e-4/checkpoint-240000/"

# python evals/lm_harness_eval.py --model mamba_ssm --tasks hellaswag --device cuda --batch_size 256 --seed 50 --trust_remote_code --model_args pretrained="/mnt/server5_hard2/seokil/mamba_output/mamba-ProDiaL16/hellaswag/130m/seed50/1e-4/checkpoint-250000/"

# python evals/lm_harness_eval.py --model mamba_ssm --tasks hellaswag --device cuda --batch_size 256 --seed 50 --trust_remote_code --model_args pretrained="/mnt/server5_hard2/seokil/mamba_output/mamba-ProDiaL16/hellaswag/130m/seed50/1e-4/checkpoint-260000/"

# python evals/lm_harness_eval.py --model mamba_ssm --tasks hellaswag --device cuda --batch_size 256 --seed 50 --trust_remote_code --model_args pretrained="/mnt/server5_hard2/seokil/mamba_output/mamba-ProDiaL16/hellaswag/130m/seed50/1e-4/checkpoint-270000/"

# python evals/lm_harness_eval.py --model mamba_ssm --tasks hellaswag --device cuda --batch_size 256 --seed 50 --trust_remote_code --model_args pretrained="/mnt/server5_hard2/seokil/mamba_output/mamba-ProDiaL16/hellaswag/130m/seed50/1e-4/checkpoint-280000/"

# python evals/lm_harness_eval.py --model mamba_ssm --tasks hellaswag --device cuda --batch_size 256 --seed 50 --trust_remote_code --model_args pretrained="/mnt/server5_hard2/seokil/mamba_output/mamba-ProDiaL16/hellaswag/130m/seed50/1e-4/checkpoint-290000/"

# python evals/lm_harness_eval.py --model mamba_ssm --tasks hellaswag --device cuda --batch_size 256 --seed 50 --trust_remote_code --model_args pretrained="/mnt/server5_hard2/seokil/mamba_output/mamba-ProDiaL16/hellaswag/130m/seed50/1e-4/checkpoint-300000/"
