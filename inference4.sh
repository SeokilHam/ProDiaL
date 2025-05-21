#!/bin/bash
#SBATCH -p part1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH -e stderr_%j.txt
#SBATCH --gres=gpu:1

# Insturction-Tuning command example
export CUDA_VISIBLE_DEVICES=6
python evals/lm_harness_eval.py --model mamba_ssm --tasks winogrande --device cuda --batch_size 256 --seed 42 --model_args pretrained="/mnt/server5_hard2/seokil/mamba_output/mamba2-FT-out/winogrande/370m/1e-6/checkpoint-1000/"

python evals/lm_harness_eval.py --model mamba_ssm --tasks winogrande --device cuda --batch_size 256 --seed 42 --model_args pretrained="/mnt/server5_hard2/seokil/mamba_output/mamba2-FT-out/winogrande/370m/1e-6/checkpoint-2000/"

python evals/lm_harness_eval.py --model mamba_ssm --tasks winogrande --device cuda --batch_size 256 --seed 42 --model_args pretrained="/mnt/server5_hard2/seokil/mamba_output/mamba2-FT-out/winogrande/370m/1e-6/checkpoint-3000/"

python evals/lm_harness_eval.py --model mamba_ssm --tasks winogrande --device cuda --batch_size 256 --seed 42 --model_args pretrained="/mnt/server5_hard2/seokil/mamba_output/mamba2-FT-out/winogrande/370m/1e-6/checkpoint-4000/"

python evals/lm_harness_eval.py --model mamba_ssm --tasks winogrande --device cuda --batch_size 256 --seed 42 --model_args pretrained="/mnt/server5_hard2/seokil/mamba_output/mamba2-FT-out/winogrande/370m/1e-6/checkpoint-5000/"

python evals/lm_harness_eval.py --model mamba_ssm --tasks winogrande --device cuda --batch_size 256 --seed 42 --model_args pretrained="/mnt/server5_hard2/seokil/mamba_output/mamba2-FT-out/winogrande/370m/1e-6/checkpoint-6000/"

python evals/lm_harness_eval.py --model mamba_ssm --tasks winogrande --device cuda --batch_size 256 --seed 42 --model_args pretrained="/mnt/server5_hard2/seokil/mamba_output/mamba2-FT-out/winogrande/370m/1e-6/checkpoint-7000/"

python evals/lm_harness_eval.py --model mamba_ssm --tasks winogrande --device cuda --batch_size 256 --seed 42 --model_args pretrained="/mnt/server5_hard2/seokil/mamba_output/mamba2-FT-out/winogrande/370m/1e-6/checkpoint-8000/"

python evals/lm_harness_eval.py --model mamba_ssm --tasks winogrande --device cuda --batch_size 256 --seed 42 --model_args pretrained="/mnt/server5_hard2/seokil/mamba_output/mamba2-FT-out/winogrande/370m/1e-6/checkpoint-9000/"

python evals/lm_harness_eval.py --model mamba_ssm --tasks winogrande --device cuda --batch_size 256 --seed 42 --model_args pretrained="/mnt/server5_hard2/seokil/mamba_output/mamba2-FT-out/winogrande/370m/1e-6/checkpoint-10000/"

python evals/lm_harness_eval.py --model mamba_ssm --tasks winogrande --device cuda --batch_size 256 --seed 42 --model_args pretrained="/mnt/server5_hard2/seokil/mamba_output/mamba2-FT-out/winogrande/370m/1e-6/checkpoint-11000/"

python evals/lm_harness_eval.py --model mamba_ssm --tasks winogrande --device cuda --batch_size 256 --seed 42 --model_args pretrained="/mnt/server5_hard2/seokil/mamba_output/mamba2-FT-out/winogrande/370m/1e-6/checkpoint-12000/"

python evals/lm_harness_eval.py --model mamba_ssm --tasks winogrande --device cuda --batch_size 256 --seed 42 --model_args pretrained="/mnt/server5_hard2/seokil/mamba_output/mamba2-FT-out/winogrande/370m/1e-6/checkpoint-13000/"

python evals/lm_harness_eval.py --model mamba_ssm --tasks winogrande --device cuda --batch_size 256 --seed 42 --model_args pretrained="/mnt/server5_hard2/seokil/mamba_output/mamba2-FT-out/winogrande/370m/1e-6/checkpoint-14000/"

python evals/lm_harness_eval.py --model mamba_ssm --tasks winogrande --device cuda --batch_size 256 --seed 42 --model_args pretrained="/mnt/server5_hard2/seokil/mamba_output/mamba2-FT-out/winogrande/370m/1e-6/checkpoint-15000/"

python evals/lm_harness_eval.py --model mamba_ssm --tasks winogrande --device cuda --batch_size 256 --seed 42 --model_args pretrained="/mnt/server5_hard2/seokil/mamba_output/mamba2-FT-out/winogrande/370m/1e-6/checkpoint-16000/"

python evals/lm_harness_eval.py --model mamba_ssm --tasks winogrande --device cuda --batch_size 256 --seed 42 --model_args pretrained="/mnt/server5_hard2/seokil/mamba_output/mamba2-FT-out/winogrande/370m/1e-6/checkpoint-17000/"

python evals/lm_harness_eval.py --model mamba_ssm --tasks winogrande --device cuda --batch_size 256 --seed 42 --model_args pretrained="/mnt/server5_hard2/seokil/mamba_output/mamba2-FT-out/winogrande/370m/1e-6/checkpoint-18000/"

python evals/lm_harness_eval.py --model mamba_ssm --tasks winogrande --device cuda --batch_size 256 --seed 42 --model_args pretrained="/mnt/server5_hard2/seokil/mamba_output/mamba2-FT-out/winogrande/370m/1e-6/checkpoint-19000/"

python evals/lm_harness_eval.py --model mamba_ssm --tasks winogrande --device cuda --batch_size 256 --seed 42 --model_args pretrained="/mnt/server5_hard2/seokil/mamba_output/mamba2-FT-out/winogrande/370m/1e-6/checkpoint-20000/"

python evals/lm_harness_eval.py --model mamba_ssm --tasks winogrande --device cuda --batch_size 256 --seed 42 --model_args pretrained="/mnt/server5_hard2/seokil/mamba_output/mamba2-FT-out/winogrande/370m/1e-6/checkpoint-21000/"

python evals/lm_harness_eval.py --model mamba_ssm --tasks winogrande --device cuda --batch_size 256 --seed 42 --model_args pretrained="/mnt/server5_hard2/seokil/mamba_output/mamba2-FT-out/winogrande/370m/1e-6/checkpoint-22000/"

python evals/lm_harness_eval.py --model mamba_ssm --tasks winogrande --device cuda --batch_size 256 --seed 42 --model_args pretrained="/mnt/server5_hard2/seokil/mamba_output/mamba2-FT-out/winogrande/370m/1e-6/checkpoint-23000/"

python evals/lm_harness_eval.py --model mamba_ssm --tasks winogrande --device cuda --batch_size 256 --seed 42 --model_args pretrained="/mnt/server5_hard2/seokil/mamba_output/mamba2-FT-out/winogrande/370m/1e-6/checkpoint-24000/"

python evals/lm_harness_eval.py --model mamba_ssm --tasks winogrande --device cuda --batch_size 256 --seed 42 --model_args pretrained="/mnt/server5_hard2/seokil/mamba_output/mamba2-FT-out/winogrande/370m/1e-6/checkpoint-25000/"

python evals/lm_harness_eval.py --model mamba_ssm --tasks winogrande --device cuda --batch_size 256 --seed 42 --model_args pretrained="/mnt/server5_hard2/seokil/mamba_output/mamba2-FT-out/winogrande/370m/1e-6/checkpoint-26000/"

python evals/lm_harness_eval.py --model mamba_ssm --tasks winogrande --device cuda --batch_size 256 --seed 42 --model_args pretrained="/mnt/server5_hard2/seokil/mamba_output/mamba2-FT-out/winogrande/370m/1e-6/checkpoint-27000/"

python evals/lm_harness_eval.py --model mamba_ssm --tasks winogrande --device cuda --batch_size 256 --seed 42 --model_args pretrained="/mnt/server5_hard2/seokil/mamba_output/mamba2-FT-out/winogrande/370m/1e-6/checkpoint-28000/"

python evals/lm_harness_eval.py --model mamba_ssm --tasks winogrande --device cuda --batch_size 256 --seed 42 --model_args pretrained="/mnt/server5_hard2/seokil/mamba_output/mamba2-FT-out/winogrande/370m/1e-6/checkpoint-29000/"

python evals/lm_harness_eval.py --model mamba_ssm --tasks winogrande --device cuda --batch_size 256 --seed 42 --model_args pretrained="/mnt/server5_hard2/seokil/mamba_output/mamba2-FT-out/winogrande/370m/1e-6/checkpoint-30000/"

# python evals/lm_harness_eval.py --model mamba_ssm --tasks winogrande --device cuda --batch_size 256 --seed 42 --model_args pretrained="/mnt/server5_hard2/seokil/mamba_output/mamba2-FT-out/winogrande/370m/1e-6/checkpoint-31000/"

# python evals/lm_harness_eval.py --model mamba_ssm --tasks winogrande --device cuda --batch_size 256 --seed 42 --model_args pretrained="/mnt/server5_hard2/seokil/mamba_output/mamba2-FT-out/winogrande/370m/1e-6/checkpoint-32000/"

# python evals/lm_harness_eval.py --model mamba_ssm --tasks winogrande --device cuda --batch_size 256 --seed 42 --model_args pretrained="/mnt/server5_hard2/seokil/mamba_output/mamba2-FT-out/winogrande/370m/1e-6/checkpoint-33000/"

# python evals/lm_harness_eval.py --model mamba_ssm --tasks winogrande --device cuda --batch_size 256 --seed 42 --model_args pretrained="/mnt/server5_hard2/seokil/mamba_output/mamba2-FT-out/winogrande/370m/1e-6/checkpoint-34000/"

# python evals/lm_harness_eval.py --model mamba_ssm --tasks winogrande --device cuda --batch_size 256 --seed 42 --model_args pretrained="/mnt/server5_hard2/seokil/mamba_output/mamba2-FT-out/winogrande/370m/1e-6/checkpoint-35000/"

# python evals/lm_harness_eval.py --model mamba_ssm --tasks winogrande --device cuda --batch_size 256 --seed 42 --model_args pretrained="/mnt/server5_hard2/seokil/mamba_output/mamba2-FT-out/winogrande/370m/1e-6/checkpoint-36000/"

# python evals/lm_harness_eval.py --model mamba_ssm --tasks winogrande --device cuda --batch_size 256 --seed 42 --model_args pretrained="/mnt/server5_hard2/seokil/mamba_output/mamba2-FT-out/winogrande/370m/1e-6/checkpoint-37000/"

# python evals/lm_harness_eval.py --model mamba_ssm --tasks winogrande --device cuda --batch_size 256 --seed 42 --model_args pretrained="/mnt/server5_hard2/seokil/mamba_output/mamba2-FT-out/winogrande/370m/1e-6/checkpoint-38000/"

# python evals/lm_harness_eval.py --model mamba_ssm --tasks winogrande --device cuda --batch_size 256 --seed 42 --model_args pretrained="/mnt/server5_hard2/seokil/mamba_output/mamba2-FT-out/winogrande/370m/1e-6/checkpoint-39000/"

# python evals/lm_harness_eval.py --model mamba_ssm --tasks winogrande --device cuda --batch_size 256 --seed 42 --model_args pretrained="/mnt/server5_hard2/seokil/mamba_output/mamba2-FT-out/winogrande/370m/1e-6/checkpoint-40000/"

# python evals/lm_harness_eval.py --model mamba_ssm --tasks winogrande --device cuda --batch_size 256 --seed 42 --model_args pretrained="/mnt/server5_hard2/seokil/mamba_output/mamba2-FT-out/winogrande/370m/1e-6/checkpoint-41000/"

# python evals/lm_harness_eval.py --model mamba_ssm --tasks winogrande --device cuda --batch_size 256 --seed 42 --model_args pretrained="/mnt/server5_hard2/seokil/mamba_output/mamba2-FT-out/winogrande/370m/1e-6/checkpoint-42000/"

# python evals/lm_harness_eval.py --model mamba_ssm --tasks winogrande --device cuda --batch_size 256 --seed 42 --model_args pretrained="/mnt/server5_hard2/seokil/mamba_output/mamba2-FT-out/winogrande/370m/1e-6/checkpoint-43000/"

# python evals/lm_harness_eval.py --model mamba_ssm --tasks winogrande --device cuda --batch_size 256 --seed 42 --model_args pretrained="/mnt/server5_hard2/seokil/mamba_output/mamba2-FT-out/winogrande/370m/1e-6/checkpoint-44000/"

# python evals/lm_harness_eval.py --model mamba_ssm --tasks winogrande --device cuda --batch_size 256 --seed 42 --model_args pretrained="/mnt/server5_hard2/seokil/mamba_output/mamba2-FT-out/winogrande/370m/1e-6/checkpoint-45000/"

# python evals/lm_harness_eval.py --model mamba_ssm --tasks winogrande --device cuda --batch_size 256 --seed 42 --model_args pretrained="/mnt/server5_hard2/seokil/mamba_output/mamba2-FT-out/winogrande/370m/1e-6/checkpoint-46000/"

# python evals/lm_harness_eval.py --model mamba_ssm --tasks winogrande --device cuda --batch_size 256 --seed 42 --model_args pretrained="/mnt/server5_hard2/seokil/mamba_output/mamba2-FT-out/winogrande/370m/1e-6/checkpoint-47000/"

# python evals/lm_harness_eval.py --model mamba_ssm --tasks winogrande --device cuda --batch_size 256 --seed 42 --model_args pretrained="/mnt/server5_hard2/seokil/mamba_output/mamba2-FT-out/winogrande/370m/1e-6/checkpoint-48000/"

# python evals/lm_harness_eval.py --model mamba_ssm --tasks winogrande --device cuda --batch_size 256 --seed 42 --model_args pretrained="/mnt/server5_hard2/seokil/mamba_output/mamba2-FT-out/winogrande/370m/1e-6/checkpoint-49000/"

# python evals/lm_harness_eval.py --model mamba_ssm --tasks winogrande --device cuda --batch_size 256 --seed 42 --model_args pretrained="/mnt/server5_hard2/seokil/mamba_output/mamba2-FT-out/winogrande/370m/1e-6/checkpoint-50000/"