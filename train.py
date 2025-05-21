import torch
import logging
import os
from pytz import timezone
from datetime import datetime
from train_utils.data_collator import Seq2SeqCollator
from train_utils.loader import InstructionDatasetLoader
from train_utils.preprocessor import InstructionDatasetPreprocessor
from datasets import disable_caching
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from transformers import AutoTokenizer, TrainingArguments
from train_utils.mamba_trainer import MambaTrainer
from peft import LoraConfig, get_peft_model, TaskType
import transformers
from datasets import load_dataset
import pdb

logging.basicConfig(format = "[%(asctime)s][%(levelname)s][Message] - %(message)s", level = logging.INFO)
logging.Formatter.converter = lambda *args: datetime.now(tz=timezone("Asia/Seoul")).timetuple()

def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print

def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)

def get_model_and_tokenizer(args):
    logging.info(f"model | {args.model_path} | tokenizer: {args.tokenizer_path}")
    model = MambaLMHeadModel.from_pretrained(args.model_path, dtype=torch.float32, device="cuda", strict=False)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    tokenizer.eos_token = "<|endoftext|>"
    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def train(args):
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    # Loading config, model and tokenizer

    # init_distributed_mode(args)

    model, tokenizer = get_model_and_tokenizer(args)

    lora_config = LoraConfig(
        r=16,  # Low-rank dimension
        lora_alpha=16,  # Scaling factor
        target_modules=["in_proj", "out_proj"],  # Apply LoRA to attention layers
        lora_dropout=args.dropout_rate,  # Dropout rate for LoRA layers
        task_type=TaskType.QUESTION_ANS,  # Define the task type
    )
    model = get_peft_model(model, lora_config)
    disable_caching()

    # Loading Dataset
    ## Train Dataset
    train_dataset_loader = InstructionDatasetLoader(
        random_seed=args.random_seed, 
        datasets=args.instruction_datasets, 
        dataset_sizes=args.dataset_sizes, 
        cache_dir=args.cache_dir
    )
    instruction_dataset = train_dataset_loader.load()
    logging.info(f"Instruction dataset:{instruction_dataset}")
    # Preprocessing and Encoding Dataset
    ## Train Dataset
    train_preprocessor = InstructionDatasetPreprocessor(tokenizer=tokenizer, sequence_max_length=args.sequence_max_length)
    encoded_instruction_dataset = train_preprocessor(datasets=instruction_dataset)
    logging.info(f"Encoded dataset:{encoded_instruction_dataset}")


    # Extracting model parameters from huggingface model
    params = []
    for name, param in model.named_parameters():
        if 'oft' in name or 'lora' in name:
            param.requires_grad = True
            params.append(param)
            logging.info(name)
        else:
            param.requires_grad = True

    total_params = sum(p.numel() for p in params)
    logging.info(f"Total number of parameters: {total_params}")
    # Data Collator
    data_collator = Seq2SeqCollator(tokenizer, sequence_max_length=args.sequence_max_length)

    # Optimizer
    optimizer = torch.optim.AdamW(params=params, lr=args.learning_rate, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.)

    # Trainer
    trainer = MambaTrainer(
        model=model,
        train_dataset=encoded_instruction_dataset,
        tokenizer=tokenizer,
        optimizers=(optimizer, lr_scheduler),
        data_collator=data_collator,
        args=TrainingArguments(
            learning_rate=args.learning_rate,
            num_train_epochs=args.num_epochs,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            output_dir=args.output_dir,
            logging_steps=args.logging_steps,
            save_steps=args.save_steps,
            max_grad_norm=1,
            dataloader_drop_last=True,
            report_to="none",  # Disable reporting to Tensorboard or other tools (optional)
        ),
    )

    trainer.train()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Llama-Instruction-Tuning")

    # Dataset names
    parser.add_argument("--instruction_datasets", type=str, default="[alpaca,cot-collection]", help="instruction datasets | possible datasets [alpaca, cot-collections, slimorca, openorca-multiplechoice, arc, mmlu, gsm8k, winogrande, piqa, siqa]")
    parser.add_argument("--dataset_sizes", type=str, default="[all]", help="instruction dataset ratios")

    # Random Seed
    parser.add_argument("--random_seed", type=int, default=42, help="fix random seed in torch, random, numpy")

    # Sequence Length and Generation Length
    parser.add_argument("--sequence_max_length", type=int, default=512, help="llm model max sequence length for training")

    # Data & Logging Path
    parser.add_argument("--logging_dir", type=str, default="/project/llama-instruction-tuning/exps/logging", help="path for evaluation prediction results")
    parser.add_argument("--output_dir", type=str, default="/mnt/disks-standard/persist/t5/llama-alpaca/exps/checkpoints", help="model checkpoint path")
    parser.add_argument("--cache_dir", type=str, default="/mnt/disks-standard/persist/huggingface", help="dataset cache path")

    # Model evaluation & save strategy
    parser.add_argument("--evaluation_strategy", type=str, default="none", help="do model evaluation during training | possible strategies [epoch, steps]")
    parser.add_argument("--eval_steps", type=int, default=1000, help="every this size training step, do model evaluation")
    parser.add_argument("--save_strategy", type=str, default="none", help="do model save during training | possible strategies [epoch, steps]")
    parser.add_argument("--save_steps", type=int, default=1000, help="every this size training step, do model save")

    # Model & Tokenizer path
    parser.add_argument("--tokenizer_path", type=str, default="/mnt/disks-standard/persist/llama/llama-2-7b-hf", help="path for evaluation prediction results")
    parser.add_argument("--model_path", type=str, default="/mnt/disks-standard/persist/llama/llama-2-7b-hf", help="path for evaluation prediction results")

    # Tokenizer padding side
    parser.add_argument("--padding_side", type=str, default="left", help="tokenizer padding side | possible sides [left, right]")

    # Epoch & Batch size
    parser.add_argument("--num_epochs", type=int, default=3, help="num_train_epochs for training")
    parser.add_argument("--batch_size", type=int, default=4, help="training batch size")
    parser.add_argument("--per_device_eval_forward_batch_size", type=int, default=4, help="evaluation batch size")
    parser.add_argument("--per_device_eval_generate_batch_size", type=int, default=4, help="evaluation batch size")

    # Optimizer
    parser.add_argument("--learning_rate", type=float, default=5e-6, help="dataset")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="dataset")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="dataset")

    # Scheduler
    parser.add_argument("--lr_scheduler_type", type=str, default="constant", help="type of learning rate scheduler")
    parser.add_argument("--warmup_ratio", type=float, default=0.05, help="warmup ratio of linear learning rate scheduler")
    parser.add_argument("--dropout_rate", type=float, default=0.1)

    # Logging
    parser.add_argument("--logging_steps", type=int, default=100, help="Number of update steps between two logs if logging_strategy is 'step")
    
    # DDP
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    args = parser.parse_args()

    args.run_name = f"MODEL_NAME:{args.model_path}-DATASETS:{args.instruction_datasets}-EP:{args.num_epochs}-LR:{args.learning_rate}-BS:{args.batch_size}-WR:{args.warmup_ratio}-WD:{args.weight_decay}"

    logging.info(f"Training arguments: {args}")
    train(args)