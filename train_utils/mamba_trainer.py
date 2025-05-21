from transformers import Trainer
import torch
import os
from torch.utils.data import DataLoader
import numpy as np
import random
import copy
import shutil
import torch.nn.functional as F
import pdb

def seed_worker(_):
    """
    Helper function to set worker seed during Dataloader initialization.
    """
    worker_seed = torch.initial_seed() % 2**32
    set_seed(worker_seed)

def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    np.random.default_rng(seed)
    random.seed(seed)

class MambaTrainer(Trainer):
    def __init__(self, *args, config_path=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.config_path = config_path

    def compute_loss(self, model, inputs, return_outputs=False):
        # from transformers import AutoTokenizer
        # tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
        input_ids = inputs.pop("input_ids")
        lm_logits = model(input_ids).logits
        labels = inputs.pop("labels")
        vocab_size = lm_logits.shape[-1]
        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
        lm_loss = loss_fct(lm_logits.view(-1, vocab_size), labels.view(-1).long())
        # print(f"loss: {lm_loss}, lr: {self._get_learning_rate()}, epoch: {self.state.global_step}")
        return lm_loss

    def save_model(self, output_dir, _internal_call):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # # ProDiaL
        updated_model = copy.deepcopy(self.model)
        for i in range(len(updated_model.backbone.layers)):
            # # # # in_ProDiaL
            R = updated_model.backbone.layers[i].mixer.in_ProDiaL_r
            S = updated_model.backbone.layers[i].mixer.in_ProDiaL_s
            in_orth_rotation = R
            in_ProDiaL_mat = torch.block_diag(*[r_i for r_i in in_orth_rotation])
            identity_matrix = torch.eye(in_ProDiaL_mat.shape[0], device=in_ProDiaL_mat.device)
            in_ProDiaL_mat = in_ProDiaL_mat * (1 - identity_matrix) + identity_matrix - F.relu(torch.diag(torch.diag(in_ProDiaL_mat)))
            in_scale = torch.diag(S)

            updated_model.backbone.layers[i].mixer.in_proj.weight.data = in_scale @ updated_model.backbone.layers[i].mixer.in_proj.weight @ in_ProDiaL_mat # 768
            del updated_model.backbone.layers[i].mixer.in_ProDiaL_r
            del updated_model.backbone.layers[i].mixer.in_ProDiaL_s

            # # out_ProDiaL
            R = updated_model.backbone.layers[i].mixer.out_ProDiaL_r
            S = updated_model.backbone.layers[i].mixer.out_ProDiaL_s
            out_orth_rotation = R
            out_ProDiaL_mat = torch.block_diag(*[r_i for r_i in out_orth_rotation])
            identity_matrix = torch.eye(out_ProDiaL_mat.shape[0], device=out_ProDiaL_mat.device)
            out_ProDiaL_mat = out_ProDiaL_mat * (1 - identity_matrix) + identity_matrix - F.relu(torch.diag(torch.diag(out_ProDiaL_mat)))
            out_scale = torch.diag(S)

            updated_model.backbone.layers[i].mixer.out_proj.weight.data = out_scale @ updated_model.backbone.layers[i].mixer.out_proj.weight @ out_ProDiaL_mat # 1536
            del updated_model.backbone.layers[i].mixer.out_ProDiaL_r
            del updated_model.backbone.layers[i].mixer.out_ProDiaL_s

        merged_model = updated_model.merge_and_unload()
        merged_model.save_pretrained(f"{output_dir}")

        # Save the configuration of the model
        config_path = self.config_path + "/config.json"
        shutil.copy(config_path, output_dir)
        print(f"Save model in {output_dir}")

    def get_train_dataloader(self):
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
        training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        train_dataset = self.train_dataset
        data_collator = self.data_collator

        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_train_sampler()
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = seed_worker
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))