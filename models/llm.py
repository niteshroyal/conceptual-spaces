import os
import shutil
import logging

import torch
from peft import PeftModel
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from conf import configuration
from models.accelerator import accelerator


def initialization():
    log_file = os.path.join(configuration.logging_folder, os.path.splitext(os.path.basename(__file__))[0] + '.log')
    logging.basicConfig(format='%(levelname)s %(asctime)s:  %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p', filename=log_file, filemode='w', level=logging.INFO)


class LLM:

    def __init__(self):
        self.config = None
        self.bnb_config = None
        self.model = None
        self.tokenizer = None
        self.trainer = None
        self.base_model_id = configuration.base_model_id
        self.device = None
        self.ft_model = None
        self.token_sizes = []
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'


    def get_old_generator_llm_location(self):
        generator_llm = f"llm-checkpoint-{configuration.save_steps}"
        generator_llm = os.path.join(os.path.join(configuration.learned_models, self.base_model_id), generator_llm)
        return generator_llm

    def get_generator_llm(self):
        base_file_name = os.path.splitext(os.path.basename(configuration.training_dataset_ranking))[0]
        generator_llm = f"llm-checkpoint{configuration.save_steps}-{base_file_name}"
        generator_llm = os.path.join(os.path.join(configuration.learned_models, self.base_model_id), generator_llm)
        return generator_llm
    
    def rename_finetuned_model_path(self):
        old_path = self.get_old_generator_llm_location()
        new_path = self.get_generator_llm()
        if not os.path.exists(old_path):
            raise FileNotFoundError(f"The folder '{old_path}' does not exist.")
        if os.path.exists(new_path):
            shutil.rmtree(new_path)
        os.rename(old_path, new_path)
        logging.info(f"Finetuning complete. The finetuned model can be found here: {new_path}")

    def init_conf(self):
        self.config = LoraConfig(
            r=configuration.lora_r,
            lora_alpha=configuration.lora_alpha,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
                "lm_head",
            ],
            bias="none",
            lora_dropout=0.05,
            # task_type="SEQ_CLS",
            task_type="CAUSAL_LM",
        )
        if configuration.load_in_kbit == 4:
            self.bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )
        elif configuration.load_in_kbit == 8:
            self.bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=0.0
            )
        else:
            raise Exception('The value for load_in_kbit in configuration file is incorrect')
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_id,
            padding_side="left",
            add_eos_token=True,
            add_bos_token=True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def init_model(self):
        num_gpus = torch.cuda.device_count()
        if self.device == 'cuda':
            logging.info(f"There are {num_gpus} GPUs:")
            for gpu_number in range(num_gpus):
                logging.info(f"  GPU {gpu_number}: {torch.cuda.get_device_name(gpu_number)}")
        else:
            logging.info("No GPU detected\n")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model_id,
            quantization_config=self.bnb_config,
            # device_map="auto",
            use_cache=False,
            output_hidden_states=True
        )
        self.model = prepare_model_for_kbit_training(self.model)
        logging.info(self.model)
        self.model = get_peft_model(self.model, self.config)

        # # ✅ Skip accelerator.prepare_model for quantized models
        # if not getattr(self.model, "is_loaded_in_4bit", False) and not getattr(self.model, "is_loaded_in_8bit", False):
        #     self.model = accelerator.prepare_model(self.model)

        self.model = accelerator.prepare_model(self.model)
        self.model.config.pad_token_id = self.model.config.eos_token_id
        self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.model.to(self.device)

    def init_ft_model(self, location):
        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_id,
            quantization_config=self.bnb_config,
            # device_map="auto", 
            output_hidden_states=True
        )
        self.ft_model = PeftModel.from_pretrained(base_model, location)
        self.ft_model.config.pad_token_id = self.ft_model.config.eos_token_id
        self.ft_model.to(self.device)

    def print_prompts_statistics(self):
        logging.info(f"Printing the statistics of prompts' sizes queried to model = {self.base_model_id}")
        if not self.token_sizes:
            logging.info("Queried prompts' list is empty.")
            return
        min_size = min(self.token_sizes)
        max_size = max(self.token_sizes)
        mean_size = sum(self.token_sizes) / len(self.token_sizes)
        sorted_sizes = sorted(self.token_sizes)
        n = len(sorted_sizes)
        mid = n // 2
        if n % 2 == 0:
            median_size = (sorted_sizes[mid - 1] + sorted_sizes[mid]) / 2
        else:
            median_size = sorted_sizes[mid]
        logging.info(f"Minimum: {min_size}")
        logging.info(f"Maximum: {max_size}")
        logging.info(f"Median: {median_size}")
        logging.info(f"Mean: {mean_size}")


if __name__ == '__main__':
    initialization()
    obj = LLM()
    obj.init_conf()
    obj.init_model()
