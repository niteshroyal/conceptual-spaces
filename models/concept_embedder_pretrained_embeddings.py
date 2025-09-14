import os
import torch
import logging

from peft import PeftModel
from peft import prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM

from models.accelerator import accelerator
from models.concept_embedder import ConceptEmbedderViaRankingAndClassification
from conf import configuration

def initialization():
    log_file = os.path.join(configuration.logging_folder, os.path.splitext(os.path.basename(__file__))[0] + '.log')
    logging.basicConfig(format='%(levelname)s %(asctime)s:  %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p', filename=log_file, filemode='w', level=logging.INFO)


class ConceptEmbedderViaRankingAndClassificationPretrained(ConceptEmbedderViaRankingAndClassification):
    def __init__(self):
        super().__init__()
        self.base_model_id = "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp"
        self.peft_weights = "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised"

    def init_ft_model(self, location):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_id,
            quantization_config=self.bnb_config,
            output_hidden_states=True,
            torch_dtype=torch.bfloat16,  # Match the dtype used in pretraining
            device_map="auto"  # Optional: automatically map across GPUs
        )
        self.ft_model = PeftModel.from_pretrained(base_model,location)
        self.ft_model.config.pad_token_id = self.ft_model.config.eos_token_id
        self.ft_model.to(self.device)

    def init_model(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        num_gpus = torch.cuda.device_count()
        if self.device == 'cuda':
            logging.info(f"There are {num_gpus} GPUs:")
            for gpu_number in range(num_gpus):
                logging.info(f"  GPU {gpu_number}: {torch.cuda.get_device_name(gpu_number)}")
        else:
            logging.info("No GPU detected\n")

        # Load base model with quantization
        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model_id,
            quantization_config=self.bnb_config,
            use_cache=False,
            output_hidden_states=True,
            torch_dtype=torch.bfloat16,
            device_map="auto"  # Optional: let HF map it across available devices
        )

        self.model = prepare_model_for_kbit_training(self.model)
        logging.info(self.model)

        # Load PEFT adapter weights (supervised version)
        self.model = PeftModel.from_pretrained(
            self.model,
            self.peft_weights,
            is_trainable=True  # Important: allow further fine-tuning
        )

        self.model = accelerator.prepare_model(self.model)
        self.model.config.pad_token_id = self.model.config.eos_token_id
        self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.model.to(self.device)

if __name__ == '__main__':
    configuration.batch_size = 1
    configuration.max_steps = 123
    configuration.save_steps = 123
    only_eval_mode = False
    initialization()
    obj = ConceptEmbedderViaRankingAndClassificationPretrained()
    if only_eval_mode:
        obj.init_ft_model(obj.get_generator_llm())
        obj.init_evaluator()
        obj.eval()
    else:
        obj.init_model()
        obj.init_trainer()
        obj.determine_tokenizer_max_length()
        obj.finetuning()
        obj.rename_finetuned_model_path()
