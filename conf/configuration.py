import os
import importlib.util

dir_path = os.path.dirname(os.path.realpath(__file__))

configuration_file_to_consider = os.path.join(dir_path, "my_conf.py")


def load_module_from_file(filepath):
    spec = importlib.util.spec_from_file_location("conf", filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


config = load_module_from_file(configuration_file_to_consider)

username = config.username

logging_folder = config.logging_folder
data_folder = config.data_folder
ranking_dataset_path = config.ranking_dataset_path
learned_models = config.learned_models

training_dataset_ranking = config.training_dataset_ranking
validation_dataset = config.validation_dataset


base_model_id = config.base_model_id

load_in_kbit = config.load_in_kbit
lora_r = config.lora_r
lora_alpha = config.lora_alpha
tokenizer_max_length = config.tokenizer_max_length
max_steps = config.max_steps
save_steps = config.save_steps
eval_steps = config.eval_steps
batch_size = config.batch_size
run_id = config.run_id

training_dataset_classification = config.training_dataset_classification
