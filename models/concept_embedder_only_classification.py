import logging
import os
import sys

import bitsandbytes as bnb
import torch.nn.functional as F

from conf import configuration
from models.concept_embedder import ConceptEmbedderViaRankingAndClassification, process_a_batch_classification, compute_classification_loss


def initialization():
    log_file = os.path.join(configuration.logging_folder, os.path.splitext(os.path.basename(__file__))[0] + '.log')
    logging.basicConfig(format='%(levelname)s %(asctime)s:  %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p', filename=log_file, filemode='w', level=logging.INFO)

# ==============================================================================================================================
# This model uses the classification strategy to finetune the concept embedding model
# ==============================================================================================================================
class ConceptEmbedderViaClassification(ConceptEmbedderViaRankingAndClassification):

    def __init__(self):
        super().__init__()

    def get_old_generator_llm_location(self):
        generator_llm = f"llm-classifier-embedding-checkpoint-{configuration.save_steps}"
        generator_llm = os.path.join(os.path.join(configuration.learned_models, self.base_model_id), generator_llm)
        return generator_llm


    def get_generator_llm(self):
        base_file_name = os.path.splitext(os.path.basename(configuration.training_dataset_ranking))[0]
        generator_llm = f"llm-classifier-embedding-checkpoint{configuration.save_steps}-{base_file_name}"
        generator_llm = os.path.join(os.path.join(configuration.learned_models, self.base_model_id), generator_llm)
        return generator_llm

    def finetuning(self):
        num_of_prompts_in_a_datapoint = 12
        num_of_entities_in_a_datapoint = 7
        print(f"There are {self.training_data_loader_classification.len()} datapoints in the classification training dataset")
        logging.info(f"There are {self.training_data_loader_classification.len()} datapoints in the classification training dataset")
        print(f"There are {self.training_data_loader_ranking.len()} datapoints in the ranking training dataset")
        logging.info(f"There are {self.training_data_loader_ranking.len()} datapoints in the ranking training dataset")
        optimizer = bnb.optim.Adam8bit(self.model.parameters(), lr=2.5e-5)
        self.model.train()
        max_steps_counter = 0
        number_of_classification_training_datapoints_till_now = 0
        while max_steps_counter < configuration.max_steps:
            batch = self.training_data_loader_classification.get_a_minibatch()
            batch = process_a_batch_classification(batch)
            list_of_input_ids = batch['list_of_input_ids'].to(self.device)
            list_of_attention_masks = batch['list_of_attention_masks'].to(self.device)
            num_of_prompts_in_a_batch = list_of_input_ids.size(0)
            assert num_of_prompts_in_a_batch % num_of_prompts_in_a_datapoint == 0, f"Number of prompts in a batch should be multiple of number of prompts in a datapoint"
            num_of_datapoints_in_a_batch = num_of_prompts_in_a_batch // num_of_prompts_in_a_datapoint
            number_of_classification_training_datapoints_till_now += num_of_datapoints_in_a_batch
            max_steps_counter += 1
            optimizer.zero_grad()
            outputs = self.model(input_ids=list_of_input_ids, attention_mask=list_of_attention_masks)
            last_hidden_state = outputs.hidden_states[self.which_hidden_state]
            last_token_embedding = last_hidden_state[:, -1, :]
            last_token_embedding = F.normalize(last_token_embedding, p=2, dim=1)
            embedding = last_token_embedding.view(num_of_datapoints_in_a_batch, num_of_prompts_in_a_datapoint, -1)
            loss1 = compute_classification_loss(embedding, num_of_entities_in_a_datapoint, temperature=0.25)

            loss = loss1
            loss.backward()
            optimizer.step()
            self.training_loss_queue.add(loss.item())
            print(f"Till now {number_of_classification_training_datapoints_till_now} classification datapoints used for training, "
                  f"Loss: {loss.item()}, "
                  f"Sum of last {self.training_loss_queue.size} losses = {self.training_loss_queue.average()}, "
                  f"Steps: {max_steps_counter} out of {configuration.max_steps}")
            logging.info(f"Till now {number_of_classification_training_datapoints_till_now} classification datapoints used for training, "
                         f"Loss: {loss.item()}, "
                         f"Sum of last {self.training_loss_queue.size} losses = {self.training_loss_queue.average()}, "
                         f"Steps: {max_steps_counter} out of {configuration.max_steps}")
            if max_steps_counter % configuration.eval_steps == 0:
                # self.eval()
                # self.model.train()
                pass
            if max_steps_counter == configuration.save_steps:
                self.trainer.save_model()

if __name__ == '__main__':

    if sys.argv[1] == "Meta-Llama-3-8B":
        configuration.base_model_id = "meta-llama/Meta-Llama-3-8B"
    else:
        raise Exception(f"Unknown model name {sys.argv[1]}")
    
    max_steps = int(sys.argv[2])

    configuration.batch_size = 1
    configuration.max_steps = max_steps
    configuration.save_steps = max_steps
    configuration.eval_steps = int(configuration.max_steps / 1)
    configuration.tokenizer_max_length = 62

    only_eval_mode = False
    initialization()
    obj = ConceptEmbedderViaClassification()
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
