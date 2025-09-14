import logging
import os

import bitsandbytes as bnb
import torch

from conf import configuration
from models.concept_embedder import ConceptEmbedderViaRankingAndClassification, process_a_batch_ranking, compute_sign_only_loss


def initialization():
    log_file = os.path.join(configuration.logging_folder, os.path.splitext(os.path.basename(__file__))[0] + '.log')
    logging.basicConfig(format='%(levelname)s %(asctime)s:  %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p', filename=log_file, filemode='w', level=logging.INFO)


# ==============================================================================================================================
# This model uses the ranking strategy to finetune the concept embedding model
# ==============================================================================================================================
class ConceptEmbedderViaRanking(ConceptEmbedderViaRankingAndClassification):

    def __init__(self):
        super().__init__()

    def get_old_generator_llm_location(self):
        generator_llm = f"llm-ranker-embedding-checkpoint-{configuration.save_steps}"
        generator_llm = os.path.join(os.path.join(configuration.learned_models, self.base_model_id), generator_llm)
        return generator_llm

    def get_generator_llm(self):
        base_file_name = os.path.splitext(os.path.basename(configuration.training_dataset_ranking))[0]
        generator_llm = f"llm-ranker-embedding-checkpoint{configuration.save_steps}-{base_file_name}"
        generator_llm = os.path.join(os.path.join(configuration.learned_models, self.base_model_id), generator_llm)
        return generator_llm

    def finetuning(self):
        print(f"There are {self.training_data_loader_classification.len()} datapoints in the classification training dataset")
        logging.info(f"There are {self.training_data_loader_classification.len()} datapoints in the classification training dataset")
        print(f"There are {self.training_data_loader_ranking.len()} datapoints in the ranking training dataset")
        logging.info(f"There are {self.training_data_loader_ranking.len()} datapoints in the ranking training dataset")
        optimizer = bnb.optim.Adam8bit(self.model.parameters(), lr=2.5e-5)
        self.model.train()
        max_steps_counter = 0
        number_of_ranking_training_datapoints_till_now = 0
        while max_steps_counter < configuration.max_steps:
            batch = self.training_data_loader_ranking.get_a_minibatch()
            batch = process_a_batch_ranking(batch)
            input_ids1 = batch['input_ids1']
            attention_mask1 = batch['attention_mask1']
            input_ids2 = batch['input_ids2']
            attention_mask2 = batch['attention_mask2']
            input_id3 = batch['input_ids3']
            attention_mask3 = batch['attention_mask3']
            labels = batch['labels'].to(self.device)
            input_ids = torch.cat((input_ids1, input_ids2, input_id3), dim=0).to(self.device)
            attention_mask = torch.cat((attention_mask1, attention_mask2, attention_mask3), dim=0).to(self.device)
            n = input_ids1.size(0)
            number_of_ranking_training_datapoints_till_now += n
            max_steps_counter += 1
            optimizer.zero_grad()
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            last_hidden_state = outputs.hidden_states[self.which_hidden_state]
            last_token_embedding = last_hidden_state[:, -1, :]
            embedding1 = last_token_embedding[:n]
            embedding2 = last_token_embedding[n:n + n]
            embedding3 = last_token_embedding[n + n:]
            loss2 = compute_sign_only_loss(embedding1, embedding2, embedding3, labels)

            loss = loss2
            loss.backward()
            optimizer.step()
            self.training_loss_queue.add(loss.item())
            print(f"Till now {number_of_ranking_training_datapoints_till_now} ranking datapoints used for training, "
                  f"Loss: {loss.item()}, "
                  f"Sum of last {self.training_loss_queue.size} losses = {self.training_loss_queue.average()}, "
                  f"Steps: {max_steps_counter} out of {configuration.max_steps}")
            logging.info(f"Till now {number_of_ranking_training_datapoints_till_now} ranking datapoints used for training, "
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
    configuration.batch_size = 1
    configuration.max_steps = 123
    configuration.save_steps = 123

    only_eval_mode = False
    initialization()
    obj = ConceptEmbedderViaRanking()
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
