import logging
import os
import sys
from collections import deque

import bitsandbytes as bnb
import evaluate
import torch
import torch.nn.functional as F
import transformers

from conf import configuration
from models.prompts_concept_embeddings import get_eval_record, get_eval_label, get_train_record, get_record, get_label, load_datasets
from models.llm import LLM


def initialization():
    log_file = os.path.join(configuration.logging_folder, os.path.splitext(os.path.basename(__file__))[0] + '.log')
    logging.basicConfig(format='%(levelname)s %(asctime)s:  %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p', filename=log_file, filemode='w', level=logging.INFO)


metric = evaluate.load("accuracy")

train_dataset_ranking, train_dataset_classification, eval_dataset = None, None, None

def load_train_eval_datasets():
    global train_dataset_ranking, train_dataset_classification, eval_dataset
    train_dataset_classification, train_dataset_ranking, eval_dataset = load_datasets()


def process_a_batch_ranking(batch):
    input_ids1_list = []
    attention_mask1_list = []
    input_ids2_list = []
    attention_mask2_list = []
    input_ids3_list = []
    attention_mask3_list = []
    labels_list = []
    for item in batch:
        input_ids1_list.append(item['input_ids1'])
        attention_mask1_list.append(item['attention_mask1'])
        input_ids2_list.append(item['input_ids2'])
        attention_mask2_list.append(item['attention_mask2'])
        input_ids3_list.append(item['input_ids3'])
        attention_mask3_list.append(item['attention_mask3'])
        labels_list.append(item['labels'])
    input_ids1 = torch.stack(input_ids1_list, dim=0)
    attention_mask1 = torch.stack(attention_mask1_list, dim=0)
    input_ids2 = torch.stack(input_ids2_list, dim=0)
    attention_mask2 = torch.stack(attention_mask2_list, dim=0)
    input_ids3 = torch.stack(input_ids3_list, dim=0)
    attention_mask3 = torch.stack(attention_mask3_list, dim=0)
    labels = torch.stack(labels_list, dim=0)
    return {
        'input_ids1': input_ids1,
        'attention_mask1': attention_mask1,
        'input_ids2': input_ids2,
        'attention_mask2': attention_mask2,
        'input_ids3': input_ids3,
        'attention_mask3': attention_mask3,
        'labels': labels
    }

def process_a_batch_classification(batch):
    labels_list = []
    list_of_input_ids = []
    list_of_attention_masks = []

    for item in batch:
        list_of_input_ids = list_of_input_ids + item['list_of_input_ids']
        list_of_attention_masks = list_of_attention_masks + item['list_of_attention_masks']
        labels_list.append(item['labels'])

    input_ids = torch.stack(list_of_input_ids, dim=0)
    attention_mask = torch.stack(list_of_attention_masks, dim=0)
    labels = torch.stack(labels_list, dim=0)
    return {
        'list_of_input_ids': input_ids,
        'list_of_attention_masks': attention_mask,
        'labels': labels
    }

def compute_sign_only_loss(embedding1, embedding2, embedding3, labels, alpha=10.0): # Changed alpha from 20.0 to 50.0
    embedding1 = F.normalize(embedding1, p=2, dim=1)
    embedding2 = F.normalize(embedding2, p=2, dim=1)
    embedding3 = F.normalize(embedding3, p=2, dim=1)
    difference = embedding1 - embedding2
    logits = torch.sum(difference * embedding3, dim=1)
    labels_signed = labels.float() * 2 - 1
    loss = torch.sigmoid(-alpha * labels_signed * logits).mean()
    return loss

def compute_classification_loss(embedding, n, temperature=1.0):
    centroid = embedding[:, :n, :].mean(dim=1)
    centroid = F.normalize(centroid, p=2, dim=1)
    class_embeddings = embedding[:, n:, :]
    logits = torch.bmm(class_embeddings, centroid.unsqueeze(2)).squeeze(2)
    logits /= temperature
    labels = torch.zeros(embedding.size(0), dtype=torch.long).to(embedding.device)
    loss = F.cross_entropy(logits, labels)
    return loss

class FixedSizeLossQueue:
    def __init__(self, size=100):
        self.size = size
        self.queue = deque([0] * size, maxlen=size)
        self.elements_added = 0

    def add(self, item):
        self.elements_added += 1
        self.queue.append(item)

    def get_queue(self):
        return list(self.queue)

    def average(self):
        if self.elements_added == 0:
            return 0
        actual_elements = min(self.elements_added, self.size)
        return sum(self.queue) / actual_elements


class TrainDatasetLoaderClassification:
    def __init__(self, datapoints, tokenizer):
        self.tokenizer = tokenizer
        self.datapoints = datapoints
        self.batches = []
        self.batch_current_idx = -1
        self.batches_size = -1
        self.token_sizes = []

    def len(self):
        return len(self.datapoints)
    
    def get_record(self, datapoint):
        return get_train_record(datapoint)
    
    def get_label(self, datapoint):
        return -1
    
    def get_tokenized_data(self, datapoint):
        entities_prompts, properties_prompts = self.get_record(datapoint)
        label = self.get_label(datapoint)
        prompts = entities_prompts + properties_prompts
        list_of_input_ids = []
        list_of_attention_masks = []
        for prompt in prompts:
            result = self.tokenizer(
            prompt,
            truncation=True,
            max_length=configuration.tokenizer_max_length,
            padding="max_length",
            return_tensors="pt"
            )
            self.token_sizes.append(len(self.tokenizer(prompt)['input_ids']))
            list_of_input_ids.append(result['input_ids'].flatten())
            list_of_attention_masks.append(result['attention_mask'].flatten())
            
        return {
            'list_of_input_ids': list_of_input_ids,
            'list_of_attention_masks': list_of_attention_masks,
            'labels': torch.tensor(label, dtype=torch.float)
        }

    def process(self):
        processed_datapoints = []
        for datapoint in self.datapoints:
            tokenized = self.get_tokenized_data(datapoint)
            processed_datapoints.append(tokenized)
        batch_size = configuration.batch_size
        batch = []
        for datapoint in processed_datapoints:
            if len(batch) == batch_size:
                self.batches.append(batch)
                batch = [datapoint]
            else:
                batch.append(datapoint)
        if batch:
            self.batches.append(batch)
        self.batches_size = len(self.batches)

    def get_a_minibatch(self):
        self.batch_current_idx += 1
        if self.batch_current_idx == self.batches_size:
            self.batch_current_idx = 0
        else:
            pass
        batch = self.batches[self.batch_current_idx]
        return batch


class TrainDatasetLoaderRanking:
    def __init__(self, datapoints, tokenizer):
        self.tokenizer = tokenizer
        self.datapoints = datapoints
        self.batches = []
        self.batch_current_idx = -1
        self.batches_size = -1
        self.token_sizes = []

    def len(self):
        return len(self.datapoints)

    def get_tokenized_data(self, datapoint):
        [prompt1, prompt2, prop] = get_record(datapoint)
        label = get_label(datapoint)
        result1 = self.tokenizer(
            prompt1,
            truncation=True,
            max_length=configuration.tokenizer_max_length,
            padding="max_length",
            return_tensors="pt"
        )
        result2 = self.tokenizer(
            prompt2,
            truncation=True,
            max_length=configuration.tokenizer_max_length,
            padding="max_length",
            return_tensors="pt"
        )
        result3 = self.tokenizer(
            prop,
            truncation=True,
            max_length=configuration.tokenizer_max_length,
            padding="max_length",
            return_tensors="pt"
        )
        self.token_sizes.append(len(self.tokenizer(prompt1)['input_ids']))
        self.token_sizes.append(len(self.tokenizer(prompt2)['input_ids']))
        self.token_sizes.append(len(self.tokenizer(prop)['input_ids']))
        return {
            'input_ids1': result1['input_ids'].flatten(),
            'attention_mask1': result1['attention_mask'].flatten(),
            'input_ids2': result2['input_ids'].flatten(),
            'attention_mask2': result2['attention_mask'].flatten(),
            'input_ids3': result3['input_ids'].flatten(),
            'attention_mask3': result3['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.float)
        }

    def process(self):
        processed_datapoints = []
        for datapoint in self.datapoints:
            tokenized = self.get_tokenized_data(datapoint)
            processed_datapoints.append(tokenized)
        batch_size = int(configuration.batch_size * 12)
        batch_size = int(batch_size / 3)
        batch = []
        for datapoint in processed_datapoints:
            if len(batch) == batch_size:
                self.batches.append(batch)
                batch = [datapoint]
            else:
                batch.append(datapoint)
        if batch:
            self.batches.append(batch)
        self.batches_size = len(self.batches)

    def get_a_minibatch(self):
        self.batch_current_idx += 1
        if self.batch_current_idx == self.batches_size:
            self.batch_current_idx = 0
        else:
            pass
        batch = self.batches[self.batch_current_idx]
        return batch



class EvalDatasetLoader(TrainDatasetLoaderClassification):
    def __init__(self, datapoints, tokenizer):
        super().__init__(datapoints, tokenizer)

    def get_record(self, datapoint):
        return get_eval_record(datapoint)
    
    def get_label(self, datapoint):
        return get_eval_label(datapoint)

    def process(self):
        processed_datapoints = []
        for datapoint in self.datapoints:
            # if not datapoint["relation"] == "bitter in taste":
            #     continue
            tokenized = self.get_tokenized_data(datapoint)
            processed_datapoints.append(tokenized)
        batch_size = int(configuration.batch_size * 12)
        batch_size = int(batch_size / 3)
        batch = []
        for datapoint in processed_datapoints:
            if len(batch) == batch_size:
                self.batches.append(batch)
                batch = [datapoint]
            else:
                batch.append(datapoint)
        if batch:
            self.batches.append(batch)
        self.batches_size = len(self.batches)


# ==============================================================================================================================
# This is a hybrid model that uses both the classification strategy and ranking strategy to finetune the concept embedding model
# ==============================================================================================================================
class ConceptEmbedderViaRankingAndClassification(LLM):

    def __init__(self):
        super().__init__()
        self.init_conf()
        self.training_data_loader_classification = None
        self.training_data_loader_ranking = None
        self.eval_data_loader = None
        self.training_loss_queue = FixedSizeLossQueue(100)
        self.which_hidden_state = -1


    def get_old_generator_llm_location(self):
        generator_llm = f"llm-ranker-classifier-embedding-checkpoint-{configuration.save_steps}"
        generator_llm = os.path.join(os.path.join(configuration.learned_models, self.base_model_id), generator_llm)
        return generator_llm


    def get_generator_llm(self):
        base_file_name = os.path.splitext(os.path.basename(configuration.training_dataset_ranking))[0]
        generator_llm = f"llm-ranker-classifier-embedding-checkpoint{configuration.save_steps}-{base_file_name}"
        generator_llm = os.path.join(os.path.join(configuration.learned_models, self.base_model_id), generator_llm)
        return generator_llm

    def get_embeddings(self, list_of_texts):
        model_input = self.tokenizer(
            list_of_texts,
            truncation=True,
            max_length=configuration.tokenizer_max_length,
            padding="max_length",
            return_tensors="pt"
        ).to("cuda")
        input_ids_list = self.tokenizer(list_of_texts)['input_ids']
        for ids in input_ids_list:
            self.token_sizes.append(len(ids))
        self.ft_model.eval()
        with torch.no_grad():
            outputs = self.ft_model(**model_input)
        last_hidden_state = outputs.hidden_states[self.which_hidden_state]
        last_token_embedding = last_hidden_state[:, -1, :]
        embeddings = F.normalize(last_token_embedding, p=2, dim=1)
        return embeddings

    def get_embeddings_batch(self, dataset):
        n = len(dataset)
        batch_size = configuration.batch_size * 12
        all_embeddings = []
        counter = 0
        for i in range(0, n, batch_size):
            batch = dataset[i:i + batch_size]
            embeddings = self.get_embeddings(batch)
            all_embeddings.append(embeddings)
            counter += len(embeddings)
            if counter % 100 == 0:
                logging.info(f'Computed concept embeddings for {counter} datapoints')
        all_embeddings = torch.cat(all_embeddings, dim=0)
        logging.info(f'Computed concept embeddings for {all_embeddings.size(0)} datapoints')
        return all_embeddings

    def eval(self):
        num_of_prompts_in_a_datapoint = 3
        counter = 0
        self.model.eval()
        references = torch.randn(0).to(self.device)
        predictions = torch.randn(0).to(self.device)
        with torch.no_grad():
            for batch in self.eval_data_loader.batches:
                batch = process_a_batch_classification(batch)
                list_of_input_ids = batch['list_of_input_ids'].to(self.device)
                list_of_attention_masks = batch['list_of_attention_masks'].to(self.device)
                labels = batch['labels'].to(self.device)
                outputs = self.model(input_ids=list_of_input_ids, attention_mask=list_of_attention_masks)
                last_hidden_state = outputs.hidden_states[self.which_hidden_state]
                last_token_embedding = last_hidden_state[:, -1, :]
                last_token_embedding = F.normalize(last_token_embedding, p=2, dim=1)

                num_of_prompts_in_a_batch = list_of_input_ids.size(0)
                assert num_of_prompts_in_a_batch % num_of_prompts_in_a_datapoint == 0, "Number of prompts in a batch should be a multiple of number of prompts in a datapoint"
                num_of_datapoints_in_a_batch = num_of_prompts_in_a_batch // num_of_prompts_in_a_datapoint

                embedding = last_token_embedding.view(num_of_datapoints_in_a_batch, num_of_prompts_in_a_datapoint, -1)

                embedding1 = embedding[:, 0, :]
                embedding2 = embedding[:, 1, :]
                embedding3 = embedding[:, 2, :]

                logits = torch.sum((embedding1 - embedding2) * embedding3, dim=1)
                prediction = (logits > 0).long()
                
                references = torch.cat((references, labels), dim=0)
                predictions = torch.cat((predictions, prediction), dim=0)
                counter += num_of_datapoints_in_a_batch
                logging.info(f'Out of {self.eval_data_loader.len()} eval datapoints {counter} is processed')
                print(f'Out of {self.eval_data_loader.len()} eval datapoints {counter} is processed')
        result = metric.compute(predictions=predictions, references=references)
        logging.info(f"Average pairwise eval_accuracy is {result['accuracy']}")
        print(f"Average pairwise eval_accuracy is {result['accuracy']}")

    def init_trainer(self):
        # file_name = '/home/niteshkumar/research/concept_embeddings/data/temp.jsonl'
        # with open(file_name, "w", encoding="utf-8") as f:
        #     for item in train_dataset_classification:
        #         f.write(json.dumps(item, ensure_ascii=False) + "\n")

        self.training_data_loader_classification = TrainDatasetLoaderClassification(train_dataset_classification, self.tokenizer)
        self.training_data_loader_classification.process()
        self.training_data_loader_ranking = TrainDatasetLoaderRanking(train_dataset_ranking, self.tokenizer)
        self.training_data_loader_ranking.process()
        self.eval_data_loader = EvalDatasetLoader(eval_dataset, self.tokenizer)
        self.eval_data_loader.process()
        self.trainer = transformers.Trainer(
            model=self.model,
            args=transformers.TrainingArguments(
                output_dir=self.get_old_generator_llm_location()
            )
        )
    
    def init_evaluator(self):
        self.eval_data_loader = EvalDatasetLoader(eval_dataset, self.tokenizer)
        self.eval_data_loader.process()
        self.model = self.ft_model
        self.model.to(self.device)

    def finetuning(self):
        lamda = 0.25
        num_of_prompts_in_a_datapoint = 12
        num_of_entities_in_a_datapoint = 7
        print(f"There are {self.training_data_loader_classification.len()} datapoints in the classification training dataset")
        logging.info(f"There are {self.training_data_loader_classification.len()} datapoints in the classification training dataset")
        print(f"There are {self.training_data_loader_ranking.len()} datapoints in the ranking training dataset")
        logging.info(f"There are {self.training_data_loader_ranking.len()} datapoints in the ranking training dataset")
        optimizer = bnb.optim.Adam8bit(self.model.parameters(), lr=2.5e-5)
        self.model.train()
        max_steps_counter = 0
        number_of_ranking_training_datapoints_till_now = 0
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
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            last_hidden_state = outputs.hidden_states[self.which_hidden_state]
            last_token_embedding = last_hidden_state[:, -1, :]
            embedding1 = last_token_embedding[:n]
            embedding2 = last_token_embedding[n:n + n]
            embedding3 = last_token_embedding[n + n:]
            loss2 = compute_sign_only_loss(embedding1, embedding2, embedding3, labels)

            loss = loss1 + lamda*loss2
            loss.backward()
            optimizer.step()
            self.training_loss_queue.add(loss.item())
            print(f"Till now {number_of_classification_training_datapoints_till_now} classification datapoints used for training, "
                  f"Till now {number_of_ranking_training_datapoints_till_now} ranking datapoints used for training, "
                  f"Loss: {loss.item()}, "
                  f"Sum of last {self.training_loss_queue.size} losses = {self.training_loss_queue.average()}, "
                  f"Steps: {max_steps_counter} out of {configuration.max_steps}")
            logging.info(f"Till now {number_of_classification_training_datapoints_till_now} classification datapoints used for training, "
                         f"Till now {number_of_ranking_training_datapoints_till_now} ranking datapoints used for training, "
                         f"Loss: {loss.item()}, "
                         f"Sum of last {self.training_loss_queue.size} losses = {self.training_loss_queue.average()}, "
                         f"Steps: {max_steps_counter} out of {configuration.max_steps}")
            if max_steps_counter % configuration.eval_steps == 0:
                # self.eval()
                # self.model.train()
                pass
            if max_steps_counter == configuration.save_steps:
                self.trainer.save_model()

    def determine_tokenizer_max_length(self):
        self.token_sizes = (self.training_data_loader_classification.token_sizes + 
                            self.training_data_loader_ranking.token_sizes + 
                            self.eval_data_loader.token_sizes)
        self.print_prompts_statistics()


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
    load_train_eval_datasets()
    obj = ConceptEmbedderViaRankingAndClassification()
    if only_eval_mode:
        obj.init_ft_model(obj.get_generator_llm())
        obj.init_evaluator()
        obj.eval()
    else:
        logging.info(f"Going to train {configuration.base_model_id} for {configuration.max_steps} max_steps ")
        obj.init_model()
        obj.init_trainer()
        obj.determine_tokenizer_max_length()
        obj.finetuning()
        obj.rename_finetuned_model_path()
        logging.info(f'Finetuning completed')
