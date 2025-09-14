import os
import sys
import shutil
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import deque
import bitsandbytes as bnb

from peft import PeftModel
from transformers import AutoModelForCausalLM

import evaluate
from sklearn.metrics import classification_report

from conf import configuration
from models.llm import LLM
from models.prompts_pairwise import train_text, train_label, load_datasets


def initialization():
    log_file = os.path.join(configuration.logging_folder,
                            os.path.splitext(os.path.basename(__file__))[0] + '.log')
    logging.basicConfig(format='%(levelname)s %(asctime)s:  %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p', filename=log_file, filemode='w', level=logging.INFO)
    with open(configuration.configuration_file_to_consider, 'r') as handle:
        conf = handle.read()
    logging.info(conf)
    

metric = evaluate.load("accuracy")

train_dataset, eval_dataset = None, None

def load_train_eval_datasets():
    global train_dataset, eval_dataset
    train_dataset, eval_dataset = load_datasets()


def process_a_batch(batch):
    input_ids_list = []
    attention_mask_list = []
    labels_list = []
    for item in batch:
        input_ids_list.append(item['input_ids'])
        attention_mask_list.append(item['attention_mask'])
        labels_list.append(item['labels'])
    input_ids = torch.stack(input_ids_list, dim=0)
    attention_mask = torch.stack(attention_mask_list, dim=0)
    labels = torch.stack(labels_list, dim=0)
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels
    }


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


class TrainDatasetLoader:
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
        prompt = train_text(datapoint)
        label = train_label(datapoint)
        result = self.tokenizer(
            prompt,
            truncation=True,
            max_length=configuration.tokenizer_max_length,
            padding="max_length",
            return_tensors="pt"
        )
        self.token_sizes.append(len(self.tokenizer(prompt)['input_ids']))
        return {
            'input_ids': result['input_ids'].flatten(),
            'attention_mask': result['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
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


class EvalDatasetLoader(TrainDatasetLoader):
    def __init__(self, datapoints, tokenizer):
        super().__init__(datapoints, tokenizer)



# class MultiClassClassifier(nn.Module):
#     def __init__(self, embedding_dim, num_classes):
#         super(MultiClassClassifier, self).__init__()
#         self.fc = nn.Linear(embedding_dim, num_classes)

#     def forward(self, x):
#         return self.fc(x)


class BinaryClassifier(nn.Module):
    def __init__(self, embedding_dim):
        super(BinaryClassifier, self).__init__()
        self.fc = nn.Linear(embedding_dim, 1)

    def forward(self, x):
        return self.fc(x).squeeze(-1)

class LLMClassifier(LLM):

    def __init__(self):
        super().__init__()
        self.init_conf()
        self.training_data_loader = None
        self.eval_data_loader = None
        self.training_loss_queue = FixedSizeLossQueue(100)
    
    # def init_classifier(self):
    #     embedding_dim = self.model.config.hidden_size
    #     num_classes = 2
    #     self.multi_class_classifier = MultiClassClassifier(embedding_dim, num_classes)

    def init_classifier(self):
        embedding_dim = self.model.config.hidden_size
        self.multi_class_classifier = BinaryClassifier(embedding_dim).to(self.device, dtype=self.model.dtype)


    def init_ft_model(self):
        location = self.get_generator_llm()
        logging.info(f"Loading finetuned model from {location}")
        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_id,
            quantization_config=self.bnb_config,
            output_hidden_states=True
        )
        self.ft_model = PeftModel.from_pretrained(base_model, location)
        self.ft_model.config.pad_token_id = self.ft_model.config.eos_token_id
        self.ft_model.to(self.device)
        embedding_dim = self.ft_model.config.hidden_size
        self.multi_class_classifier = BinaryClassifier(embedding_dim).to(self.device, dtype=self.ft_model.dtype)
        classifier_path = os.path.join(location, "classifier_head.pt")
        if not os.path.exists(classifier_path):
            raise FileNotFoundError(f"Classifier head not found at: {classifier_path}")
        self.multi_class_classifier.load_state_dict(torch.load(classifier_path, map_location=self.device))
        self.multi_class_classifier.to(self.device)
        self.multi_class_classifier.eval()

    def get_old_generator_llm_location(self):
        generator_llm = f"pairwise-checkpoint-{configuration.save_steps}"
        generator_llm = os.path.join(os.path.join(configuration.learned_models, self.base_model_id), generator_llm)
        return generator_llm

    def get_generator_llm(self):
        base_file_name = os.path.splitext(os.path.basename(configuration.training_dataset_ranking))[0]
        generator_llm = f"pairwise-checkpoint{configuration.save_steps}-{base_file_name}"
        generator_llm = os.path.join(os.path.join(configuration.learned_models, self.base_model_id), generator_llm)
        return generator_llm

    def determine_tokenizer_max_length(self):
        self.token_sizes = self.training_data_loader.token_sizes + self.eval_data_loader.token_sizes
        self.print_prompts_statistics()

    def init_trainer(self):
        self.training_data_loader = TrainDatasetLoader(train_dataset, self.tokenizer)
        self.training_data_loader.process()
        self.eval_data_loader = EvalDatasetLoader(eval_dataset, self.tokenizer)
        self.eval_data_loader.process()

    def evaluate(self, eval_prompts, max_new_tokens=configuration.tokenizer_max_length):
        model_input = self.tokenizer(
            eval_prompts,
            truncation=True,
            max_length=max_new_tokens,
            padding="max_length",
            return_tensors="pt"
        ).to("cuda")
        self.ft_model.eval()
        with torch.no_grad():
            eval_outputs = self.ft_model(**model_input)
            last_hidden_state = eval_outputs.hidden_states[-1]
            last_token_embedding = last_hidden_state[:, -1, :]
            embedding = last_token_embedding
            eval_logits = self.multi_class_classifier(embedding) 
            # eval_predictions = torch.argmax(eval_logits, dim=1)
            eval_predictions = (torch.sigmoid(eval_logits) > 0.5).long()
            return eval_predictions

    def eval(self):
        counter = 0
        self.model.eval()
        self.multi_class_classifier.eval()
        references = torch.tensor([], dtype=torch.long).to(self.device)
        predictions = torch.tensor([], dtype=torch.long).to(self.device)

        with torch.no_grad():
            for batch in self.eval_data_loader.batches:
                batch = process_a_batch(batch)
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)  # integer class labels
                n = input_ids.size(0)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                last_hidden_state = outputs.hidden_states[-1]
                last_token_embedding = last_hidden_state[:, -1, :]
                embedding = last_token_embedding
                # embedding = F.normalize(embedding, p=2, dim=1)

                logits = self.multi_class_classifier(embedding)  # logits for each class
                # prediction = torch.argmax(logits, dim=1)

                prediction = (torch.sigmoid(logits) > 0.5).long()

                references = torch.cat((references, labels), dim=0)
                predictions = torch.cat((predictions, prediction), dim=0)

                counter += n
                logging.info(f'Out of {self.eval_data_loader.len()} eval datapoints {counter} is processed')
                print(f'Out of {self.eval_data_loader.len()} eval datapoints {counter} is processed')

        # Assuming metric supports multi-class classification directly:
        result = metric.compute(predictions=predictions, references=references)
        logging.info(f"Average eval accuracy is {result['accuracy']}")
        print(f"Average eval accuracy is {result['accuracy']}")

        # Show classification report
        y_true = references.cpu().numpy()
        y_pred = predictions.cpu().numpy()
        report = classification_report(y_true, y_pred, digits=4)
        logging.info(f"Classification Report:\n{report}")
        print(f"Classification Report:\n{report}")

        self.model.train()
        self.multi_class_classifier.train()


    def finetuning(self):
        self.multi_class_classifier.to(self.device)
        print(f"There are {self.training_data_loader.len()} datapoints in the training dataset")
        logging.info(f"There are {self.training_data_loader.len()} datapoints in the training dataset")

        optimizer = bnb.optim.Adam8bit(
            list(self.model.parameters()) + list(self.multi_class_classifier.parameters()),
            lr=2.5e-5
        )

        self.model.train()
        self.multi_class_classifier.train()
        max_steps_counter = 0
        number_of_training_datapoints_till_now = 0
        while max_steps_counter < configuration.max_steps:
            batch = self.training_data_loader.get_a_minibatch()
            batch = process_a_batch(batch)
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            n = input_ids.size(0)
            number_of_training_datapoints_till_now += n
            max_steps_counter += 1
            optimizer.zero_grad()
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            last_hidden_state = outputs.hidden_states[-1]
            last_token_embedding = last_hidden_state[:, -1, :]
            embedding = last_token_embedding
            # embedding = F.normalize(embedding, p=2, dim=1)

            logits = self.multi_class_classifier(embedding)
            # loss = F.cross_entropy(logits, labels)

            labels = labels.float() 
            loss = F.binary_cross_entropy_with_logits(logits, labels)

            loss.backward()
            optimizer.step()

            self.training_loss_queue.add(loss.item())
            print(f"Till now {number_of_training_datapoints_till_now} datapoints used for training, "
                  f"Loss: {loss.item()}, "
                  f"Sum of last {self.training_loss_queue.size} losses = {self.training_loss_queue.average()}, "
                  f"Steps: {max_steps_counter} out of {configuration.max_steps}")
            logging.info(f"Till now {number_of_training_datapoints_till_now} datapoints used for training, "
                         f"Loss: {loss.item()}, "
                         f"Sum of last {self.training_loss_queue.size} losses = {self.training_loss_queue.average()}, "
                         f"Steps: {max_steps_counter} out of {configuration.max_steps}")
            if max_steps_counter % configuration.eval_steps == 0:
                # self.eval()
                pass
            if max_steps_counter == configuration.save_steps:
                save_path = self.get_old_generator_llm_location()
                if os.path.exists(save_path):
                    shutil.rmtree(save_path)
                os.makedirs(save_path, exist_ok=True)
                self.model.save_pretrained(save_path)
                self.tokenizer.save_pretrained(save_path)
                classifier_path = os.path.join(save_path, "classifier_head.pt")
                torch.save(self.multi_class_classifier.state_dict(), classifier_path)


if __name__ == '__main__':

    configuration.batch_size = 4
    configuration.max_steps = 2000
    configuration.save_steps = 2000
    configuration.eval_steps = int(configuration.max_steps / 1)
    # configuration.base_model_id = "allenai/OLMo-2-1124-13B"
    configuration.base_model_id = "mistralai/Mistral-Small-24B-Base-2501"

    # if sys.argv[1] == "Meta-Llama-3-8B":
    #     configuration.base_model_id = "meta-llama/Meta-Llama-3-8B"
    # elif sys.argv[1] == "Qwen3-8B":
    #     configuration.base_model_id = "Qwen/Qwen3-8B"
    # elif sys.argv[1] == "Qwen3-14B":
    #     configuration.base_model_id = "Qwen/Qwen3-14B"
    # elif sys.argv[1] == "Mistral-Nemo-Base-2407":
    #     configuration.base_model_id = "mistralai/Mistral-Nemo-Base-2407"
    # elif sys.argv[1] == "Mistral-Small-24B-Base-2501":
    #     configuration.base_model_id = "mistralai/Mistral-Small-24B-Base-2501"
    # elif sys.argv[1] == "Phi-4":
    #     configuration.base_model_id = "microsoft/phi-4"
    # elif sys.argv[1] == "OLMo-2-1124-7B":
    #     configuration.base_model_id = "allenai/OLMo-2-1124-7B"
    # elif sys.argv[1] == "OLMo-2-1124-13B":
    #     configuration.base_model_id = "allenai/OLMo-2-1124-13B"
    # else:
    #     raise Exception(f"Unknown model name {sys.argv[1]}")

    initialization()
    load_train_eval_datasets()
    obj = LLMClassifier()
    obj.init_model()
    obj.init_classifier()
    obj.init_trainer()
    obj.determine_tokenizer_max_length()
    obj.finetuning()
    obj.rename_finetuned_model_path()