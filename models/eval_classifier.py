import os
import sys
import json
import torch
import logging
from peft import PeftModel
from transformers import AutoModelForSequenceClassification
from models.classifier import LargeLanguageModelClassifier

from conf import configuration
from models.prompts_pairwise import train_text, train_label


# def initialization():
#     log_file = os.path.join(configuration.logging_folder, os.path.splitext(os.path.basename(__file__))[0] +
#                             '_' + sys.argv[1] + '_.log')
#     logging.basicConfig(format='%(levelname)s %(asctime)s:  %(message)s',
#                         datefmt='%m/%d/%Y %I:%M:%S %p', filename=log_file, filemode='w', level=logging.INFO)

def initialization():
    log_file = os.path.join(configuration.logging_folder, os.path.splitext(os.path.basename(__file__))[0] + '.log')
    logging.basicConfig(format='%(levelname)s %(asctime)s:  %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p', filename=log_file, filemode='w', level=logging.INFO)



class EvalLargeLanguageModel(LargeLanguageModelClassifier):

    def __init__(self):
        super().__init__()
        self.fine_tuned_model_location = None
        self.ft_model = None
        self.init_conf()

    def init_ft_model(self):
        self.fine_tuned_model_location = self.get_generator_llm()
        base_model = AutoModelForSequenceClassification.from_pretrained(
            self.base_model_id,
            num_labels=2,
            quantization_config=self.bnb_config
        )
        location = os.path.join(self.fine_tuned_model_location, f"checkpoint-{configuration.save_steps}")
        print(f"Loading model from {location}")
        logging.info(f"Loading model from {location}")
        self.ft_model = PeftModel.from_pretrained(base_model, location)
        self.ft_model.config.pad_token_id = self.ft_model.config.eos_token_id

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
            eval_logits = eval_outputs.logits
            eval_predictions = torch.argmax(eval_logits, dim=1)
            return eval_predictions


# def evaluate_on_validation_dataset(tester):
#     hits = []
#     counter = 0
#     with open(configuration.validation_dataset, 'r') as f:
#         for line in f:
#             datapoint = json.loads(line)
#             text = train_text(datapoint)
#             answer = train_label(datapoint)
#             prediction = tester.evaluate(text)
#             if (prediction == 0).item():
#                 p_answer = 0
#             else:
#                 p_answer = 1
#             if answer == p_answer:
#                 hits.append(1)
#             else:
#                 hits.append(0)
#             counter += 1
#             logging.info(f"Answer: {answer}, Predicted answer: {p_answer}")
#             if counter % 10 == 0:
#                 text = (f"Number of validation datapoints processed = {counter}, "
#                         f"Accuracy till now is {(sum(hits) * 100) / len(hits)}")
#                 logging.info(text)
#                 print(text)
#     accuracy = (sum(hits) * 100) / len(hits)
#     text = f"Final accuracy is {accuracy}%"
#     logging.info(text)
#     print(text)

new_datapoints = []


def write_results():
    global new_datapoints
    file_name = configuration.validation_dataset
    path, extension = file_name.rsplit('.', 1)
    file_name = f"{path}_results.{extension}"
    with open(file_name, 'w', encoding='utf-8') as file_handler:
        for item in new_datapoints:
            json.dump(item, file_handler)
            file_handler.write('\n')


def record_results(batch_datapoints, batch_p_answers):
    global new_datapoints
    temp = []
    for i, datapoint in enumerate(batch_datapoints):
        datapoint['predicted_answer'] = batch_p_answers[i]
        temp.append(datapoint)
    new_datapoints += temp


def evaluate_on_validation_dataset(tester, considered_group=None):
    hits = []
    counter = 0
    batch_texts = []
    batch_answers = []
    batch_p_answers = []
    batch_datapoints = []
    with open(configuration.validation_dataset, 'r') as f:
        for line in f:
            datapoint = json.loads(line)
            if considered_group is None:
                pass
            else:
                if datapoint["property"] != considered_group:
                    continue
            text = train_text(datapoint)
            answer = train_label(datapoint)
            batch_texts.append(text)
            batch_answers.append(answer)
            batch_datapoints.append(datapoint)
            if len(batch_texts) == configuration.batch_size:
                predictions = tester.evaluate(batch_texts)
                for i, prediction in enumerate(predictions):
                    p_answer = prediction.item()
                    answer = batch_answers[i]
                    batch_p_answers.append(p_answer)
                    if answer == p_answer:
                        hits.append(1)
                    else:
                        hits.append(0)
                counter += len(batch_texts)
                record_results(batch_datapoints, batch_p_answers)
                batch_p_answers = []
                batch_datapoints = []
                batch_texts = []
                batch_answers = []
                logging.info(f"Processed {counter} datapoints, Accuracy till now: {(sum(hits) * 100) / len(hits)}%")
                print(f"Processed {counter} datapoints, Accuracy till now: {(sum(hits) * 100) / len(hits)}%")
    if batch_texts:
        predictions = tester.evaluate(batch_texts)
        for i, prediction in enumerate(predictions):
            p_answer = prediction.item()
            answer = batch_answers[i]
            batch_p_answers.append(p_answer)
            if answer == p_answer:
                hits.append(1)
            else:
                hits.append(0)
        counter += len(batch_texts)
        record_results(batch_datapoints, batch_p_answers)
        logging.info(f"Final batch processed. Total processed {counter} datapoints.")
    accuracy = (sum(hits) * 100) / len(hits)
    text = f"Final accuracy is {accuracy}% for {considered_group}"
    logging.info(text)
    print(text)
    write_results()


if __name__ == '__main__':
    initialization()
    configuration.batch_size = 4
    configuration.max_steps = 2000
    configuration.save_steps = 2000



    # configuration.base_model_id = "meta-llama/Meta-Llama-3-8B"
    # configuration.base_model_id = "Qwen/Qwen3-8B"
    configuration.base_model_id = "Qwen/Qwen3-14B"
    # configuration.base_model_id = "mistralai/Mistral-Nemo-Base-2407"
    # configuration.base_model_id = "mistralai/Mistral-Small-24B-Base-2501"
    # configuration.base_model_id = "microsoft/phi-4"
    # configuration.base_model_id = "allenai/OLMo-2-1124-7B"
    # configuration.base_model_id = "allenai/OLMo-2-1124-13B"




    logging.info(f'Running base model = {configuration.base_model_id}')
    obj = EvalLargeLanguageModel()
    obj.init_ft_model()


    configuration.validation_dataset = os.path.join(configuration.data_folder, "easy_test_tastes_2040.jsonl")
    logging.info(f'Validation file is set to {configuration.validation_dataset}')
    groups = ["sweet taste food item", "salty taste food item", "sour taste food item", 
              "bitter taste food item", "umami taste food item", "fatty taste food item"]
    for group in groups:
        evaluate_on_validation_dataset(obj, group)
    logging.info(f'Validation file evaluated is {configuration.validation_dataset}')


    del obj
