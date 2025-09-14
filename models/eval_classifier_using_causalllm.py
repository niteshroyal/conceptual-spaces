import os
import json
import logging
from models.classifier_using_causalllm import LLMClassifier

from conf import configuration
from models.prompts_pairwise import train_text, train_label


def initialization():
    log_file = os.path.join(configuration.logging_folder, os.path.splitext(os.path.basename(__file__))[0] + '.log')
    logging.basicConfig(format='%(levelname)s %(asctime)s:  %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p', filename=log_file, filemode='w', level=logging.INFO)


class EvalLLMClassifier(LLMClassifier):

    def __init__(self):
        super().__init__()
        self.fine_tuned_model_location = None
        self.ft_model = None
        self.init_conf()

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
    batch_size = 2
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
            if len(batch_texts) == batch_size:
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
    # configuration.base_model_id = "facebook/bart-base"
    # configuration.base_model_id = "Qwen/Qwen3-8B"
    # configuration.base_model_id = "Qwen/Qwen3-14B"
    # configuration.base_model_id = "mistralai/Mistral-Nemo-Base-2407"
    # configuration.base_model_id = "mistralai/Mistral-Small-24B-Base-2501"
    configuration.base_model_id = "microsoft/phi-4"
    # configuration.base_model_id = "allenai/OLMo-2-1124-7B"
    # configuration.base_model_id = "allenai/OLMo-2-1124-13B"


    logging.info(f'Running base model = {configuration.base_model_id}')
    obj = EvalLLMClassifier()
    obj.init_ft_model()

    configuration.validation_dataset = os.path.join(configuration.data_folder, "easy_test_tastes_2040.jsonl")
    logging.info(f'Validation file is set to {configuration.validation_dataset}')
    groups = ["sweet taste food item", "salty taste food item", "sour taste food item", 
              "bitter taste food item", "umami taste food item", "fatty taste food item"]
    for group in groups:
        evaluate_on_validation_dataset(obj, group)
    logging.info(f'Validation file evaluated is {configuration.validation_dataset}')


    del obj
