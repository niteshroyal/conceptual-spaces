import os
import json
import logging
import random

from conf import configuration
from utils.openai_api import OpenAI


def initialization():
    log_file = os.path.join(configuration.logging_folder,
                            os.path.splitext(os.path.basename(__file__))[0] + f'-all-at-once.log')
    logging.basicConfig(format='%(levelname)s %(asctime)s:  %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p', filename=log_file, filemode='w', level=logging.INFO)


# def get_prompt():
#     prompt = (
#         f"Answer the following with Yes or No only. If you don't know then guess between Yes and No."
#     )
#     return prompt

def get_prompt():
    prompt = (
        f"Answer the following with Yes or No only. In the worst case, if you don't know the answer then choose "
        f"randomly between Yes and No."
    )
    return prompt


def get_finalprompt(question):
    prompt = get_prompt()
    finalprompt = (
        f"{prompt}\n\n"
        f"{question}"
    )
    return finalprompt


def extract_answer(text):
    text = text.strip()
    text = text.replace('.', '')
    return text


def get_answer(response):
    parts = response.split("### Answer:")
    if len(parts) < 2:
        return "No answer section found."
    response_section = parts[1].split("### End")[0]
    return response_section.strip()


class EvaluateGPT4(OpenAI):

    def __init__(self, model=None):
        super().__init__(model)
        random.seed(42)

    def evaluate_on_validation_dataset(self, considered_group=None):
        hits = []
        counter = 0
        no_answer = 0
        with open(configuration.validation_dataset, 'r') as f:
            for line in f:
                datapoint = json.loads(line)
                if considered_group is None:
                    pass
                else:
                    if datapoint['property'] != considered_group:
                        continue
                prompt = get_finalprompt(datapoint["question"])
                response = self.get_gpt_response(prompt)
                # print(response + '\t' + datapoint['answer'])
                answer = extract_answer(response)
                logging.info(f"Question: {datapoint['question']}, Answer: {datapoint['answer']}, "
                             f"Prediction: {answer}, Response: {response}")
                if answer not in ['Yes', 'No']:
                    answer = random.choice(['Yes', 'No'])
                    no_answer += 1
                else:
                    pass
                if datapoint['answer'] == answer:
                    hits.append(1)
                else:
                    hits.append(0)
                counter += 1
                if counter % 50 == 0:
                    logging.info(f"Number of validation datapoints processed = {counter}, "
                                 f"Accuracy till now is {(sum(hits) * 100) / len(hits)}")
                    print(f"Number of validation datapoints processed = {counter}, "
                          f"Accuracy till now is {(sum(hits) * 100) / len(hits)}")
        accuracy = (sum(hits) * 100) / len(hits)
        logging.info(f'Final pairwise accuracy for {considered_group} is {accuracy}%, '
                     f'Total number of datapoints processed is {counter}, No answer = {no_answer}/{counter}, '
                     f'Validation file = {configuration.validation_dataset}')
        print(f'Final pairwise accuracy for {considered_group} is {accuracy}%, '
              f'Total number of datapoints processed is {counter}, No answer = {no_answer}/{counter}, '
              f'Validation file = {configuration.validation_dataset}')


if __name__ == '__main__':
    initialization()

    # models = ["gpt-4-0613", "gpt-3.5-turbo-0613"]
    # models = ["gpt-4o-2024-11-20"]
    # models = ["gpt-4.1-2025-04-14"]
    # model = "gpt-4.1-2025-04-14"
    model = "gpt-4o-2024-11-20"

    for eval_data in ["rocks", "odours", "music", "movies", "books", "wikidata1", "wikidata2", "physical"]:
        difficulty = "easy"
        if eval_data == "tastes":
            test_dataset = ['tastes']
            num_test_examples_per_dataset = 2040
        elif eval_data == "rocks":
            test_dataset = ['rocks']
            num_test_examples_per_dataset = 2380
        elif eval_data == "odours":
            test_dataset = ['odours']
            num_test_examples_per_dataset = 1360
        elif eval_data == "music":
            test_dataset = ['music']
            num_test_examples_per_dataset = 3060
        elif eval_data == "movies":
            test_dataset = ['movies']
            num_test_examples_per_dataset = 500
        elif eval_data == "books":
            test_dataset = ['books']
            num_test_examples_per_dataset = 500
        elif eval_data == "wikidata1":
            test_dataset = ['wikidata1']
            num_test_examples_per_dataset = 500
        elif eval_data == "wikidata2":
            test_dataset = ['wikidata2']
            num_test_examples_per_dataset = 500
        elif eval_data == "physical":
            test_dataset = ['physical']
            num_test_examples_per_dataset = 500
        else:
            raise Exception(f"Unknown dataset name {eval_data}")

        test_dataset_str = '+'.join(test_dataset)

        test_output_file = os.path.join(configuration.data_folder, f'cross_domain_{difficulty}_test_{test_dataset_str}_{num_test_examples_per_dataset}.jsonl')

        configuration.validation_dataset = test_output_file

        obj = EvaluateGPT4(model)

        logging.info(f'OpenAI model is set to {model}')

        if eval_data == "tastes":    
            logging.info(f'Validation file is set to {configuration.validation_dataset}')
            groups = ["sweet taste food item", "salty taste food item", "sour taste food item", 
                    "bitter taste food item", "umami taste food item", "fatty taste food item"]
            for group in groups:
                obj.evaluate_on_validation_dataset(group)
        
        elif eval_data == "rocks":
            logging.info(f'Validation file is set to {configuration.validation_dataset}')
            groups = ["light colored type of rock", "coarse-grained type of rock", "rough textured type of rock", "shiny type of rock", 
                    "uniform-textured type of rock", "variable-colored type of rock", "dense type of rock"]
            for group in groups:
                obj.evaluate_on_validation_dataset(group)
        
        elif eval_data == "odours":
            logging.info(f'Validation file is set to {configuration.validation_dataset}')
            groups = ["pleasant odour", "intense odour", "irritating odour", "familiar odour"]
            for group in groups:
                obj.evaluate_on_validation_dataset(group)

        elif eval_data == "music":
            logging.info(f'Validation file is set to {configuration.validation_dataset}')
            groups = ["music that evokes the feeling of wonder", "music that evokes the feeling of tranquility", 
                    "music that evokes the feeling of tenderness", "music that evokes the feeling of nostalgia", 
                    "music that evokes the feeling of peace", "music that evokes the feeling of joy", 
                    "music that evokes the feeling of energy", "music that evokes the feeling of sadness", 
                    "music that evokes the feeling of tension"]
            for group in groups:
                obj.evaluate_on_validation_dataset(group)

        elif eval_data == "movies":
            logging.info(f'Validation file is set to {configuration.validation_dataset}')
            obj.evaluate_on_validation_dataset()

        elif eval_data == "books":
            logging.info(f'Validation file is set to {configuration.validation_dataset}')
            obj.evaluate_on_validation_dataset()

        elif eval_data == "wikidata1":
            logging.info(f'Validation file is set to {configuration.validation_dataset}')
            obj.evaluate_on_validation_dataset()

        elif eval_data == "wikidata2":
            logging.info(f'Validation file is set to {configuration.validation_dataset}')
            obj.evaluate_on_validation_dataset()

        elif eval_data == "physical":
            logging.info(f'Validation file is set to {configuration.validation_dataset}')
            groups = ["heavy entity", "tall entity", "large size entity"]
            for group in groups:
                obj.evaluate_on_validation_dataset(group)

        else:
            raise Exception(f"Unknown dataset name {eval_data}")

