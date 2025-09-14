import os
import torch
import random
import logging
from models.llm import LLM

from conf import configuration
from models.eval_classifier import evaluate_on_validation_dataset


def initialization():
    log_file = os.path.join(configuration.logging_folder, os.path.splitext(os.path.basename(__file__))[0] + '.log')
    logging.basicConfig(format='%(levelname)s %(asctime)s:  %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p', filename=log_file, filemode='w', level=logging.INFO)


class ZeroShot(LLM):

    def __init__(self):
        super().__init__()
        self.init_conf()
        self.init_model()

    def evaluate(self, eval_prompts, tokenizer_max_length=configuration.tokenizer_max_length):
        max_length = tokenizer_max_length + 100
        new_eval_prompts = []
        for prompt in eval_prompts:
            new_eval_prompts.append('The task is to answer questions that involve comparing perceptual features of two entities. '
                                    'Please answer with Yes or No only. In the worst case, if you do not know the answer then choose randomly between Yes and No.\n'
                                    'This question is about two surfaces: Is mirror more reflective than still water surface?\nYes\n'
                                    'This question is about two materials: Is silk fabric more lustrous than polished metal?\nNo\n' +
                                    'This question is about two sounds: Is operatic aria more melodious than car alarm?\nYes\n' +
                                    prompt + '\n')
        inputs = self.tokenizer(
            new_eval_prompts,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt"
        ).to("cuda")
        self.model.eval()

        results = self.tokenizer(new_eval_prompts)

        for i in range(len(results['input_ids'])):
            if len(results['input_ids'][i]) >= max_length:
                raise ValueError(f"Input {i} exceeds max length: {len(results['input_ids'][i])}")
            else:
                pass
        

        with torch.no_grad():
            g_str = self.model.generate(**inputs, max_new_tokens=10, pad_token_id=2)

            prediction = []
            for i in range(0, len(g_str)):
                st = self.tokenizer.decode(g_str[i], skip_special_tokens=True)
                logging.info(st)
                pred = process_string(st)
                logging.info(f'Answer part in process string = {pred}')
                prediction.append(pred)
            prediction = torch.tensor(prediction)
        return prediction

def process_string(text):
    lines = text.strip().split('\n')
    if len(lines) > 8:
        target_line = lines[8].strip()
        first_word = target_line.split()[0].lower()
        if "yes" in first_word:
            return 1
        elif "no" in first_word:
            return 0
        else:
            pass
    return random.randint(0, 1)


if __name__ == '__main__':
    initialization()

    configuration.batch_size = 4
    configuration.base_model_id = "meta-llama/Meta-Llama-3-8B"

    obj = ZeroShot()

    configuration.validation_dataset = os.path.join(configuration.data_folder, "easy_test_tastes_2040.jsonl")
    logging.info(f'Validation file is set to {configuration.validation_dataset}')
    groups = ["sweet taste food item", "salty taste food item", "sour taste food item", 
              "bitter taste food item", "umami taste food item", "fatty taste food item"]
    for group in groups:
        evaluate_on_validation_dataset(obj, group)
    logging.info(f'Validation file evaluated is {configuration.validation_dataset}')