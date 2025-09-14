import os
import json
import logging

from conf import configuration
from models.prompts_concept_embeddings import get_eval_record, get_eval_label

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

    batch_size = int(configuration.batch_size * 12)
    batch_size = int(batch_size / 3)
    batch_size = 2

    with open(os.path.join(configuration.logging_folder, 'hits.txt'), 'a') as g:
        with open(configuration.validation_dataset, 'r') as f:
            for line in f:
                datapoint = json.loads(line)
                if considered_group is None:
                    pass
                else:
                    if datapoint["property"] != considered_group:
                        continue
                [entity1_prompt, entity2_prompt], [property_prompt] = get_eval_record(datapoint)
                answer = get_eval_label(datapoint)
                batch_texts.append([entity1_prompt, entity2_prompt, property_prompt])
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
                            g.write(f"{considered_group} : 1\n")
                        else:
                            hits.append(0)
                            g.write(f"{considered_group} : 0\n")
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
                    g.write(f"{considered_group} : 1\n")
                else:
                    hits.append(0)
                    g.write(f"{considered_group} : 0\n")
            counter += len(batch_texts)
            record_results(batch_datapoints, batch_p_answers)
            logging.info(f"Final batch processed. Total processed {counter} datapoints.")
    accuracy = (sum(hits) * 100) / len(hits)
    text = f"Final accuracy is {accuracy}% for {considered_group}"
    logging.info(text)
    print(text)
    write_results()
