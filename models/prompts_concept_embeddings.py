import logging

from datasets import load_dataset

from conf import configuration


def load_datasets():
    train_dataset_classification = load_dataset('json', data_files=configuration.training_dataset_classification, split='train').shuffle(seed=42)
    train_dataset_ranking = load_dataset('json', data_files=configuration.training_dataset_ranking, split='train').shuffle(seed=42)
    eval_dataset = load_dataset('json', data_files=configuration.validation_dataset, split='train').shuffle(seed=42)

    # # Write line by line data in train_dataset_classification to a file for manual inspection
    # f = open(f"{configuration.logging_folder}/train_dataset_classification_for_manual_inspection.jsonl", "w")
    # for example in train_dataset_classification:
    #     f.write(str(example).replace("'", '"') + "\n")
    # f.close()

    return train_dataset_classification, train_dataset_ranking, eval_dataset


def get_eval_record(example):
    entity1 = example["entity1"]
    entity2 = example["entity2"]
    entity_type = example["entity_type"]
    prop = example["high_property"]

    if entity_type == "rock":
        prompt1 = get_prompt(entity1 + ' rock')
        prompt2 = get_prompt(entity2 + ' rock')
    elif entity_type == "food item":
        prompt1 = get_prompt('food item - ' + entity1)
        prompt2 = get_prompt('food item - ' + entity2)
        # prompt1 = get_prompt(entity1)
        # prompt2 = get_prompt(entity2)
    elif entity_type == "thing":
        prompt1 = get_prompt(entity1 + ' odour')
        prompt2 = get_prompt(entity2 + ' odour')
    elif entity_type == "music":
        prompt1 = get_prompt('music - ' + entity1)
        prompt2 = get_prompt('music - ' + entity2)
    elif entity_type == "movie":
        prompt1 = get_prompt('movie - ' + entity1)
        prompt2 = get_prompt('movie - ' + entity2)
    elif entity_type == "book":
        prompt1 = get_prompt('book - ' + entity1)
        prompt2 = get_prompt('book - ' + entity2)
    else:
        prompt1 = get_prompt(entity1)
        prompt2 = get_prompt(entity2)

    relation = get_prompt(prop)
    logging.info(f"Prompt1: {prompt1}")
    logging.info(f"Prompt2: {prompt2}")
    logging.info(f"Relation: {relation}")

    return [prompt1, prompt2], [relation]


def get_eval_label(example):
    answer = example['answer']
    if answer == "Yes":
        return 1
    elif answer == "No":
        return 0
    else:
        raise Exception('Incorrect answer label')


def get_train_record(example):
    entity_type = example["entity type"]
    positive_prop = example["property"]
    negative_props = example["negatives"]
    list_of_entities = example["examples"]
    properties = [positive_prop + ' ' + entity_type] + negative_props

    properties_prompts = []
    for prop in properties:
        properties_prompts.append(get_prompt(prop))

    entities_prompts = []
    for entity in list_of_entities:
        entities_prompts.append(get_prompt(entity))
    
    logging.info(f"Entities prompts: {entities_prompts}")
    logging.info(f"Properties prompts: {properties_prompts}")

    return entities_prompts, properties_prompts


def get_record(example):
    entity1 = example["entity1"]
    entity2 = example["entity2"]
    prop = example["property"]

    prompt1 = get_prompt(entity1)
    prompt2 = get_prompt(entity2)
    relation = get_prompt(prop)

    logging.info(f"Prompt1: {prompt1}")
    logging.info(f"Prompt2: {prompt2}")
    logging.info(f"Relation: {relation}")

    return prompt1, prompt2, relation

def get_prompt(entity):
    entity = entity.lower()
    prompt = f"The description of the term '{entity}' in one word is "
    return prompt

def get_label(example):
    answer = example['answer']
    if answer == "Yes":
        return 1
    elif answer == "No":
        return 0
    else:
        raise Exception('Incorrect answer label')
    
