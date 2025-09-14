from datasets import load_dataset

from conf import configuration


def load_datasets():
    train_dataset = load_dataset('json', data_files=configuration.training_dataset_ranking, split='train').shuffle(seed=42)
    eval_dataset = load_dataset('json', data_files=configuration.validation_dataset, split='train').shuffle(seed=42)
    return train_dataset, eval_dataset

def train_text(example):
    text = example['question']
    return text

def train_label(example):
    label = example['answer']
    if label == "Yes":
        return 1
    elif label == "No":
        return 0
    else:
        raise Exception('Incorrect label')
