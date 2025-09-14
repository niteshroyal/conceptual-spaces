import os
from conf import configuration


train_dataset = ['rocks', 'movies', 'books', 'music', 'odours', 'wikidata1', 'wikidata2', 'physical']
test_dataset = ['tastes']

train_output_file = os.path.join(configuration.data_folder, f'ranking_dataset_{'+'.join(train_dataset)}.jsonl')
test_output_file = os.path.join(configuration.data_folder, f'ranking_dataset_{'+'.join(test_dataset)}.jsonl')
