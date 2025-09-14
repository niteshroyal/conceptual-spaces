import os
import sys
import logging
from conf import configuration

from models.concept_embedder import ConceptEmbedderViaRankingAndClassification, load_train_eval_datasets
from models.concept_embedder_only_classification import ConceptEmbedderViaClassification
from models.eval_concept_embedder import EvalConceptEmbedderViaRankingAndClassification, EvalConceptEmbedderViaClassification
from models.linear_mapping import EvalLinearMapping
from models.utils import evaluate_on_validation_dataset

def initialization():
    log_file = os.path.join(configuration.logging_folder, os.path.splitext(os.path.basename(__file__))[0] + '_' + sys.argv[1] + '_' + sys.argv[2] + '_' + sys.argv[3] + '.log')
    logging.basicConfig(format='%(levelname)s %(asctime)s:  %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p', filename=log_file, filemode='w', level=logging.INFO)



if __name__ == '__main__':
    max_steps = int(sys.argv[3])
    configuration.batch_size = 1
    configuration.max_steps = max_steps
    configuration.save_steps = max_steps
    configuration.eval_steps = int(configuration.max_steps / 1)
    configuration.tokenizer_max_length = 50 # 80

    if sys.argv[1] == "Meta-Llama-3-8B":
        configuration.base_model_id = "meta-llama/Meta-Llama-3-8B"
    elif sys.argv[1] == "Mistral-Small-24B-Base-2501":
        configuration.base_model_id = "mistralai/Mistral-Small-24B-Base-2501"
    else:
        raise Exception(f"Unknown model name {sys.argv[1]}")
    
    difficulty = "easy"
    if sys.argv[2] == "tastes":
        train_dataset = ['music', 'odours', 'rocks']
        # train_dataset = ['movies', 'books', 'music', 'odours', 'wikidata1', 'wikidata2', 'physical', 'rocks']
        test_dataset = ['tastes']
        num_train_examples_per_dataset = 500
        num_test_examples_per_dataset = 2040
    elif sys.argv[2] == "rocks":
        train_dataset = ['music', 'odours', 'tastes']
        test_dataset = ['rocks']
        num_train_examples_per_dataset = 500
        num_test_examples_per_dataset = 2380
    elif sys.argv[2] == "odours":
        train_dataset = ['music', 'tastes', 'rocks']
        test_dataset = ['odours']
        num_train_examples_per_dataset = 500
        num_test_examples_per_dataset = 1360
    elif sys.argv[2] == "music":
        train_dataset = ['odours', 'tastes', 'rocks']
        test_dataset = ['music']
        num_train_examples_per_dataset = 500
        num_test_examples_per_dataset = 3060
    elif sys.argv[2] == "movies":
        train_dataset = ['odours', 'music', 'tastes', 'rocks']
        test_dataset = ['movies']
        num_train_examples_per_dataset = 500
        num_test_examples_per_dataset = 500
    elif sys.argv[2] == "books":
        train_dataset = ['odours', 'music', 'tastes', 'rocks']
        test_dataset = ['books']
        num_train_examples_per_dataset = 500
        num_test_examples_per_dataset = 500
    elif sys.argv[2] == "wikidata1":
        train_dataset = ['odours', 'music', 'tastes', 'rocks']
        test_dataset = ['wikidata1']
        num_train_examples_per_dataset = 500
        num_test_examples_per_dataset = 500
    elif sys.argv[2] == "wikidata2":
        train_dataset = ['odours', 'music', 'tastes', 'rocks']
        test_dataset = ['wikidata2']
        num_train_examples_per_dataset = 500
        num_test_examples_per_dataset = 500
    elif sys.argv[2] == "physical":
        train_dataset = ['odours', 'music', 'tastes', 'rocks']
        test_dataset = ['physical']
        num_train_examples_per_dataset = 500
        num_test_examples_per_dataset = 500
    else:
        raise Exception(f"Unknown dataset name {sys.argv[2]}")

    train_dataset_str = '+'.join(train_dataset)
    test_dataset_str = '+'.join(test_dataset)


    if sys.argv[2] == "tastes":
        train_output_file = os.path.join(configuration.data_folder, f'{difficulty}_train_{train_dataset_str}_{num_train_examples_per_dataset}.jsonl')
        test_output_file = os.path.join(configuration.data_folder, f'{difficulty}_test_{test_dataset_str}_{num_test_examples_per_dataset}.jsonl')
    else:
        train_output_file = os.path.join(configuration.data_folder, f'cross_domain_{difficulty}_train_{train_dataset_str}_{num_train_examples_per_dataset}.jsonl')
        test_output_file = os.path.join(configuration.data_folder, f'cross_domain_{difficulty}_test_{test_dataset_str}_{num_test_examples_per_dataset}.jsonl')

    configuration.training_dataset_ranking = train_output_file
    configuration.validation_dataset = test_output_file


    initialization()
    # load_train_eval_datasets()
    # # obj = ConceptEmbedderViaRankingAndClassification()
    # obj = ConceptEmbedderViaClassification()
    # obj.init_model()
    # obj.init_trainer()
    # obj.determine_tokenizer_max_length()
    # obj.finetuning()
    # obj.rename_finetuned_model_path()


    # logging.info(f'Finetuning completed and model saved to {obj.get_generator_llm()}')

    # del obj

    logging.info('Starting evaluation')
    # obj = EvalConceptEmbedderViaRankingAndClassification()
    # obj.init()
    
    # obj = EvalConceptEmbedderViaClassification()
    # obj.init()

    obj = EvalLinearMapping()
    obj.init()


    if sys.argv[2] == "tastes":    
        logging.info(f'Validation file is set to {configuration.validation_dataset}')
        groups = ["sweet taste food item", "salty taste food item", "sour taste food item", 
                "bitter taste food item", "umami taste food item", "fatty taste food item"]
        for group in groups:
            evaluate_on_validation_dataset(obj, group)
    
    elif sys.argv[2] == "rocks":
        logging.info(f'Validation file is set to {configuration.validation_dataset}')
        groups = ["light colored type of rock", "coarse-grained type of rock", "rough textured type of rock", "shiny type of rock", 
                "uniform-textured type of rock", "variable-colored type of rock", "dense type of rock"]
        for group in groups:
            evaluate_on_validation_dataset(obj, group)
    
    elif sys.argv[2] == "odours":
        logging.info(f'Validation file is set to {configuration.validation_dataset}')
        groups = ["pleasant odour", "intense odour", "irritating odour", "familiar odour"]
        for group in groups:
            evaluate_on_validation_dataset(obj, group)

    elif sys.argv[2] == "music":
        logging.info(f'Validation file is set to {configuration.validation_dataset}')
        groups = ["music that evokes the feeling of wonder", "music that evokes the feeling of tranquility", 
                  "music that evokes the feeling of tenderness", "music that evokes the feeling of nostalgia", 
                  "music that evokes the feeling of peace", "music that evokes the feeling of joy", 
                  "music that evokes the feeling of energy", "music that evokes the feeling of sadness", 
                  "music that evokes the feeling of tension"]
        for group in groups:
            evaluate_on_validation_dataset(obj, group)

    elif sys.argv[2] == "movies":
        logging.info(f'Validation file is set to {configuration.validation_dataset}')
        evaluate_on_validation_dataset(obj)

    elif sys.argv[2] == "books":
        logging.info(f'Validation file is set to {configuration.validation_dataset}')
        evaluate_on_validation_dataset(obj)

    elif sys.argv[2] == "wikidata1":
        logging.info(f'Validation file is set to {configuration.validation_dataset}')
        evaluate_on_validation_dataset(obj)

    elif sys.argv[2] == "wikidata2":
        logging.info(f'Validation file is set to {configuration.validation_dataset}')
        evaluate_on_validation_dataset(obj)

    elif sys.argv[2] == "physical":
        logging.info(f'Validation file is set to {configuration.validation_dataset}')
        groups = ["heavy entity", "tall entity", "large size entity"]
        for group in groups:
            evaluate_on_validation_dataset(obj, group)
    else:
        raise Exception(f"Unknown dataset name {sys.argv[2]}")


    del obj
