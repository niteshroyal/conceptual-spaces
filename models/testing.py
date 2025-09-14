import os
import logging

from conf import configuration
from models.eval_concept_embedder import EvalConceptEmbedderViaRankingAndClassification, EvalPreTrainedModel, EvalConceptEmbedderViaClassification, EvalConceptEmbedderViaRanking
from models.eval_concept_embedder_pretrained import EvalConceptEmbedderViaRankingAndClassificationPretrained, EvalConceptEmbedderViaRankingPretrained, EvalConceptEmbedderViaClassificationPretrained, EvalPreTrainedEmbeddingModel

from models.utils import evaluate_on_validation_dataset

def initialization():
    log_file = os.path.join(configuration.logging_folder, os.path.splitext(os.path.basename(__file__))[0] + '.log')
    logging.basicConfig(format='%(levelname)s %(asctime)s:  %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p', filename=log_file, filemode='w', level=logging.INFO)


if __name__ == '__main__':
    initialization()

    configuration.batch_size = 1
    configuration.max_steps = 123
    configuration.save_steps = 123
    # configuration.base_model_id = "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp"
    # configuration.base_model_id = "McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp"
    configuration.base_model_id = "nvidia/NV-Embed-v2"
    # configuration.base_model_id = "intfloat/e5-mistral-7b-instruct"

    logging.info(f'Running base model = {configuration.base_model_id}')


    # Hybrid model that combines ranking and classification
    obj = EvalConceptEmbedderViaRankingAndClassification()
    obj.init()

    # # The model is not fine-tuned
    # obj = EvalPreTrainedModel()

    # # This model use classification only
    # obj = EvalConceptEmbedderViaClassification()
    # obj.init()


    # # This model use ranking only
    # obj = EvalConceptEmbedderViaRanking()
    # obj.init()  


    # # This model use ranking and classification strategy; however, it uses pre-trained embedding model as the base model
    # obj = EvalConceptEmbedderViaRankingAndClassificationPretrained()
    # obj.init()

    # # This model use ranking strategy; however, it uses pre-trained embedding model as the base model
    # obj = EvalConceptEmbedderViaRankingPretrained()
    # obj.init()


    # # This model use classification strategy; however, it uses pre-trained embedding model as the base model
    # obj = EvalConceptEmbedderViaClassificationPretrained()
    # obj.init()


    # # This model use a pre-trained embedding model as the base model, which is not further fine-tuned
    # obj = EvalPreTrainedEmbeddingModel()

    
    configuration.validation_dataset = os.path.join(configuration.data_folder, "easy_test_tastes_2040.jsonl")
    logging.info(f'Validation file is set to {configuration.validation_dataset}')
    groups = ["sweet taste food item", "salty taste food item", "sour taste food item", 
              "bitter taste food item", "umami taste food item", "fatty taste food item"]
    for group in groups:
        evaluate_on_validation_dataset(obj, group)

    # configuration.validation_dataset = os.path.join(configuration.data_folder, "test_hard_ranking_dataset_rocks_280.jsonl")
    # logging.info(f'Validation file is set to {configuration.validation_dataset}')
    # groups = ["light colored type of rock", "coarse-grained type of rock", "rough textured type of rock", "shiny type of rock", 
    #           "uniform-textured type of rock", "variable-colored type of rock", "dense type of rock"]
    # for group in groups:
    #     evaluate_on_validation_dataset(obj, group)


    logging.info(f'Validation file evaluated is {configuration.validation_dataset}')

    del obj
