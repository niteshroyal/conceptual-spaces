import os
import hashlib
import logging

import torch
import torch.nn.functional as F
from conf import configuration
from models.concept_embedder import ConceptEmbedderViaRankingAndClassification
from models.concept_embedder_only_classification import ConceptEmbedderViaClassification
from models.concept_embedder_only_ranking import ConceptEmbedderViaRanking


def initialization():
    log_file = os.path.join(configuration.logging_folder,
                            os.path.splitext(os.path.basename(__file__))[0] + '.log')
    logging.basicConfig(format='%(levelname)s %(asctime)s:  %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p', filename=log_file, filemode='w', level=logging.INFO)


def get_hash(key):
    return hashlib.md5(key.encode()).hexdigest()


class EvalConceptEmbedderViaRankingAndClassification:
    
    def __init__(self):
        self.finetuned_model = None
        self.embedding_cache = dict()


    def init(self):
        self.finetuned_model = ConceptEmbedderViaRankingAndClassification()
        location = self.finetuned_model.get_generator_llm()
        self.finetuned_model.init_ft_model(location)
        logging.info(f"Loading fine tuned model from {location}")

    
    def compute_embeddings_for_absent_concepts(self, concepts):
        absent_concepts = [concept for concept in concepts if get_hash(concept) not in self.embedding_cache]
        if absent_concepts:
            embeddings = self.finetuned_model.get_embeddings_batch(absent_concepts).tolist()
            for concept, embd in zip(absent_concepts, embeddings):
                self.embedding_cache[get_hash(concept)] = embd

    def get_concept_embedding(self, concept):
        if isinstance(concept, list):
            self.compute_embeddings_for_absent_concepts(concept)
            return [self.embedding_cache[get_hash(item)] for item in concept]
        else:
            if get_hash(concept) not in self.embedding_cache:
                self.compute_embeddings_for_absent_concepts([concept])
            return self.embedding_cache[get_hash(concept)]
    
    def evaluate(self, datapoints_coverted_to_prompts):
        entity1_prompts = []
        entity2_prompts = []
        property_prompts = []
        for entity1_prompt, entity2_prompt, property_prompt in datapoints_coverted_to_prompts:
            entity1_prompts.append(entity1_prompt)
            entity2_prompts.append(entity2_prompt)
            property_prompts.append(property_prompt)
        all_prompts = entity1_prompts + entity2_prompts + property_prompts
        all_embeddings = self.finetuned_model.get_embeddings_batch(all_prompts)
        entity1_embeddings = all_embeddings[:len(entity1_prompts)]
        entity2_embeddings = all_embeddings[len(entity1_prompts):len(entity1_prompts) + len(entity2_prompts)]
        property_embeddings = all_embeddings[len(entity1_prompts) + len(entity2_prompts):]
        logits = torch.sum((entity1_embeddings - entity2_embeddings) * property_embeddings, dim=1)
        prediction = (logits > 0).long()
        return prediction
    
class EvalConceptEmbedderViaClassification(EvalConceptEmbedderViaRankingAndClassification):

    def __init__(self):
        super().__init__()

    def init(self):
        self.finetuned_model = ConceptEmbedderViaClassification()
        location = self.finetuned_model.get_generator_llm()
        self.finetuned_model.init_ft_model(location)
        logging.info(f"Loading fine tuned model from {location}")


class EvalConceptEmbedderViaRanking(EvalConceptEmbedderViaRankingAndClassification):

    def __init__(self):
        super().__init__()

    def init(self):
        self.finetuned_model = ConceptEmbedderViaRanking()
        location = self.finetuned_model.get_generator_llm()
        self.finetuned_model.init_ft_model(location)
        logging.info(f"Loading fine tuned model from {location}")


class PreTrainedModel(ConceptEmbedderViaRankingAndClassification):

    def __init__(self):
        super().__init__()


    def get_embeddings(self, list_of_texts):
        model_input = self.tokenizer(
            list_of_texts,
            truncation=True,
            max_length=configuration.tokenizer_max_length,
            padding="max_length",
            return_tensors="pt"
        ).to("cuda")
        input_ids_list = self.tokenizer(list_of_texts)['input_ids']
        for ids in input_ids_list:
            self.token_sizes.append(len(ids))
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**model_input)
        last_hidden_state = outputs.hidden_states[self.which_hidden_state]
        last_token_embedding = last_hidden_state[:, -1, :]
        embeddings = F.normalize(last_token_embedding, p=2, dim=1)
        return embeddings


class EvalPreTrainedModel(EvalConceptEmbedderViaRankingAndClassification):

    def __init__(self):
        super().__init__()
        self.finetuned_model = PreTrainedModel()
        self.finetuned_model.init_model()
