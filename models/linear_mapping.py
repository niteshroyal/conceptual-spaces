import os
import torch
import logging

import torch.nn.functional as F

from conf import configuration
from models.concept_embedder import ConceptEmbedderViaRankingAndClassification, TrainDatasetLoaderClassification, EvalDatasetLoader, \
    train_dataset_classification, process_a_batch_classification, eval_dataset

from models.eval_concept_embedder import EvalConceptEmbedderViaRankingAndClassification
from models.prompts_concept_embeddings import load_datasets


def initialization():
    log_file = os.path.join(configuration.logging_folder, os.path.splitext(os.path.basename(__file__))[0] + '.log')
    logging.basicConfig(format='%(levelname)s %(asctime)s:  %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p', filename=log_file, filemode='w', level=logging.INFO)


train_dataset_ranking, train_dataset_classification, eval_dataset = None, None, None

def load_train_eval_datasets():
    global train_dataset_ranking, train_dataset_classification, eval_dataset
    train_dataset_classification, train_dataset_ranking, eval_dataset = load_datasets()


class LinearMapping(ConceptEmbedderViaRankingAndClassification):

    def __init__(self):
        super().__init__()

    def init_trainer(self):
        self.training_data_loader_classification = TrainDatasetLoaderClassification(train_dataset_classification, self.tokenizer)
        self.training_data_loader_classification.process()

    def init_evaluator(self):
        self.eval_data_loader = EvalDatasetLoader(eval_dataset, self.tokenizer)
        self.eval_data_loader.process()

    def training(self):
        num_of_prompts_in_a_datapoint = 12
        num_of_entities_in_a_datapoint = 7
        self.model.eval()

        all_centroids = []
        all_prototypes = []
        counter = 0
        with torch.no_grad():
            for batch in self.training_data_loader_classification.batches:
            # for batch in self.training_data_loader_classification.batches[:configuration.max_steps]:
                counter += 1
                logging.info(f"Processing batch no. {counter}")
                batch = process_a_batch_classification(batch)
                list_of_input_ids = batch['list_of_input_ids'].to(self.device)
                list_of_attention_masks = batch['list_of_attention_masks'].to(self.device)
                num_of_prompts_in_a_batch = list_of_input_ids.size(0)
                assert num_of_prompts_in_a_batch % num_of_prompts_in_a_datapoint == 0, f"Number of prompts in a batch should be multiple of number of prompts in a datapoint"
                num_of_datapoints_in_a_batch = num_of_prompts_in_a_batch // num_of_prompts_in_a_datapoint

                outputs = self.model(input_ids=list_of_input_ids, attention_mask=list_of_attention_masks)
                last_hidden_state = outputs.hidden_states[self.which_hidden_state]
                last_token_embedding = last_hidden_state[:, -1, :]
                last_token_embedding = F.normalize(last_token_embedding, p=2, dim=1)
                embedding = last_token_embedding.view(num_of_datapoints_in_a_batch, num_of_prompts_in_a_datapoint, -1)

                centroids = embedding[:, :num_of_entities_in_a_datapoint, :].mean(dim=1)
                centroids = F.normalize(centroids, p=2, dim=1)
                prototypes = embedding[:, num_of_entities_in_a_datapoint, :]

                all_centroids.append(centroids)
                all_prototypes.append(prototypes)
        logging.info("Finished computing all centroids and prototypes for linear mapping training")
        all_centroids = torch.cat(all_centroids, dim=0)
        all_prototypes = torch.cat(all_prototypes, dim=0)

        P = all_prototypes.to(self.device).to(torch.float64)
        C = all_centroids.to(self.device).to(torch.float64)
        M = C.transpose(0, 1) @ P
        U, S, Vh = torch.linalg.svd(M, full_matrices=False)
        W = Vh.transpose(0, 1) @ U.transpose(0, 1)
        if torch.linalg.det(W) < 0:
            Vh[-1, :] *= -1
            W = Vh.transpose(0, 1) @ U.transpose(0, 1)
        W = W.to(torch.float32)
        self.linear_mapping = W
        with torch.no_grad():
            I_approx = W.transpose(0, 1) @ W
            ortho_err = torch.linalg.matrix_norm(I_approx - torch.eye(W.size(0), device=W.device, dtype=W.dtype))
            logging.info(f"Linear mapping learned. W shape={tuple(W.shape)}, "
                         f"orthogonality error (Fro)={float(ortho_err):.3e}")
    
        linear_mapping_path = os.path.join(configuration.learned_models, self.base_model_id, 'linear_mapping.pth')
        os.makedirs(os.path.dirname(linear_mapping_path), exist_ok=True)
        torch.save(self.linear_mapping, linear_mapping_path)
        logging.info(f"Saved linear mapping to {linear_mapping_path}")

    def load_linear_mapping(self):
        linear_mapping_path = os.path.join(configuration.learned_models, self.base_model_id, 'linear_mapping.pth')
        if not os.path.exists(linear_mapping_path):
            raise FileNotFoundError(f"Linear mapping file not found: {linear_mapping_path}")
        self.linear_mapping = torch.load(linear_mapping_path, map_location=self.device)
        logging.info(f"Loaded linear mapping from {linear_mapping_path}")


class EvalLinearMapping(EvalConceptEmbedderViaRankingAndClassification):

    def __init__(self):
        super().__init__()

    def init(self):
        self.linear_maper = LinearMapping()
        self.linear_maper.init_model()
        self.linear_maper.model.eval()
        self.linear_maper.ft_model = self.linear_maper.model
        self.linear_maper.load_linear_mapping()
        self.linear_mapping = self.linear_maper.linear_mapping.to(self.linear_maper.device)

    def evaluate(self, datapoints_coverted_to_prompts):
        with torch.no_grad():
            entity1_prompts = []
            entity2_prompts = []
            property_prompts = []
            for entity1_prompt, entity2_prompt, property_prompt in datapoints_coverted_to_prompts:
                entity1_prompts.append(entity1_prompt)
                entity2_prompts.append(entity2_prompt)
                property_prompts.append(property_prompt)
            all_prompts = entity1_prompts + entity2_prompts + property_prompts
            all_embeddings = self.linear_maper.get_embeddings_batch(all_prompts)
            entity1_embeddings = all_embeddings[:len(entity1_prompts)]
            entity2_embeddings = all_embeddings[len(entity1_prompts):len(entity1_prompts) + len(entity2_prompts)]
            property_embeddings = all_embeddings[len(entity1_prompts) + len(entity2_prompts):]
            property_embeddings_mapped = F.normalize(property_embeddings @ self.linear_mapping, p=2, dim=1)
            logits = torch.sum((entity1_embeddings - entity2_embeddings) * property_embeddings_mapped, dim=1)
            prediction = (logits > 0).long()
            return prediction


if __name__ == '__main__':
    configuration.base_model_id = "meta-llama/Meta-Llama-3-8B"
    # configuration.base_model_id = "facebook/bart-base"

    initialization()
    configuration.training_dataset_classification = f'/home/{configuration.username}/research/conceptual-spaces/data/augmented_list_of_entities_and_negatives.jsonl'
    load_train_eval_datasets()
    
    obj = LinearMapping()
    obj.init_model()
    obj.init_trainer()
    obj.training()
