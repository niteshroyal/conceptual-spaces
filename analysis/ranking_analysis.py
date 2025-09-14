import os
import torch
import logging
import numpy as np
import pandas as pd


from scipy.stats import spearmanr

from conf import configuration
from models.eval_concept_embedder import EvalConceptEmbedderViaRankingAndClassification
from models.eval_concept_embedder_pretrained import EvalConceptEmbedderViaRankingAndClassificationPretrained
from models.prompts_concept_embeddings import get_prompt

datasets = ["food item", "rock"]

def initialization():
    log_file = os.path.join(configuration.logging_folder,
                            os.path.splitext(os.path.basename(__file__))[0] + '.log')
    logging.basicConfig(format='%(levelname)s %(asctime)s:  %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p', filename=log_file, filemode='w', level=logging.INFO)

# used_data_files_metadata = []
# for data in data_files_metadata:
#     if data["entity_type"] in datasets:
#         used_data_files_metadata.append(data)

used_data_files_metadata = [
    {
        "file_path": os.path.join(configuration.ranking_dataset_path, "Food_taste/food_Taste.txt"),
        "entity_type": "food item",
        "property_header": "Sweet_Mean",
        "entity_header": "foodLabel",
        "property": "sweet food item",
        "high property": "very sweet taste food item",
        "low property": "mildly sweet food item",
        "ascending": False,
        "ranking_header": "score",
        "selectAll": True
    },
    {
        "file_path": os.path.join(configuration.ranking_dataset_path, "Food_taste/food_Taste.txt"),
        "entity_type": "food item",
        "property_header": "Salty_Mean",
        "entity_header": "foodLabel",
        "property": "salty food item",
        "high property": "very salty taste food item",
        "low property": "a bit salty food item",
        "ascending": False,
        "ranking_header": "score",
        "selectAll": True
    },
    {
        "file_path": os.path.join(configuration.ranking_dataset_path, "Food_taste/food_Taste.txt"),
        "entity_type": "food item",
        "property_header": "Sour_Mean",
        "entity_header": "foodLabel",
        "property": "sour food item",
        "high property": "very sour taste food item",
        "low property": "mildly sour food item",
        "ascending": False,
        "ranking_header": "score",
        "selectAll": True
    },
    {
        "file_path": os.path.join(configuration.ranking_dataset_path, "Food_taste/food_Taste.txt"),
        "entity_type": "food item",
        "property_header": "Bitter_Mean",
        "entity_header": "foodLabel",
        "property": "bitter food item",
        "high property": "very bitter taste food item",
        "low property": "mildly bitter food item",
        "ascending": False,
        "ranking_header": "score",
        "selectAll": True
    },
    {
        "file_path": os.path.join(configuration.ranking_dataset_path, "Food_taste/food_Taste.txt"),
        "entity_type": "food item",
        "property_header": "Umami_Mean",
        "entity_header": "foodLabel",
        "property": "umami food item",
        "high property": "very umami taste food item",
        "low property": "mildly umami food item",
        "ascending": False,
        "ranking_header": "score",
        "selectAll": True
    },
    {
        "file_path": os.path.join(configuration.ranking_dataset_path, "Food_taste/food_Taste.txt"),
        "entity_type": "food item",
        "property_header": "Fat_Mean",
        "entity_header": "foodLabel",
        "property": "fatty food item",
        "high property": "very fatty taste food item",
        "low property": "mildly fatty food item",
        "ascending": False,
        "ranking_header": "score",
        "selectAll": True
    },
]


def calculate_spearman_rho(original_rank, predicted_rank):
    res = spearmanr(np.array(original_rank), np.array(predicted_rank))
    rho = res.correlation
    return rho


class ConceptualSpaceRanker:
    def __init__(self):
        self.finetuned_model = None
    
    def init(self):
        self.finetuned_model = EvalConceptEmbedderViaRankingAndClassification()
        # self.finetuned_model = EvalConceptEmbedderViaRankingAndClassificationPretrained()
        self.finetuned_model.init()

    def get_entity_prompt(self, entity_name, entity_type=""):
        if entity_type == "food item":
            entity_name = "food item - " + entity_name
            # pass
        elif entity_type == "rock":
            entity_name = entity_name + " rock"
        else:
            pass
        entity_prompt = get_prompt(entity_name)
        return entity_prompt
    
    def get_property_prompt(self, property):
        property_prompt = get_prompt(property)
        return property_prompt

    def ranker(self, entities, entity_type, property):
        property_prompt = self.get_property_prompt(property)
        entities_prompt = []
        for entity in entities:
            entities_prompt.append(self.get_entity_prompt(entity["entity_name"], entity_type))
        
        embeddings = self.finetuned_model.get_concept_embedding([property_prompt] + entities_prompt)
        property_embedding = embeddings[0]
        entity_embeddings = embeddings[1:]
        
        property_embedding = torch.tensor(property_embedding)
        entity_embeddings = torch.tensor(entity_embeddings)

        scores = torch.matmul(entity_embeddings, property_embedding)
        scores = scores.tolist()
        sorted_indices = sorted(range(len(scores)), key=lambda k: scores[k], reverse=True)

        sorted_entities = [entities[i] for i in sorted_indices]

        for i, entity in enumerate(sorted_entities):
            entity["predicted_rank"] = i + 1
            entity["predicted_score"] = scores[sorted_indices[i]]
        
        original_rank = []
        predicted_rank = []

        for entity in entities:
            original_rank.append(entity["original_rank"])
            predicted_rank.append(entity["predicted_rank"])

        rho = calculate_spearman_rho(original_rank, predicted_rank)

        logging.info(f'Spearman rho for property "{property}" is {rho}')
        print(f'Spearman rho for property "{property}" is {rho}')


def read_data(file_path):
    return pd.read_csv(file_path, sep='\t')

def main(ranker_obj):
    for data_file in used_data_files_metadata:
        file_path = data_file["file_path"]
        entity_type = data_file["entity_type"]
        property_header = data_file["property_header"]
        entity_header = data_file["entity_header"]
        high_property = data_file["high property"]
        ascending = data_file["ascending"]
    
        df = read_data(file_path)
        top_entities = df

        converted = pd.to_numeric(top_entities[property_header], errors='coerce')
        num_invalid = converted.isna().sum()

        if num_invalid > 0:
            print(f"⚠️ Warning: {num_invalid} non-numeric value(s) found in '{property_header}' column. They will be treated as NaN and ignored.")

        top_entities[property_header] = converted
        top_entities = top_entities.dropna(subset=[property_header])

        top_entities = top_entities.sort_values(by=property_header, ascending=ascending)

        original_ranked_entities = []

        rank = 0
        for index, row in top_entities.iterrows():
            entity_name = row[entity_header]
            original_score = row[property_header]
            original_rank = rank + 1
            original_ranked_entities.append({
                "entity_name": entity_name,
                "original_score": original_score,
                "original_rank": original_rank,
                "predicted_rank": None,
                "predicted_score": None
            })
            rank += 1
        
        ranker_obj.ranker(original_ranked_entities, entity_type, high_property)

if __name__ == '__main__':
    initialization()
    ranker = ConceptualSpaceRanker()
    ranker.init()
    main(ranker)
    

