import os
import logging


import pickle
import numpy as np
import plotly.graph_objs as go
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from analysis.ranking_analysis import ConceptualSpaceRanker
from models.eval_concept_embedder import EvalPreTrainedModel
from models.eval_concept_embedder_pretrained import EvalPreTrainedEmbeddingModel
from adjustText import adjust_text

from conf import configuration


# list_of_properties = ["sweet food", "very sweet food", "sour food", "very sour food", "salty food", "very salty food", "umami food", "very umami food", "fatty food", "very fatty food", "bitter food", "very bitter food", "food"]


# list_of_properties = ["very sweet food", "very sour food", "very salty food", "very umami food", "very fatty food", "very bitter food", "food item"]



# list_of_entities = ["banana", "apple", "lemon", "cabbage", "cake", "cheese", "tomato", "cucumber", "lemon", "vinegar", "avocado", "walnut", "salmon", "dried mushroom", "soy sauce", "parmesan cheese", "dark chocolate", "coffee", "bitter gourd"]

# list_of_entities = ["Honey", "Dates", "Grapes", "Milk", "Cucumber", "Lemon", "Vinegar", "Yogurt", "Tomato", "Banana", "Soy Sauce", "Pretzels", "Cheese", "Bread", "Apple", "Parmesan Cheese", "Shiitake Mushroom", "Ripe Tomato", "Green Peas", "Butter", "Avocado", "Salmon", "Chicken Breast", "Lettuce", "Cocoa Powder", "Coffee Beans", "Dark Chocolate", "Kale", "Green Tea", "Chocolate"]


# list_of_entities = ["Honey", "Dates", "Milk", "Cucumber", "Lemon", "Tomato", "Banana", "Soy Sauce", "Cheese", "Bread", "Apple", "Parmesan Cheese", "Shiitake Mushroom", "Ripe Tomato", "Green Peas", "Butter", "Avocado", "Salmon", "Chicken Breast", "Cocoa Powder", "Coffee Beans", "Dark Chocolate", "Green Tea", "Chocolate"]



list_of_properties = ["very sweet food", "very sour food", "very salty food", "very umami food", "very fatty food", "very bitter food", "food", "sweet food"]


list_of_entities = ["Cotton Candy", "Honey", "Maple Syrup", "Dates", "Milk Chocolate", "Mango", "Banana", "Sweet Corn", "Carrot", "Whole Wheat Bread", "Vinegar", "Soy Sauce", "Cheese", "Parmesan Cheese", "Salmon", "Dark Chocolate", "Chocolate", "Butter"]




def initialization():
    log_file = os.path.join(configuration.logging_folder, os.path.splitext(os.path.basename(__file__))[0] + '.log')
    logging.basicConfig(format='%(levelname)s %(asctime)s:  %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p', filename=log_file, filemode='w', level=logging.INFO)


class principal_component_analysis:
    def __init__(self):
        self.ranker = ConceptualSpaceRanker()
        self.model = None
        # self.model_name = 'pre_trained_LLM'
        # self.model_name = 'pre_trained_embedding_model'
        self.model_name = 'finetuned_LLM'
    
    def init(self):
        if self.model_name == 'pre_trained_LLM':
            self.model = EvalPreTrainedModel()
        elif self.model_name == 'finetuned_LLM':
            self.ranker.init()
            self.model = self.ranker.finetuned_model
        elif self.model_name == 'pre_trained_embedding_model':
            self.model = EvalPreTrainedEmbeddingModel()
    
        

    def get_embeddings(self, list_of_properties=list_of_properties, list_of_entities=list_of_entities):
        self.init()
        prompts = []
        for property in list_of_properties:
            prompts.append(self.ranker.get_property_prompt(property))
        for entity in list_of_entities:
            prompts.append(self.ranker.get_entity_prompt(entity))
        embeddings = self.model.get_concept_embedding(prompts)
        concept_embeddings = {}
        for property in list_of_properties:
            concept_embeddings[property] = embeddings.pop(0)
        for entity in list_of_entities:
            concept_embeddings[entity] = embeddings.pop(0)
        with open(f"results/concept_embeddings_{self.model_name}.pkl", "wb") as f:
            pickle.dump(concept_embeddings, f)
        return concept_embeddings
    
    def pca(self):
        # Load concept embeddings from file
        with open(f"results/concept_embeddings_{self.model_name}.pkl", "rb") as f:
            concept_embeddings = pickle.load(f)
        # concept_embeddings = self.get_embeddings()
        labels = list(concept_embeddings.keys())
        vectors = np.array([concept_embeddings[label] for label in labels])

        # Run PCA to reduce to 2D
        pca_model = PCA(n_components=2)
        reduced = pca_model.fit_transform(vectors)

        # Separate properties and entities for plotting
        properties = set(list_of_properties)
        entities = set(list_of_entities)

        # for i, label in enumerate(labels):
        #     x, y = reduced[i]
        #     if label in properties:
        #         plt.scatter(x, y, marker='x', color='red')
        #         plt.text(x + 0.01, y + 0.01, label, fontsize=24, color='red')
        #     elif label in entities:
        #         plt.scatter(x, y, marker='o', color='blue')
        #         plt.text(x + 0.01, y + 0.01, label, fontsize=24, color='blue')

        # plt.title("")
        # plt.xlabel("PC1")
        # plt.ylabel("PC2")
        # plt.grid(True)
        # plt.show()
        # plt.savefig("pca_projection.png")
        # plt.close()
        # plt.show()



        texts = []
        for i, label in enumerate(labels):
            x, y = reduced[i]
            if label in properties:
                plt.scatter(x, y, marker='x', color='red', s=100)
                texts.append(plt.text(x, y, label, fontsize=18, color='red'))
            elif label in entities:
                plt.scatter(x, y, marker='o', color='blue', s=100)
                texts.append(plt.text(x, y, label, fontsize=18, color='blue'))

        plt.title("")

        plt.xlabel("PC1", fontsize=18)   
        plt.ylabel("PC2", fontsize=18)  

        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)

        plt.grid(True)


        # Adjust text positions to minimize overlaps
        adjust_text(texts, arrowprops=dict(arrowstyle="->", color='grey', lw=0.5))


        plt.savefig("pca_projection.png")
        plt.show()
        plt.close()


if __name__ == "__main__":
    initialization()
    pca = principal_component_analysis()
    if configuration.username == "nitesh":
        pca.pca()
    elif configuration.username == "niteshkumar":
        pca.get_embeddings()
