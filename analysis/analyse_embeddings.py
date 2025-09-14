from analysis.ranking_analysis import ConceptualSpaceRanker
from models.eval_concept_embedder import EvalPreTrainedModel


list_of_concepts = ["sweet food", "Honey", "Dates", "Milk", "Cucumber"]


class EmbeddingAnalysis:
    def __init__(self):
        self.ranker = ConceptualSpaceRanker()
        self.model = None
    
    def init(self):
        model = 'finetuned'
        if model == 'not finetuned':
            self.model = EvalPreTrainedModel()
        elif model == 'finetuned':
            self.ranker.init()
            self.model = self.ranker.finetuned_model
        

    def get_embeddings(self, list_of_concepts=list_of_concepts):
        prompts = []
        for concept in list_of_concepts:
            prompts.append(self.ranker.get_entity_prompt(concept))
        embeddings = self.model.get_concept_embedding(prompts)
        concept_embeddings = {}
        for concept in list_of_concepts:
            concept_embeddings[concept] = embeddings.pop(0)
        return concept_embeddings
    
if __name__ == "__main__":
    obj = EmbeddingAnalysis()
    obj.init()
    concept_embeddings = obj.get_embeddings()
    print(concept_embeddings)
