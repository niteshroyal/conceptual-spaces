import os
import torch
import logging
import torch.nn.functional as F

from torch import Tensor
# from llm2vec import LLM2Vec
from transformers import AutoTokenizer, AutoModel, AutoConfig
from peft import PeftModel

from conf import configuration
from models.concept_embedder_pretrained_embeddings import ConceptEmbedderViaRankingAndClassificationPretrained
from models.concept_embedder_pretrained_embeddings_only_ranking import ConceptEmbedderViaRankingPretrained
from models.concept_embedder_pretrained_embeddings_only_classification import ConceptEmbedderViaClassificationPretrained
from models.eval_concept_embedder import EvalConceptEmbedderViaRankingAndClassification, EvalConceptEmbedderViaRanking, EvalConceptEmbedderViaClassification


def initialization():
    log_file = os.path.join(configuration.logging_folder,
                            os.path.splitext(os.path.basename(__file__))[0] + '.log')
    logging.basicConfig(format='%(levelname)s %(asctime)s:  %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p', filename=log_file, filemode='w', level=logging.INFO)

def last_token_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

class EvalConceptEmbedderViaRankingAndClassificationPretrained(EvalConceptEmbedderViaRankingAndClassification):
    
    def __init__(self):
        super().__init__()

    def init(self):
        self.finetuned_model = ConceptEmbedderViaRankingAndClassificationPretrained()
        location = self.finetuned_model.get_generator_llm()
        self.finetuned_model.init_ft_model(location)
        logging.info(f"Loading fine tuned model from {location}")


class EvalConceptEmbedderViaRankingPretrained(EvalConceptEmbedderViaRanking):

    def __init__(self):
        super().__init__()

    def init(self):
        self.finetuned_model = ConceptEmbedderViaRankingPretrained()
        location = self.finetuned_model.get_generator_llm()
        self.finetuned_model.init_ft_model(location)
        logging.info(f"Loading fine tuned model from {location}")


class EvalConceptEmbedderViaClassificationPretrained(EvalConceptEmbedderViaClassification):
    def __init__(self):
        super().__init__()

    def init(self):
        self.finetuned_model = ConceptEmbedderViaClassificationPretrained()
        location = self.finetuned_model.get_generator_llm()
        self.finetuned_model.init_ft_model(location)
        logging.info(f"Loading fine tuned model from {location}")



class PreTrainedEmbeddingModel(ConceptEmbedderViaRankingAndClassificationPretrained):

    def __init__(self):
        super().__init__()


    def get_embeddings(self, list_of_texts, model_name="LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised"):
        # model_input = self.tokenizer(
        #     list_of_texts,
        #     truncation=True,
        #     max_length=configuration.tokenizer_max_length,
        #     padding="max_length",
        #     return_tensors="pt"
        # ).to("cuda")
        # input_ids_list = self.tokenizer(list_of_texts)['input_ids']
        # for ids in input_ids_list:
        #     self.token_sizes.append(len(ids))
        # self.model.eval()
        # with torch.no_grad():
        #     outputs = self.model(**model_input)
        # last_hidden_state = outputs.hidden_states[self.which_hidden_state]
        # last_token_embedding = last_hidden_state[:, -1, :]
        # embeddings = F.normalize(last_token_embedding, p=2, dim=1)
        # return embeddings

        if model_name == "E5-Mistral-7B":
            # Tokenize the input texts
            batch_dict = self.tokenizer(list_of_texts, max_length=512, padding=True, truncation=True, return_tensors='pt')
            # batch_dict = {k: v.to(self.device) for k, v in batch_dict.items()}
            outputs = self.model(**batch_dict)
            embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
            embeddings = F.normalize(embeddings, p=2, dim=1)

            return embeddings
        else:
            embeddings = self.model.encode(list_of_texts)
            embeddings = F.normalize(embeddings, p=2, dim=1)
            return embeddings
    

    
    def init_model(self, model_name="LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised"):

        if model_name == "LLM2Vec-Llama3-8B":
            tokenizer = AutoTokenizer.from_pretrained("McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp")
            config = AutoConfig.from_pretrained(
                "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp", 
                trust_remote_code=True
            )
            model = AutoModel.from_pretrained(
                "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp",
                trust_remote_code=True,
                config=config,
                torch_dtype=torch.bfloat16,
                device_map="cuda" if torch.cuda.is_available() else "cpu",
            )

            model = PeftModel.from_pretrained(
                model,
                "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp",
            )

            self.model = LLM2Vec(model, tokenizer, pooling_mode="mean", max_length=512)
        
        elif model_name == "LLM2Vec-Mistral-7B":
            tokenizer = AutoTokenizer.from_pretrained(
                "McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp"
            )
            config = AutoConfig.from_pretrained(
                "McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp", trust_remote_code=True
            )
            model = AutoModel.from_pretrained(
                "McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp",
                trust_remote_code=True,
                config=config,
                torch_dtype=torch.bfloat16,
                device_map="cuda" if torch.cuda.is_available() else "cpu",
            )

            # Loading MNTP (Masked Next Token Prediction) model.
            model = PeftModel.from_pretrained(
                model,
                "McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp",
            )

            # Wrapper for encoding and pooling operations
            l2v = LLM2Vec(model, tokenizer, pooling_mode="mean", max_length=512)
            self.model = l2v

        elif model_name == "E5-Mistral-7B":

            tokenizer = AutoTokenizer.from_pretrained('intfloat/e5-mistral-7b-instruct')
            model = AutoModel.from_pretrained('intfloat/e5-mistral-7b-instruct')
            self.model = model #.to(self.device)
            self.tokenizer = tokenizer
            print("E5-Mistral-7B model loaded successfully.")


        elif model_name == "LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised":

            # Loading base Mistral model, along with custom code that enables bidirectional connections in decoder-only LLMs. MNTP LoRA weights are merged into the base model.
            tokenizer = AutoTokenizer.from_pretrained(
                "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp"
            )
            config = AutoConfig.from_pretrained(
                "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp", trust_remote_code=True
            )
            model = AutoModel.from_pretrained(
                "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp",
                trust_remote_code=True,
                config=config,
                torch_dtype=torch.bfloat16,
                device_map="cuda" if torch.cuda.is_available() else "cpu",
            )
            model = PeftModel.from_pretrained(
                model,
                "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp",
            )
            model = model.merge_and_unload()  # This can take several minutes on cpu

            # Loading supervised model. This loads the trained LoRA weights on top of MNTP model. Hence the final weights are -- Base model + MNTP (LoRA) + supervised (LoRA).
            model = PeftModel.from_pretrained(
                model, "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised"
            )

            # Wrapper for encoding and pooling operations
            l2v = LLM2Vec(model, tokenizer, pooling_mode="mean", max_length=128)
            self.model = l2v


        else:
            raise ValueError(f"Unknown model name: {model_name}")



class EvalPreTrainedEmbeddingModel(EvalConceptEmbedderViaRankingAndClassificationPretrained):

    def __init__(self):
        super().__init__()
        self.finetuned_model = PreTrainedEmbeddingModel()
        self.finetuned_model.init_model()