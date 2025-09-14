import os
import logging
import json
import unicodedata
from conf import configuration
from utils.new_openai_api import get_gpt_response

gpt_model = "gpt-4.1"
INPUT_PATH = f"/home/{configuration.username}/research/conceptual-spaces/data/list_of_entities_and_negatives_temp.jsonl"
OUTPUT_PATH = f"/home/{configuration.username}/research/conceptual-spaces/data/augmented_list_of_entities_and_negatives.jsonl"


INSTRUCTION_PROMPT = """\
You are given a datapoint in JSONL format. Each datapoint has the following fields:

1) "entity type": the type of entity (e.g., "river")
2) "property": a descriptive property of that entity type (e.g., "long")
3) "examples": a list of exactly seven valid examples that match both the property and the entity type
4) "negatives": a list of four items that do not match the property. The first three are still related to the same entity type, but with incorrect or contrasting properties. The last negative is unrelated to the entity type at all.

Here is an example:

"""

TASK_PROMPT = """\

Your task is to generate 10 new datapoints in the same format, using the same entity type as the above example but different properties. Ensure that,

1) Each "examples" field contains exactly seven correct examples.
2) Each "negatives" field contains three incorrect river-related negatives and one unrelated negative.
3) Try to keep the properties concise (1-3 words).
4) Keep the output strictly in JSONL format (one JSON object per line, no extra text).
5) Output ONLY raw JSONL. Do NOT wrap in code fences. No prose.
"""

def initialization():
    log_file = os.path.join(configuration.logging_folder,
                            os.path.splitext(os.path.basename(__file__))[0] + '.log')
    logging.basicConfig(format='%(levelname)s %(asctime)s:  %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p', filename=log_file, filemode='w', level=logging.INFO)

def _normalize_obj(obj):
    """Recursively normalize all strings to NFC (composed) form."""
    if isinstance(obj, dict):
        return { _normalize_obj(k): _normalize_obj(v) for k, v in obj.items() }
    if isinstance(obj, list):
        return [ _normalize_obj(x) for x in obj ]
    if isinstance(obj, str):
        return unicodedata.normalize("NFC", obj)
    return obj

def write_to_file(seed, response):
    with open(OUTPUT_PATH, "a", encoding="utf-8") as f:
        lines = response.strip().split('\n')
        for line in lines:
            line = line.strip()
            if line:
                try:
                    json_obj = json.loads(line)
                    json_obj = _normalize_obj(json_obj)
                    f.write(json.dumps(json_obj, ensure_ascii=False) + "\n")
                except json.JSONDecodeError as e:
                    logging.error(f"Failed to decode JSON from line: {line}. Error: {e}")
                    print(f"Failed to decode JSON from line: {line}. Error: {e}")
        seed_obj = _normalize_obj(json.loads(seed))
        f.write(json.dumps(seed_obj, ensure_ascii=False) + "\n")

def generate_additional_examples(filepath):
    count = 0
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            prompt = INSTRUCTION_PROMPT + line + '\n' + TASK_PROMPT
            response = get_gpt_response(prompt, model=gpt_model)
            write_to_file(line, response)
            count += 1
            logging.info(f"Processed {count} entries.")
            print(f"Processed {count} entries.")


if __name__ == '__main__':
    initialization()
    generate_additional_examples(INPUT_PATH)
