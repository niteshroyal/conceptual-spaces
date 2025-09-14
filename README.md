**This repository contains datasets and source code used in our paper presented in ACL 2024:** 

[Ranking Entities along Conceptual Space Dimensions with LLMs: An Analysis of Fine-Tuning Strategies](https://aclanthology.org/2024.findings-acl.474.pdf)


### Training and Test Datasets

* After executing git clone, goto the main folder containing source code. For me it is:
```commandline
cd /home/nitesh/elexir/RankingUsingLLMs
```

* All datasets used in the paper can be found in the datasets folder. Use:
```commandline
ls -l datasets/
```

* To see training datasets use:
```commandline
ls -l datasets/training
```

* To see test datasets use:
```commandline
ls -l datasets/testing
```

* Note that pointwise and pairwise models require data in different format so training files used for pointwise models are named as follows:
```commandline
ls -l datasets/training/*-train-pointwise.jsonl
```

* If you want to view the raw dataset from which the above datasets are preprocessed then use:
```commandline
ls -l datasets/raw_datasets
```


### Installation:

```commandline
conda create -n llm python=3.10.11
conda activate llm
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install peft Flask datasets matplotlib bitsandbytes protobuf evaluate scikit-learn
huggingface-cli login
```

### Finetune and Test:


* Before finetuning, please make appropriate changes in the configuration file

```commandline
vim conf/my_conf.py
```

* Then create training and validation dataset for finetuning
```commandline
# Activate the conda environment
conda activate llm


# Goto the main folder containing source code
cd /home/nitesh/elexir/RankingUsingLLMs


# Set the PYTHONPATH environment variable
export PYTHONPATH="${PYTHONPATH}:/home/nitesh/elexir/RankingUsingLLMs"
```

* To fine tune with pairwise model:
```commandline
python llm_classifier/classifier.py
```

* When finetuning is complete and we need to test:
```commandline
python llm_classifier/eval_classifier.py 
```


* To fine tune with pointwise model:
```commandline
python llm_classifier/pointwise.py
```

* When finetuning is complete and we need to test:
```commandline
python llm_classifier/eval_pointwise.py
```













