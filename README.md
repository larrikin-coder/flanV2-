# Fine-tuning and Deduplication of Flan-V2
- Fine-tuning and Deduping Flan-V2 dataset first task was to remove rows containing token size > 100 tokens
- Loading dataset from huggingface-cli and datasets module
- Reducing the rows with token size > 100
```py
filtered_dataset = dataset["train"].select(
    i for i in range(len(dataset["train"]))
    if len(dataset["train"][i]["messages"][0]['content']) > 100
)
```
- Deduping dataset by using cosine similarity as the intersection point.
```py
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from numpy import dot, linalg
from collections import defaultdict
import numpy as np

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(filtered_dataset['messages'].apply(lambda x: x[0]['content']))
cosine_sim_matrix = cosine_similarity(X, X)

def get_duplicates(cosine_sim_matrix, threshold=0.9):
    duplicates = set()
    for i in range(len(cosine_sim_matrix)):
        for j in range(i + 1, len(cosine_sim_matrix)):
            if cosine_sim_matrix[i, j] > threshold:
                duplicates.add(j)
    return duplicates

duplicates = get_duplicates(cosine_sim_matrix)
unique_texts = [text for idx, text in enumerate(responses) if idx not in duplicates]

deduplicated_dataset_dict = DatasetDict({
    'train': Dataset.from_dict({'responses': unique_texts})
})
```
- Dataset link for Deduped and filtered dataset : <a href="https://huggingface.co/datasets/larrikin7/shaurya-flanV2-100k-filtered">https://huggingface.co/datasets/larrikin7/shaurya-flanV2-100k-filtered</a>
<br>
- For finetuning <b>llama2-7b-chat-hf</b> was used to train the dataset on 1 epoch because of low compute resources. <b>LORA</b> (Low-Rank Adaptation for finetuning Large Models).
- Lora enabled us to have great accuracy even when the dataset was trained for 1 epochs and low compute.

```py
import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging
)
from peft import LoraConfig, PeftModel
from trl import SFTTrainer
     

model_name = "NousResearch/Llama-2-7b-chat-hf"
dataset_name = "Roudranil/Flan-v2-Filtered"
new_model = "shaurya-llama2-finetuned-w-flanV2"

lora_r = 64
lora_alpha = 16
lora_dropout = 0.1

use_4bit = True
bnb_4bit_compute_dtype = "float16"
bnb_4bit_quant_type = "nf4"
use_nested_quant = False

output_dir = "./results"
num_train_epochs = 1

fp16 = False
bf16 = False

per_device_train_batch_size = 4
per_device_eval_batch_size = 4
gradient_accumulation_steps = 1
gradient_checkpointing = True
learning_rate = 2e-4
max_grad_norm = 0.3
weight_decay = 0.001

optim = "paged_adamw_32bit"
lr_scheduler_type = "cosine"
warmup_ratio = 0.03

max_steps = -1
logging_steps = 25
save_steps = 0
group_by_length = True

max_seq_length = None
packing = False

device_map = {"":0}
```
- Other variables values were chosen based on industry standards.
