from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from huggingface_hub import snapshot_download
import os

# https://huggingface.co/docs/transformers/installation
# https://pytorch.org/get-started/previous-versions/

HUGGING_FACE_API_KEY = os.environ.get("HUGGING_FACE_API_KEY")
model_id = "lmsys/fastchat-t5-3b-v1.0"

model_dir = snapshot_download(repo_id=model_id, token=HUGGING_FACE_API_KEY, local_dir="fastchat-t5-3b-v1.0")

# Force slow tokenizer (pure Python)
tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False, device_map="cpu")
model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)

pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_length=1000)
print(pipe("What are competitors to Apache Kafka?")[0]["generated_text"])
#print(pipe("1+1 is equal to ?")[0]["generated_text"])
