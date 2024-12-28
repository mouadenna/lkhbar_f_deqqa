import os
import sys
import json


os.environ["TRANSFORMERS_CACHE"] = "/tmp"
sys.path.append('/mnt/test/libraries')


import json
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load model and tokenizer globally to reuse across invocations
ckpt = "/mnt/test/Models/models--plguillou--t5-base-fr-sum-cnndm/v1"
tokenizer = AutoTokenizer.from_pretrained(ckpt,legacy=False)
model = AutoModelForSeq2SeqLM.from_pretrained(ckpt)

def generate_summary(text: str) -> str:
    """
    Generate summary using the transformer model with improved settings.
    """
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask



    output = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_length=256,
        min_length=100,
        num_beams=4,
        length_penalty=0.2,
        no_repeat_ngram_size=3
    )
    summary = tokenizer.decode(output[0], skip_special_tokens=True)
    return summary

def lambda_handler(event, context):
    text = event.get('text', '')
    summary = generate_summary(text)
    return {
        'statusCode': 200,
        'body': json.dumps({'summary': summary})
    }
