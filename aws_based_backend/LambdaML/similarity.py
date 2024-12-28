
import os
import sys
import json


# Set cache directory for Sentence Transformers and append libraries to path
os.environ["TRANSFORMERS_CACHE"] = "/tmp"
sys.path.append('/mnt/test/libraries')


import numpy as np
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity

# Set cache directory for Transformers (if needed)
os.environ["TRANSFORMERS_CACHE"] = "/tmp"

# Load the model and tokenizer globally
model_path = "/mnt/test/Models/sentence-transformers_all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModel.from_pretrained(model_path)

def calculate_cosine_similarity(sentences: list) -> dict:
    """
    Calculate embeddings and cosine similarity for a list of sentences.
    Args:
        sentences (list): List of strings to analyze.
    Returns:
        dict: Embeddings and similarity scores.
    """
    # Tokenize and encode sentences
    inputs = tokenizer(sentences, return_tensors="pt", padding=True, truncation=True)

    # Generate embeddings using the model
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).detach().numpy()

    # Compute cosine similarity matrix
    similarity_matrix = cosine_similarity(embeddings)

    # Format results
    results = {
        'sentences': sentences,
        'similarity_matrix': similarity_matrix.tolist(),
    }
    return results

def lambda_handler(event, context):
    """
    Lambda function handler to process sentences and return cosine similarity.
    Args:
        event (dict): Event input with 'sentences' key.
        context (object): Lambda context.
    Returns:
        dict: Response with similarity results.
    """
    # Get sentences from the event
    sentences = event.get('sentences', [])
    if not sentences or not isinstance(sentences, list):
        return {
            'statusCode': 400,
            'body': json.dumps({'error': 'Invalid input. Please provide a list of sentences.'})
        }

    # Process sentences to calculate similarity
    try:
        result = calculate_cosine_similarity(sentences)
        return {
            'statusCode': 200,
            'body': json.dumps(result)
        }
    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }
