

import sys
import os

# Define the EFS mount path
EFS_MOUNT_PATH = '/mnt/test/backend/'

# Add the 'libraries' folder from EFS to the Python module search path
sys.path.append(os.path.join(EFS_MOUNT_PATH, 'libraries'))#,"Models"

import json
from typing import Dict, List
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import nltk
from nltk.corpus import stopwords
import spacy
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import logging

class AdvancedNewsProcessor:
    def __init__(self, similarity_threshold=0.5, min_cluster_size=1, sentence_transformer_model='all-mpnet-base-v2'):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        self.nlp = spacy.load('fr_core_news_sm')
        
        self.similarity_threshold = similarity_threshold
        self.min_cluster_size = min_cluster_size
        
        # Load models from EFS path
        self.sentence_transformer_model_path = "/mnt/efs/sentence_transformer"
        self.sentence_transformer = SentenceTransformer(self.sentence_transformer_model_path)
        
        self.french_stopwords = set(stopwords.words('french'))
        self.french_stopwords.update(['a', 'les', 'le', 'la', 'dans', 'sur', 'pour'])
        self.tfidf = TfidfVectorizer(stop_words=list(self.french_stopwords), ngram_range=(1, 2), max_features=10000)
        self.category_keywords = {}  # Define your categories here
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Load HuggingFace model from EFS path
        self.ckpt = "/mnt/efs/huggingface_model/your-model-directory"
        self.tokenizer = AutoTokenizer.from_pretrained(self.ckpt)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.ckpt).to(self.device)
        
        self.similarity_matrix = None
        self.article_indices = {}

    def process_articles(self, articles: List[Dict]) -> Dict:
        categorized_articles = {category: [] for category in self.category_keywords}
        for article in articles:
            text = f"{article['title']}\n{article['body']}"
            category = self.assign_category(text)
            categorized_articles[category].append(article)

        processed_data = {}
        for category, category_articles in categorized_articles.items():
            if not category_articles:
                continue
            clusters = self.cluster_articles(category_articles, category)
            cluster_summaries = []
            for cluster in list(clusters["clusters"].values()):
                summary = self.summarize_cluster(cluster["articles"])
                cluster_summaries.append(summary)
            processed_data[category] = {
                'clusters': cluster_summaries,
                'total_articles': len(category_articles)
            }
        return processed_data

    def assign_category(self, text: str) -> str:
        doc = self.nlp(text.lower())
        scores = {category: 0 for category in self.category_keywords}
        for token in doc:
            if token.pos_ in ['NOUN', 'PROPN', 'VERB'] and not token.is_stop:
                word = token.text.lower()
                for category, keywords in self.category_keywords.items():
                    if word in keywords:
                        position_weight = 1 / (1 + token.i/len(doc))
                        scores[category] += position_weight
        return max(scores.items(), key=lambda x: x[1])[0]

    def summarize_cluster(self, cluster: List[Dict]) -> Dict:
        combined_text = ' '.join([f"{art['title']} {art['body']}" for art in cluster])
        summary = self.generate_summary(combined_text)
        sources_info = []
        for art in cluster:
            source = art.get('source', 'Source Inconnue')
            date = art['publication_date']
            url = art.get('url', '#')
            sources_info.append({
                'source': source,
                'date': date,
                'url': url,
                'title': art.get('title', 'Titre non disponible')
            })
        keywords = self.extract_keywords([f"{art['title']} {art['body']}" for art in cluster])
        metrics = {
            'article_count': len(cluster),
            'unique_sources': len(set(info['source'] for info in sources_info)),
            'average_article_length': np.mean([len(art.get('body', '')) for art in cluster])
        }
        return {
            'summary': summary,
            'sources_info': sources_info,
            'keywords': keywords,
            'metrics': metrics
        }

    def generate_summary(self, text: str) -> str:
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        input_ids = inputs.input_ids.to(self.device)
        attention_mask = inputs.attention_mask.to(self.device)
        output = self.model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_length=256,
            min_length=100,
            num_beams=4,
            length_penalty=0.2,
            no_repeat_ngram_size=3,
            early_stopping=True,
            do_sample=False,
            temperature=1.0,
            top_k=50,
            repetition_penalty=1.2
        )
        output = output.to('cpu')
        return self.tokenizer.decode(output[0], skip_special_tokens=True)

    def extract_keywords(self, texts: List[str], top_n: int = 5) -> List[str]:
        combined_text = ' '.join(texts)
        doc = self.nlp(combined_text)
        words = []
        weights = []
        for i, token in enumerate(doc):
            if not token.is_stop and token.is_alpha and len(token.text) > 2:
                position_weight = 1 / (1 + i/len(doc))
                words.append(token.text.lower())
                weights.append(position_weight)
        tfidf_matrix = self.tfidf.fit_transform([combined_text])
        tfidf_words = self.tfidf.get_feature_names_out()
        tfidf_scores = tfidf_matrix.toarray()[0]
        word_scores = {}
        for word, weight in zip(words, weights):
            if word in tfidf_words:
                tfidf_idx = np.where(tfidf_words == word)[0][0]
                word_scores[word] = weight * tfidf_scores[tfidf_idx]
        sorted_words = sorted(word_scores.items(), key=lambda x: x[1], reverse=True)
        return [word for word, score in sorted_words[:top_n]]

def lambda_handler(event, context):
    processor = AdvancedNewsProcessor()
    articles = event['articles']
    processed_data = processor.process_articles(articles)
    return {
        'statusCode': 200,
        'body': json.dumps(processed_data)
    }
