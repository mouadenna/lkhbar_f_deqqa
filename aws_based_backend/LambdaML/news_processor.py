import os
import sys
import json

sys.path.append('/mnt/test/libraries')

from typing import Dict, List
import nltk
from nltk.corpus import stopwords
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import logging
from categories import categories
import boto3
from datetime import datetime
from spacy.util import load_model_from_path
from pathlib import Path



# Initialize the Lambda client
lambda_client = boto3.client('lambda', region_name='us-east-1')





class AdvancedNewsProcessor:
    def __init__(
        self,
        similarity_threshold: float = 0.5,
        min_cluster_size: int = 1,
    ):
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        

        nltk.data.path.append("nltk_data")
        


        
        model_path = Path('/mnt/test/spacyModel/fr_core_news_sm/fr_core_news_sm-3.7.0')
        self.nlp = load_model_from_path(model_path)

        # Initialize clustering parameters
        self.similarity_threshold = similarity_threshold
        self.min_cluster_size = min_cluster_size


        # Enhanced stopwords
        self.french_stopwords = set(stopwords.words('french'))
        self.french_stopwords.update(['a', 'les', 'le', 'la', 'dans', 'sur', 'pour'])

        # Initialize vectorizers
        self.tfidf = TfidfVectorizer(
            stop_words=list(self.french_stopwords),
            ngram_range=(1, 2),
            max_features=10000
        )

        # Category keywords
        self.category_keywords = categories
        

        self.similarity_matrix = None
        self.article_indices = {}  
        self.embeddings=None

    def process_articles(self, articles: List[Dict]) -> Dict:
        """Process articles with clustering and advanced summarization"""
        # Group articles by category
        categorized_articles = {category: [] for category in self.category_keywords}
        for article in articles:
            text = f"{article['title']}\n{article['body']}"
            category = self.assign_category(text)
            categorized_articles[category].append(article)

        processed_data = {}
        for category, category_articles in categorized_articles.items():
            if not category_articles:
                continue


            
            # Cluster articles within category
            clusters = self.cluster_articles(category_articles, category) 
            # Generate summaries for each cluster
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
        """Assign category based on keyword matching and NLP analysis"""
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
    def analyze_similarity(self, articles: List[Dict]) -> Dict:
        """
        Analyze similarities between articles and return detailed metrics
        
        Returns:
            Dict containing similarity matrix, pairwise comparisons, and statistics
        """
        texts = [f"{art['title']} {art['body']}" for art in articles]
        titles = [art['title'] for art in articles]
        
        
        self.logger.info("Generating embeddings for similarity analysis...")
        
        
        payload = {
            "sentences": texts
            }

        # Invoke the Lambda function
        response = lambda_client.invoke(
            FunctionName='arn:aws:lambda:us-east-1:---------:function:cosine_similarity:1',
            Payload=json.dumps(payload)  
            )


        response_payload = response['Payload'].read().decode('utf-8')  
        response_data = json.loads(response_payload)  
        embeddings = np.array(json.loads(response_data["body"])["similarity_matrix"])

        # Calculate similarity matrix
        similarity_matrix = np.clip(cosine_similarity(embeddings),0,1)
        self.embeddings=embeddings
        
        # Store for later use
        self.similarity_matrix = similarity_matrix
        self.article_indices = {i: title for i, title in enumerate(titles)}
        

        
        return {
            'similarity_matrix': similarity_matrix,
        }
    
    


    def get_similar_articles(self, article_idx: int, threshold: float = 0.5) -> List[Dict]:
        """Get articles similar to a specific article"""
        if self.similarity_matrix is None:
            raise ValueError("No similarity matrix available. Run analyze_similarity first.")
        
        similarities = self.similarity_matrix[article_idx]
        similar_indices = np.where(similarities >= threshold)[0]
        
        similar_articles = []
        for idx in similar_indices:
            if idx != article_idx:
                similar_articles.append({
                    'title': self.article_indices[idx],
                    'similarity_score': similarities[idx]
                })
        
        return sorted(similar_articles, key=lambda x: x['similarity_score'], reverse=True)



    def find_optimal_eps(self, embeddings: np.ndarray, min_samples: int) -> float:
        """Find optimal eps parameter for DBSCAN using silhouette score"""
        distances = 1 - cosine_similarity(embeddings)
        
        eps_range = np.linspace(0.1, 0.9, 9)
        best_score = -1
        best_eps = 0.5
        
        for eps in eps_range:
            clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')
            labels = clustering.fit_predict(distances)
            
            if len(set(labels)) <= 1:
                continue
                
            score = silhouette_score(distances, labels, metric='precomputed')
            if score > best_score:
                best_score = score
                best_eps = eps
        
        return best_eps

    def cluster_articles(self, articles: List[Dict], category: str = None) -> Dict:
        """Enhanced clustering with detailed similarity analysis"""
        if len(articles) < 2:
            return {'clusters': {0:{'articles':articles}}, 'similarity_analysis': None}
            
        
        # Perform similarity analysis
        similarity_analysis = self.analyze_similarity(articles)
        
        # Use the similarity matrix for clustering
        distance_matrix = 1 - similarity_analysis['similarity_matrix']

        #eps=self.find_optimal_eps(self.embeddings,self.min_cluster_size)
        #print("found",eps,"\n","set",self.similarity_threshold)
        # Perform clustering
        clustering = DBSCAN(
            eps=1 - self.similarity_threshold,
            min_samples=self.min_cluster_size,
            metric='precomputed',
            n_jobs=-1
        )
        print(distance_matrix.min())
        labels = clustering.fit_predict((distance_matrix))
        
        # Group articles by cluster with similarity information
        clustered_articles = {}
        for idx, cluster_id in enumerate(labels):
            if cluster_id == -1:
                new_cluster_id = max(clustered_articles.keys()) + 1 if clustered_articles else 0
                cluster_id = new_cluster_id
                
            if cluster_id not in clustered_articles:
                clustered_articles[cluster_id] = {
                    'articles': [],
                    'internal_similarities': [],
                    'center_article': None,
                    'topics': None
                }
            
            clustered_articles[cluster_id]['articles'].append(articles[idx])
            
            # Calculate similarities within cluster
            cluster_indices = np.where(labels == cluster_id)[0]
            if len(cluster_indices) > 1:
                cluster_similarities = similarity_analysis['similarity_matrix'][cluster_indices][:, cluster_indices]
                avg_similarity = (cluster_similarities.sum() - len(cluster_indices)) / (len(cluster_indices) * (len(cluster_indices) - 1))
                clustered_articles[cluster_id]['internal_similarities'].append(avg_similarity)
        
        # Calculate cluster metrics and find central articles
        for cluster_id, cluster in clustered_articles.items():
            if len(cluster['articles']) > 1:
                # Find the article most similar to others in the cluster
                cluster_indices = [i for i, art in enumerate(articles) if art in cluster['articles']]
                similarities_sum = similarity_analysis['similarity_matrix'][cluster_indices][:, cluster_indices].sum(axis=1)
                center_idx = cluster_indices[np.argmax(similarities_sum)]
                cluster['center_article'] = articles[center_idx]
                
                # Get topics for the cluster
                cluster['topics'] = self.get_cluster_topics(cluster['articles'])
        
        return {
            'clusters': clustered_articles,
            'similarity_analysis': similarity_analysis
        }


    def get_cluster_topics(self, cluster: List[Dict], top_n: int = 5) -> List[str]:
        """Extract representative topics for a cluster using TF-IDF"""
        texts = [f"{art['title']} {art['body']}" for art in cluster]
        
        # Fit TF-IDF on the cluster texts
        tfidf_matrix = self.tfidf.fit_transform(texts)
        
        # Get average TF-IDF scores across all documents
        avg_scores = tfidf_matrix.mean(axis=0).A1
        
        # Get top words
        top_indices = avg_scores.argsort()[-top_n:][::-1]
        top_words = [self.tfidf.get_feature_names_out()[i] for i in top_indices]
        
        return top_words

    def extract_keywords(self, texts: List[str], top_n: int = 5) -> List[str]:
        """
        Enhanced keyword extraction using TF-IDF and position weighting
        """
        if not texts:
            return []

        # Combine texts
        combined_text = ' '.join(texts)
        doc = self.nlp(combined_text)

        # Get word counts with position weighting
        words = []
        weights = []
        for i, token in enumerate(doc):
            if not token.is_stop and token.is_alpha and len(token.text) > 2:
                position_weight = 1 / (1 + i/len(doc))  # Higher weight for words appearing earlier
                words.append(token.text.lower())
                weights.append(position_weight)

        # Combine with TF-IDF scores
        tfidf_matrix = self.tfidf.fit_transform([combined_text])
        tfidf_words = self.tfidf.get_feature_names_out()
        tfidf_scores = tfidf_matrix.toarray()[0]

        # Combine scores
        word_scores = {}
        for word, weight in zip(words, weights):
            if word in tfidf_words:
                tfidf_idx = np.where(tfidf_words == word)[0][0]
                word_scores[word] = weight * tfidf_scores[tfidf_idx]

        # Sort and return top keywords
        sorted_words = sorted(word_scores.items(), key=lambda x: x[1], reverse=True)
        return [word for word, score in sorted_words[:top_n]]

    def summarize_cluster(self, cluster: List[Dict]) -> Dict:
        """
        Enhanced cluster summarization with additional metrics
        """

        def parse_date(date_str):
            try:
                # Attempt to parse the ISO 8601 format (including time if present)
                return datetime.fromisoformat(date_str)
            except ValueError:
                raise ValueError(f"Date '{date_str}' does not match expected formats.")
                
        # Extract dates for temporal analysis
        
        dates = [art['publication_date']
                for art in cluster 
                if 'publication_date' in art]
                

        # Calculate temporal span
        temporal_span = None
        dates = [parse_date(date_str) if isinstance(date_str, str) else date_str for date_str in dates]

        if dates:
            temporal_span = {
                'start': min(dates).strftime('%d/%m/%Y'),
                'end': max(dates).strftime('%d/%m/%Y'),
                'duration_days': (max(dates) - min(dates)).days + 1
            }

                
        # Generate summary
        combined_text = ' '.join([f"{art['title']} {art['body']}" for art in cluster])
        summary = self.generate_summary(combined_text)

        # Extract sources info
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

        # Extract keywords
        keywords = self.extract_keywords([f"{art['title']} {art['body']}" for art in cluster])

        # Calculate cluster metrics
        metrics = {
            'article_count': len(cluster),
            'unique_sources': len(set(info['source'] for info in sources_info)),
            'temporal_span': temporal_span,
            'average_article_length': np.mean([len(art.get('body', '')) for art in cluster])
        }

        return {
            'summary': summary,
            'sources_info': sources_info,
            'keywords': keywords,
            'metrics': metrics
        }

    def generate_summary(self, text: str) -> str:
        """
        Generate summary using the transformer model with improved settings
        """

        payload = {
            "text": text
            }
        # Invoke the Lambda function
        response = lambda_client.invoke(
            FunctionName='arn:aws:lambda:us-east-1:---------:function:summarize_function',
            InvocationType='RequestResponse',
            Payload=json.dumps(payload)  
        )


        response_payload = response['Payload'].read().decode('utf-8')  
        response_data = json.loads(response_payload)  

        return json.loads(response_data["body"])["summary"]




def lambda_handler(event, context):
    processor = AdvancedNewsProcessor()
    articles = event['articles']
    processed_data = processor.process_articles(articles)

    s3_key = f"processed_data/{datetime.now().date().isoformat()}.json"  # Generate a unique key for the file
    s3.put_object(
        Bucket='processednewsbucket',
        Key=s3_key,
        Body=json.dumps(processed_data),  # Convert to JSON string
        ContentType='application/json'
    )
    return {
        'statusCode': 200,
        'body': json.dumps({'message': 'Data processed and saved to S3'})
    }
