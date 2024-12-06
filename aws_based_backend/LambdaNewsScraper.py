import json
import boto3
import requests
from bs4 import BeautifulSoup
from datetime import datetime
import logging
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse, urljoin

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Lambda-specific configurations
MAX_ARTICLES = int(os.environ.get('MAX_ARTICLES', '5'))
CONCURRENT_REQUESTS = int(os.environ.get('CONCURRENT_REQUESTS', '3'))
REQUEST_TIMEOUT = int(os.environ.get('REQUEST_TIMEOUT', '5'))
S3_BUCKET = os.environ.get('S3_BUCKET', 'news-brief-raw-data')

class NewsScraper:
    def __init__(self):
        # Configure session with retries
        self.session = self._create_session()
        self.timestamp = datetime.utcnow().strftime('%Y%m%d')

    def _create_session(self):
        session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        return session

    def validate_url(self, url):
        """Validate and normalize URL"""
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except Exception:
            return False

    def parse_date(self, date_str):
        """Parse French format date string"""
        try:
            date_str = date_str.replace('Le ', '').replace(' à ', ' ').replace('h', ':')
            return datetime.strptime(date_str, '%d/%m/%Y %H:%M')
        except Exception as e:
            logger.error(f"Error parsing date {date_str}: {str(e)}")
            return datetime.utcnow()

    def scrape_article(self, article_url):
        """Scrape individual article"""
        try:
            if not self.validate_url(article_url):
                raise ValueError(f"Invalid URL: {article_url}")

            article_response = self.session.get(article_url, timeout=REQUEST_TIMEOUT)
            article_response.raise_for_status()
            article_soup = BeautifulSoup(article_response.text, 'html.parser')

            # Extract article data
            title = article_soup.find('h1')
            title = title.text.strip() if title else "No title available"

            date_elem = article_soup.find('div', class_='article-body-subheadline-date')
            formatted_date = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
            if date_elem:
                date_str = date_elem.text.strip()
                parsed_date = self.parse_date(date_str)
                formatted_date = parsed_date.strftime('%Y-%m-%d %H:%M:%S')

            paragraphs = article_soup.find_all('p', class_='default__StyledText-sc-10mj2vp-0 fSEbof body-paragraph')
            body_text = "\n".join([p.text.strip() for p in paragraphs])

            return {
                'title': title,
                'url': article_url,
                'date': formatted_date,
                'content': body_text,
                'processing_timestamp': self.timestamp
            }
        except Exception as e:
            logger.error(f"Error scraping article {article_url}: {str(e)}")
            return None

    def scrape_le360(self, base_url='https://fr.le360.ma/politique'):
        """Main scraping function with concurrent processing"""
        logger.info(f"Starting scrape of {base_url}")
        articles = []

        try:
            # Validate base URL
            if not self.validate_url(base_url):
                raise ValueError(f"Invalid base URL: {base_url}")

            # Get the main page
            response = self.session.get(base_url, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')

            # Find all article items
            article_items = soup.find_all('div', class_='article-list-item')
            logger.info(f"Found {len(article_items)} articles")

            # Extract article URLs
            article_urls = []
            for item in article_items[:MAX_ARTICLES]:
                link = item.find('a')['href']
                if not link.startswith('http'):
                    link = urljoin(base_url, link)
                article_urls.append(link)

            # Concurrent processing of articles
            with ThreadPoolExecutor(max_workers=CONCURRENT_REQUESTS) as executor:
                future_to_url = {executor.submit(self.scrape_article, url): url 
                               for url in article_urls}
                
                for future in as_completed(future_to_url):
                    url = future_to_url[future]
                    try:
                        article = future.result()
                        if article:
                            articles.append(article)
                            logger.info(f"Successfully scraped article: {url}")
                    except Exception as e:
                        logger.error(f"Error processing article {url}: {str(e)}")

        except Exception as e:
            logger.error(f"Error scraping main page: {str(e)}")

        return articles

def lambda_handler(event, context):
    """Lambda handler with improved error handling and monitoring"""
    timestamp = datetime.utcnow().strftime('%Y%m%d')
    
    # Calculate remaining time for Lambda execution
    remaining_time = context.get_remaining_time_in_millis() if context else None
    if remaining_time and remaining_time < 10000:  # Less than 10 seconds remaining
        logger.warning("Insufficient time remaining for Lambda execution")
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': 'Insufficient time remaining',
                'timestamp': timestamp
            })
        }

    try:
        logger.info("Starting Lambda execution")
        
        # Initialize scraper
        scraper = NewsScraper()
        
        # Scrape articles
        articles = scraper.scrape_le360()
        
        if not articles:
            logger.warning("No articles were scraped")
            return {
                'statusCode': 400,
                'body': json.dumps({
                    'message': 'No articles were scraped',
                    'timestamp': timestamp
                })
            }

        # Save to S3
        s3_client = boto3.client('s3')
        
        s3_client.put_object(
            Bucket=S3_BUCKET,
            Key=f'articles/{timestamp}.json',
            Body=json.dumps(articles, ensure_ascii=False),
            ContentType='application/json'
        )
        
        logger.info(f"Successfully saved {len(articles)} articles to S3")
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': f'Successfully processed {len(articles)} articles',
                'timestamp': timestamp,
                'article_count': len(articles)
            })
        }
            
    except Exception as e:
        logger.error(f"Lambda execution failed: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': str(e),
                'timestamp': timestamp
            })
        }
