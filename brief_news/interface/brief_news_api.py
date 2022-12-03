from fastapi import FastAPI
from brief_news.data.news_links import get_urls, get_urls_for_categories, get_headlines_for_categories, get_list_of_API_news_providers
from brief_news.data.scraper import General_scraper, CNN_scraper

from brief_news.ml_logic.transformer_model import summary_t5_small

import pandas as pd

app = FastAPI()

@app.get('/')
def index():
    return{'ok': True}

@app.get('/articles')
def get_articles_api(keyword: str, limit=1)-> dict:
    """
    This function uses the list of urls from the API and scrape the content from articles
    Used for testing API endpoints - not used for creating summaries
    """
    urls = get_urls(keyword, int(limit))
    articles = [General_scraper(url) for url in urls]
    df = pd.DataFrame(articles)

    return df.to_dict('records')

def get_articles(keyword: str, limit=1) -> pd.DataFrame:
    """
    This function uses the list of urls from the API and scrape the content from articles
    Used for generating summaries
    """
    urls = get_urls(keyword, int(limit))
    articles = [General_scraper(url) for url in urls]
    df = pd.DataFrame(articles)

    return df

def transfomer_summaries(data_frame: pd.DataFrame) -> pd.DataFrame:
    """
    This function returns summaries of extracted articles
    """
    summary = summary_t5_small(data_frame)

    return summary

@app.get('/summarize')
def summarize(keyword: str, limit=1) -> dict:
    """
    This function fetch an article from NewsAPI, scrape the web and summarize it
    using the transformer/model
    """
    df = get_articles(keyword, int(limit))
    df = transfomer_summaries(df)

    return df.to_dict('records')
