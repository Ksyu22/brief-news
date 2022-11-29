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
def get_articles(keyword, limit=1):
    """
    This function uses the list of urls from the API and scrape the content from articles
    """
    urls = get_urls(keyword, int(limit))
    articles = [General_scraper(url) for url in urls]
    df = pd.DataFrame(articles)

    return df.to_dict('records')

def transfomer_summaries(data_frame):
    """
    This function returns summaries of extracted articles
    """
    data_point = data_frame['article']

    summary = summary_t5_small(data_point)

    return summary

@app.get('/summarize')
def summarize(keyword, limit=1):
    """
    This function fetch an article from NewsAPI, scrape the web and summarize it
    using the transformer/model
    """
    df = get_articles(keyword, int(limit))
    summary = transfomer_summaries(df)

    return summary[0]
