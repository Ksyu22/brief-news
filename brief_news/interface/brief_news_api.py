from fastapi import FastAPI
from brief_news.data.news_links import get_urls, get_urls_for_categories, get_headlines_for_categories, get_list_of_API_news_providers
from brief_news.data.scraper import General_scraper, CNN_scraper
import pandas as pd

app = FastAPI()

@app.get('/')
def index():
    return{'ok': True}

@app.get('/articles')
def get_articles(keyword):

    urls = get_urls(keyword)
    articles = [General_scraper(url) for url in urls]
    df = pd.DataFrame(articles)

    return df.to_dict('records')
