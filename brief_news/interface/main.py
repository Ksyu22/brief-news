import pandas as pd

from brief_news.data.news_links import get_urls, get_urls_for_categories, get_headlines_for_categories, get_list_of_API_news_providers
from brief_news.data.scraper import General_scraper, CNN_scraper

from brief_news.ml_logic.transformer_model import summary_bart_large



def get_articles(keyword: str, limit=10) -> pd.DataFrame:
    """
    This function uses the list of urls from the API and scrape the content from articles
    """
    urls = get_urls(keyword, limit)
    articles = [CNN_scraper(url) for url in urls]
    df = pd.DataFrame(articles)

    return df


def transfomer_summaries(keyword: str) -> pd.DataFrame:
    """
    This function returns summaries of extracted articles
    """

    df_articles = get_articles(keyword)
    summary = summary_bart_large(df_articles)

    return summary


if __name__ == '__main__':
    print('ok')
    # get_articles('business', 'us')
    #df = get_articles('sports')
    df = transfomer_summaries('sports')
    print(df)
