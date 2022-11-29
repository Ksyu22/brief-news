from brief_news.data.news_links import get_urls, get_urls_for_categories, get_headlines_for_categories, get_list_of_API_news_providers
from brief_news.data.scraper import General_scraper, CNN_scraper

from brief_news.ml_logic.transformer_model import summary_t5_small
import pandas as pd


def get_articles(keyword: str, limit=1) -> pd.DataFrame:
    """
    This function uses the list of urls from the API and scrape the content from articles
    """
    urls = get_urls(keyword, limit)
    articles = [General_scraper(url) for url in urls]
    df = pd.DataFrame(articles)

    return df


def transfomer_summaries(data_frame: pd.DataFrame):
    """
    This function returns summaries of extracted articles
    """
    data_point = data_frame

    summary = summary_t5_small(data_point)

    return summary


test_df = get_articles('business')
sum = transfomer_summaries(test_df)
print(sum)
print(type(sum))


if __name__ == '__main__':
    print('ok')
    # get_articles('business', 'us')
    df = get_articles('business')
    df = transfomer_summaries(df)
    print(df)
