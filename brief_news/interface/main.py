import pandas as pd
from colorama import Fore, Style

from brief_news.data.news_links import get_urls, get_urls_for_categories, get_headlines_for_categories, get_list_of_API_news_providers
from brief_news.data.scraper import General_scraper, CNN_scraper

from brief_news.ml_logic.transformer_model import summary_bart_large



def get_articles(keyword: str, limit=10) -> pd.DataFrame:
    """
    This function uses the list of urls from the API and scrape the content from articles
    """
    urls = get_urls(keyword, limit)

    if len(urls) > 0:
        articles = [CNN_scraper(url) for url in urls ]
        df = pd.DataFrame(articles)

        df.dropna(subset=['article'], inplace=True)

        return df

    print(Fore.BLUE + "\nThere are no articles on this subject." + Style.RESET_ALL)
    return None


def transfomer_summaries(keyword: str) -> pd.DataFrame:
    """
    This function returns summaries of extracted articles
    """

    df_articles = get_articles(keyword)

    if df_articles != None:
        summary = summary_bart_large(df_articles)

        return summary

    print(Fore.BLUE + "\nThere are no summaries." + Style.RESET_ALL)

    return None


if __name__ == '__main__':
    print('ok')
    # get_articles('business', 'us')
    #df = get_articles('sports')
    df = transfomer_summaries('sports')
    print(df)
