from brief_news.data.news_links import get_urls, get_urls_for_categories, get_headlines_for_categories, get_list_of_API_news_providers
from brief_news.data.scraper import General_scraper, CNN_scraper

import pandas as pd

# def getting_url_category(category, country):
#     url_list = get_headlines_for_categories(category, country)
#     return url_list

def get_articles(keyword):

    urls = get_urls(keyword)
    articles = [General_scraper(url) for url in urls]
    df = pd.DataFrame(articles)

    return df

test = get_articles('business')
print(test)

if __name__ == '__main__':
    print('ok')
    #get_articles('business', 'us')
