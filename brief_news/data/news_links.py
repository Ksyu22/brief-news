import numpy as np
import pandas as pd
import requests
from datetime import datetime
from dotenv import load_dotenv, find_dotenv
import os


def get_newskey():
    """
    This function will fetch your NEWS_API from the .env file in the root folder.
    Your .env file should contain a like like: NEWS_API='asdfasdfasdfsadf'
    """
    env_path = find_dotenv()
    file = load_dotenv(env_path)
    return os.getenv('NEWS_API')


def get_news(keyword):
    """
    This function will fetch data from NEWS API based on the keyword entered.
    The API Key required is taken from the function 'get_newskey'.
    """

    #api_key taken from the get_newskey function
    api_key = get_newskey()

    #base url of the API forming the basis for the request
    base_url = "https://newsapi.org/v2/everything?"

    #parts of the news articles that the search shall refer to. It is possible to choose between "content", "title", "content"
    search_in = "content"

    #criteria for sorting the output of the API
    sort = "popularity" #relevancy, popularity, publishedAt

    #web sources to be used
    sources = "cnn" #domains where we would like to search

    #date where the search shall start, default via datetime.today: the current date when the API request is made
    date = datetime.today().strftime('%Y-%m-%d')

    # line of code to make the actual request based on the variables defined before
    source_url = f'{base_url}q={keyword}&from="{date}"&sortBy={sort}&sources={sources}&searchIn={search_in}&apiKey={api_key}'

    news = requests.get(source_url, allow_redirects=True).json()

    return(news)


def get_urls(keyword, limit=10):
    """
    This function will filter the output of the API resulting in a list
    of the URL's of the articles included in that output from NEWS API
    based on the keyword entered.
    """

    api_result = get_news(keyword)

    list_of_urls = []

    for i in range(len(api_result['articles'])):

        list_of_urls.append(api_result['articles'][i]['url'])

    return list_of_urls[:limit]


def get_API_sources():
    """
    This function will fetch data about the sources available on the News API
    """

    #api_key taken from the get_newskey function
    api_key = get_newskey()

    # line of code to make the actual request based on the variables defined before
    API_sources_url = f"https://newsapi.org/v2/top-headlines/sources?apiKey={api_key}"

    API_sources = requests.get(API_sources_url, allow_redirects=True).json()

    return(API_sources)


def get_list_of_API_news_providers():
    """
    This function will create a list of the id's of the news providers available on the News API
    """

    api_sources = get_API_sources()

    list_of_API_news_providers = []

    for i in range(len(api_sources['sources'])):

        list_of_API_news_providers.append(api_sources['sources'][i]['id'])

    return list_of_API_news_providers


def get_headlines_for_categories(category, country):
    """
    This function will fetch live, top, breaking headlines from NEWS API based on the category entered.
    The API Key required is taken from the function 'get_newskey'.
    """

    #api_key taken from the get_newskey function
    api_key = get_newskey()

    #base url of the API forming the basis for the request
    base_url = "https://newsapi.org/v2/top-headlines?"

    # line of code to make the actual request based on the variables defined before
    source_url = f'{base_url}category={category}&country={country}&apiKey={api_key}'

    headlines_for_category = requests.get(source_url, allow_redirects=True).json()

    return(headlines_for_category)


def get_urls_for_categories(category, country='us', limit=10):
    """
    This function will filter the output of "get_headlines_for_categories" resulting in a list
    of the URL's of the articles included in that output from NEWS API
    based on the category and country entered.
    """

    api_result_for_category = get_headlines_for_categories(category, country)

    list_of_urls_for_category = []

    for i in range(len(api_result_for_category['articles'])):

        list_of_urls_for_category.append(api_result_for_category['articles'][i]['url'])

    return list_of_urls_for_category[:limit]
