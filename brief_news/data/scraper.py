import numpy as np
import pandas as pd
import requests
#from newsapi import NewsApiClient
from datetime import datetime
from dotenv import load_dotenv, find_dotenv
import os

from bs4 import BeautifulSoup
import re
from itertools import chain
from collections import Counter


def CNN_scraper(url):
    """
    Input: 'str'
    Output: 'dict'

    The function recieve an url (must be from CNN), fetch for the html and uses BS4 to extract the paragraph
    with the class='paragraph inline-placeholder' which contains the text. Then clean and merge the strings.
    It returns a dictionary with the title and the text of the news
    """
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    text = soup.find_all('p', class_='paragraph inline-placeholder')
    text = [item.text.strip() for item in text]
    text = ''.join(text).replace('\xa0', ' ')
    title = soup.title.string.split('|')[0]

    return {'title': title, 'article': text, 'id': 0, 'orig_id': 0}


def DailyMail_scraper(url):
    """
    Input: 'str'
    Output: 'dict'

    The function recieve an url (must be from DailyMail), fetch for the html and uses BS4 to extract the paragraph
    with the class='paragraph inline-placeholder' which contains the text. Then clean and merge the strings.
    It returns a dictionary with the title and the text of the news
    """
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    text = soup.find_all('p', class_='mol-para-with-font')
    text = [item.text.strip() for item in text]
    text = ''.join(text).replace('\xa0', ' ')
    title = soup.title.string.split('|')[0]

    return {'title': title, 'article': text, 'id': 0, 'orig_id': 0}


def General_scraper(url):
    """
    Input: 'str'
    Output: 'dict'
    
    The function recieve an url, fetch for the html and uses BS4 to extract the paragraph tags. Then it
    counts the number of times that each paragraph is repeated and uses the most repeated (in a news must be text)
    to scrape the news from the website. It returns a dictionary with the title and the text of the news.
    """
    
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    text = soup.find_all('p')

    attrs_list = [' '.join(item.attrs['class']) for item in text if len(item.attrs) > 0]
    values, counts = np.unique(attrs_list, return_counts=True)
    idx = np.where(counts == np.max(counts))
    text_class = str(values[idx]).strip('[]\'')

    text = soup.find_all('p', class_=text_class)
    text = [item.text.strip() for item in text]
    text = ' '.join(text).replace('\xa0', ' ')
    title = soup.title.string.split('|')[0]

    return {'title': title, 'article': text, 'id': 0, 'orig_id': 0}
