from itertools import chain
from transformers import pipeline

import os
import tensorflow
import json
import requests
import pandas as pd




def summary_t5_small(articles_list: pd.DataFrame) -> pd.DataFrame:

    """ This function downloads the t5-small summarization model and
    infer the summary
    https://huggingface.co/t5-small
    """

    summarization = pipeline(task='summarization')

    summaries = []

    for index, row in articles_list.iterrows():
        article = row['article']
        summary = summarization([article])

        summaries.append(summary)

    # flattening the list (format list of list of dictionaries)
    articles_list['summary_text'] = pd.DataFrame(list(chain.from_iterable(summaries)))

    return articles_list


def query(payload, API_URL, headers):
    """
    Function sends post request to hugging face api for 'summarization' service
    """
    data = json.dumps(payload)
    response = requests.request("POST", API_URL, headers=headers, data=data)
    return json.loads(response.content.decode("utf-8"))


def summary_bart_large(articles_list: pd.DataFrame) -> pd.DataFrame:
    """
    Function summarizes with facebook/bart-large-cnn
    """
    hf_token = os.getenv('HUGGING_API_TOKEN')

    headers = {"Authorization": f"Bearer {hf_token}"}
    API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"

    # summarizing articles into 150 words, more parameters can be added
    articles_list['summary_text'] = articles_list['article'].apply(lambda article: query({'inputs':article, "parameters": {"max_length": 150}}, API_URL, headers)[0]['summary_text'])

    return articles_list
