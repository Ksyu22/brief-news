import pandas as pd
import tensorflow
from itertools import chain
from transformers import pipeline


def summary_t5_small(articles_list: pd.DataFrame) -> pd.DataFrame:

    """ This function downloads the t5-small summarization model and
    infer the summary"""

    summarization = pipeline(task='summarization')

    summaries = []

    for index, row in articles_list.iterrows():
        article = row['article']
        summary = summarization([article])

        summaries.append(summary)

    # flattening the list (format list of list of dictionaries)
    articles_list['summary_text'] = pd.DataFrame(list(chain.from_iterable(summaries)))

    return articles_list
