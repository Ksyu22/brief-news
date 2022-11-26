import pandas as pd
import tensorflow
from transformers import pipeline


def summary_t5_small(articles_list: pd.DataFrame) -> pd.DataFrame:

    """ this function download the t5-small summarization model and
    infer the summary"""

    summarization = pipeline(task='summarization')

    article = articles_list[]
    result = summarization([article])
    summarized_articles.append(result)

    return summarized_articles
