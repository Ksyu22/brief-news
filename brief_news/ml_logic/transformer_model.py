import pandas as pd
import tensorflow
from transformers import pipeline


def summary_t5_small(articles_list: pd.DataFrame) -> pd.DataFrame:

    """ This function downloads the t5-small summarization model and
    infer the summary"""

    summarization = pipeline(task='summarization')

    article=articles_list[0]
    result = summarization([article])

    return result
