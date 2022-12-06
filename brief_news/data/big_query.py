import pandas as pd
from google.cloud import bigquery
from dotenv import load_dotenv, find_dotenv
import os

from brief_news.ml_logic.params import PROJECT, DATASET

def get_bq_chunk(target_table: str,
              index: int,
              chunk_size: int) -> pd.DataFrame:

    """
    return a chunk of a big query dataset table
    format the output dataframe according to the provided data types
    """

    table = f'{PROJECT}.{DATASET}.{target_table}_set'

    client = bigquery.Client()

    rows = client.list_rows(table,
                            start_index=index,
                            max_results=chunk_size)

    # convert to expected data types
    big_query_df = rows.to_dataframe()

    if big_query_df.shape[0] == 0:
        return None  # end of data

    #big_query_df = big_query_df.astype(dtypes)

    return big_query_df
