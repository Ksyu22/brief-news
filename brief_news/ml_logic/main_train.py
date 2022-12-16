from colorama import Fore, Style
from math import ceil

import pandas as pd

from google.cloud import bigquery
from brief_news.data.data_source import get_chunk, save_chunk
from brief_news.ml_logic.preprocessor import cleaning, preprocessing_target


from brief_news.ml_logic.params import (CHUNK_SIZE,
                                      DATASET_SIZE,
                                      VALIDATION_DATASET_SIZE,
                                      PROJECT, DATASET)




def preprocess(source_type='test'):
    """
    Preprocess the dataset by chunks fitting in memory.
    parameters:
    - source_type: 'train' or 'val'
    """

    # iterate on the dataset, by chunks
    chunk_id = 0
    row_count = 0
    cleaned_row_count = 0
    source_name = f"{source_type}_set"
    destination_name = f"{source_type}_set_processed"

    while (True):

        print(Fore.BLUE + f"\nProcessing chunk n°{chunk_id}..." + Style.RESET_ALL)

        # extracting chunk of data
        data_chunk = get_chunk(source_name=source_type,
                                index=chunk_id * CHUNK_SIZE,
                                chunk_size=CHUNK_SIZE)


        print('Chunk was extracted')

        # Break out of while loop if data is none
        if data_chunk is None:
            print(Fore.BLUE + "\nNo data in latest chunk..." + Style.RESET_ALL)
            break

        row_count += data_chunk.shape[0]

        data_chunk = data_chunk.drop(['id', 'orig_id'], axis=1)



        # Get the length of tables a BigQuery client object.
        client = bigquery.Client()
        table = f'{PROJECT}.{DATASET}.{source_type}_set'
        n_rows = client.get_table(table).num_rows


        for iter in range(ceil(n_rows / CHUNK_SIZE)):

                article_chunk_cleaned = cleaning(data_chunk['article'])
                highlight_chunk_cleaned = preprocessing_target(data_chunk['highlights'], remove_stopwords=False)

                whole_chunk = pd.DataFrame(list(zip(article_chunk_cleaned, highlight_chunk_cleaned)), columns=['article', 'highlights'])

        cleaned_row_count += len(article_chunk_cleaned)

        print('Chunk is preprocessed')

        # save and append the chunk
        is_first = chunk_id == 0

        save_chunk(destination_name=destination_name,
                    is_first=is_first,
                    data=whole_chunk)

        chunk_id += 1

    if row_count == 0:
        print("\n✅ no new data for the preprocessing 👌")
        return None

    print(f"\n✅ data processed saved entirely: {row_count} rows ({cleaned_row_count} cleaned)")

    return None





if __name__ == '__main__':
    print(preprocess('test'))
