
import pandas as pd

from brief_news.data.big_query import get_bq_chunk
from brief_news.ml_logic.preprocessor import cleaning, preprocessing_target

from brief_news.ml_logic.params import (CHUNK_SIZE,
                                      DATASET_SIZE,
                                      VALIDATION_DATASET_SIZE)




def preprocess(source_type='train'):
    """
    Preprocess the dataset by chunks fitting in memory.
    parameters:
    - source_type: 'train' or 'val'
    """

    # iterate on the dataset, by chunks
    chunk_id = 0
    row_count = 0
    cleaned_row_count = 0
    source_name = f"{source_type}_{DATASET_SIZE}"
    destination_name = f"{source_type}_processed_{DATASET_SIZE}"


    #print(Fore.BLUE + f"\nProcessing chunk n°{chunk_id}..." + Style.RESET_ALL)

    # extracting chunk of data
    data_chunk = get_bq_chunk(target_table=source_type,
                            index=chunk_id * CHUNK_SIZE,
                            chunk_size=CHUNK_SIZE)

    print('Chunk was extracted')

    row_count += data_chunk.shape[0]

    #checking if not test dataset cause will be used in evaluate preprocessing
    if source_type!='test':

        article_chunk_cleaned = cleaning(data_chunk['article'])
        highlight_chunk_cleaned = preprocessing_target(data_chunk['highlights'])

    print('Chunks were cleaned')

    return article_chunk_cleaned, highlight_chunk_cleaned




if __name__ == '__main__':
    preprocess('train')
