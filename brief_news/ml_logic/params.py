"""
brief-news model package params
load and validate the environment variables in the `.env`
"""

import os
import numpy as np

DATASET_SIZE = os.environ.get("DATASET_SIZE")
VALIDATION_DATASET_SIZE = os.environ.get("VALIDATION_DATASET_SIZE")
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE"))
# LOCAL_DATA_PATH = os.path.expanduser(os.environ.get("LOCAL_DATA_PATH"))
# LOCAL_REGISTRY_PATH = os.path.expanduser(os.environ.get("LOCAL_REGISTRY_PATH"))
PROJECT = os.environ.get("PROJECT")
DATASET = os.environ.get("DATASET")
HUGGING_API_TOKEN = os.environ.get("HUGGING_API_TOKEN")

MAX_LEN_TEXT = os.environ.get("MAX_LEN_TEXT")
MAX_LEN_SUM = os.environ.get("MAX_LEN_SUM")




# Use this to optimize loading of raw_data with headers: pd.read_csv(..., dtypes=..., headers=True)
DTYPES_RAW_OPTIMIZED = {

    "id": "int8",
    "article": "str",
    "highlights": "str",
    "orig_id": "str"
}#"key": "O",

COLUMN_NAMES_RAW = DTYPES_RAW_OPTIMIZED.keys()

# Use this to optimize loading of raw_data without headers: pd.read_csv(..., dtypes=..., headers=False)

DTYPES_RAW_OPTIMIZED_HEADLESS = {
    #0: "O",
    0: "int8",
    1: "str",
    1: "str",
    3: "str"
}

DTYPES_RAW_OPTIMIZED = DTYPES_RAW_OPTIMIZED_HEADLESS.keys()
