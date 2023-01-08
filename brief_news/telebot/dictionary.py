# import requests
from brief_news.interface.main import transfomer_summaries

def get_info(keyword):
    '''Get summaries from the DS model.'''

    mapping = {'0':'business',
            '1':'entertainment',
            '2':'general',
            '3':'health',
            '4':'science',
            '5':'sports',
            '6':'technology'}

    if keyword in mapping:

        df = transfomer_summaries(mapping[keyword])
        return df

    else:
        # return general if category not in a list
        df = transfomer_summaries(mapping['2'])
        return df
