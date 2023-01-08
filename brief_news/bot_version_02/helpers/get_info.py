import os
import requests
import constants.enpoints as ENDPOINTS

API_URL = os.environ.get("API_URL")

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
        api_result = fetch_api(mapping[keyword])
        return api_result

    else:
        api_result = fetch_api(mapping['2'])
        return api_result

def fetch_api(keyword):
    return requests.get(
        url='{0}{1}'.format(API_URL, ENDPOINTS.SUMMARIZE),
        params={'keyword': keyword, 'limit': 2}
    ).json()