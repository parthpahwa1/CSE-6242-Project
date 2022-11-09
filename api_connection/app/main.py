import requests
import json
import pandas as pd
import time
import pprint
pp = pprint.PrettyPrinter(indent=4)
import psycopg2 as pg
from sqlalchemy import create_engine



# Twitch Client ID
clientID = 'qvsl21co22jan49fwd6tn0mw5dohjb'
client_id = clientID
secret = "7jy9s6ah91zhig2783bavrua3h3y7t"

#Request app access token
url = 'https://id.twitch.tv/oauth2/token'
body = 'client_id={0}&client_secret={1}&grant_type={2}'.format(client_id, secret, "client_credentials")

result = requests.post(url, data = body)
#Transform the result into json and get the app access token and token type
app_access_token = json.loads(result.text)["access_token"]
token_type = json.loads(result.text)["token_type"].capitalize()

def get_top_100_games(clientID = clientID , app_access_token = app_access_token):
    ''' Given Client ID, pings twitch API for top 100 games. Returns the entire request object'''
    # Need to pass client ID with each request in header
    fullToken = "Bearer " + app_access_token

    headers = {'Client-Id': clientID, "Authorization": fullToken}

    url = '''https://api.twitch.tv/helix/games/top?first=100'''
    r = requests.get(url, headers=headers)
    return r

data = get_top_100_games(clientID = clientID , app_access_token = app_access_token)

# engine = create_engine('postgresql://199.247.28.78/Twitch?user=postgres&password=postgres')
# df.to_sql('stream_data', engine, if_exists='append',index=False)


####################################################################################
####################################################################################
####################################################################################

from fastapi import FastAPI

app = FastAPI()


@app.get("/")
async def get_top_100_games_co():
    return {"top_100_games": data}
    


# http://127.0.0.1:8000
# root path to get that response


# 

