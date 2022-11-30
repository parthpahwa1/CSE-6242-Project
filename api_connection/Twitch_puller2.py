import requests
import json
import pandas as pd
import time
import psycopg2 as pg
from sqlalchemy import create_engine

# Twitch Client ID
clientID = 'zmy348wal3qwmoeum39p6pt9n8zm6k'
client_id = clientID
secret = "48gq7a9yp7fsxxxlu7zfl21e5eu3lt"

#Request app access token
url = 'https://id.twitch.tv/oauth2/token'
body = 'client_id={0}&client_secret={1}&grant_type={2}'.format(client_id, secret, "client_credentials")

result = requests.post(url, data = body)
#Transform the result into json and get the app access token and token type
app_access_token = json.loads(result.text)["access_token"]
token_type = json.loads(result.text)["token_type"].capitalize()

def get_top_100_games(clientID = clientID , app_access_token = app_access_token):
    print("get_top_100_games")
    ''' Given Client ID, pings twitch API for top 100 games. Returns the entire request object'''
    # Need to pass client ID with each request in header
    fullToken = "Bearer " + app_access_token

    headers = {'Client-Id': clientID, "Authorization": fullToken}

    url = '''https://api.twitch.tv/helix/games/top?first=100'''
    r = requests.get(url, headers=headers)
    return r

def get_game_ids(clientID = clientID,app_access_token = app_access_token):
    print("get_game_ids")
    ''' Grabs top 100 games, then grabs top 100 ID's '''
    top_100 = get_top_100_games()

    fullToken = "Bearer " + app_access_token

    headers = {'Client-Id': clientID, "Authorization": fullToken}
    url = '''https://api.twitch.tv/helix/games'''
    for counter,game in enumerate(json.loads(top_100.text)['data']):
        # First element requires ? before id=, the rest require &id=
        if counter == 0:
            url += '?id=' + game['id']
        else:
            url += '&id=' + game['id']
    r = requests.get(url, headers=headers)


    return r

def push_gameids_to_SQL(r):
    print("push_gameids_to_SQL")
    game_df = pd.json_normalize(json.loads(r.text)['data'])
    curr_time = time.strftime("%Y-%m-%d %H:%M:%S",time.gmtime())
    game_df['time_logged'] = curr_time
    
    game_df.rename(columns = {'id': 'game_id','name': 'game_name','box_art_url': 'pic_url'},inplace = True)

    df1 = game_df[['game_id', 'game_name','pic_url','time_logged']]
    engine = create_engine('postgresql://twitch.caampywfg0rz.us-east-1.rds.amazonaws.com:5432/Twitch?user=GaTech_team_96&password=i-love-my-coffee-without-milk-and-sugar-at-800AM')
    df1.to_sql('game_info', engine, if_exists='append',index=False)
    engine.dispose()

def check_api_limit_reached(req, ignore_limit = False):
    '''Check remaining API pings for request REQ. If API requests is <=1, wait for 30s
    so for all requests to refill. Returns remaining requests'''
    if int(req.headers['Ratelimit-Remaining']) <= 1: # No more requests, need to pause for 30s
        if ignore_limit:
            return int(req.headers['Ratelimit-Remaining'])
        print('Waiting for API limit to refresh (30s)...')
        time.sleep(30)
        print('Continuing...')
    return int(req.headers['Ratelimit-Remaining'])

def get_top_100_streamers_for_each_game(game_dict):
    print("get_top_100_streamers_for_each_game")
    '''Given the twitch response for top 100 games, this will cycle through and pull the top 100
    streamers for each game, stored under a dict entry of the title of that game'''
    stream_dict = dict()
    fullToken = "Bearer " + app_access_token
    headers = {'Client-Id': clientID, "Authorization": fullToken}

    url = 'https://api.twitch.tv/helix/streams?first=100&game_id='
    for game in game_dict['data']:
        req = requests.get(url + game['id'],headers=headers)
        check_api_limit_reached(req)
        stream_dict[game['name']]=json.loads(req.text)
    return stream_dict

def json_to_dataframe(json_data):
    print("json_to_dataframe")
    total_streams_df = pd.DataFrame(
        columns = ['game_id','id','language','started_at','title','type','user_id','user_name','viewer_count'])
    for game_key in list(json_data.keys()):
        game_streams_df = pd.json_normalize(json_data[game_key]['data'])
        total_streams_df = pd.concat([total_streams_df, game_streams_df], sort = False)
    #total_streams_df.drop(columns = ['community_ids','thumbnail_url','tag_ids'], inplace = True)
    return total_streams_df


def run_all(event,context):
    print("run started")
    clientID = 'zmy348wal3qwmoeum39p6pt9n8zm6k'
    client_id = clientID
    secret = "48gq7a9yp7fsxxxlu7zfl21e5eu3lt"
    # Request app access token
    url = 'https://id.twitch.tv/oauth2/token'
    body = 'client_id={0}&client_secret={1}&grant_type={2}'.format(client_id, secret, "client_credentials")

    result = requests.post(url, data=body)
    # Transform the result into json and get the app access token and token type
    global app_access_token
    global token_type

    app_access_token = json.loads(result.text)["access_token"]
    token_type = json.loads(result.text)["token_type"].capitalize()

    top_100_game_ids = get_game_ids()
    push_gameids_to_SQL(top_100_game_ids)

    r = get_top_100_games()
    r_dict = json.loads(r.text)
    stream_dict = get_top_100_streamers_for_each_game(r_dict)
    df = json_to_dataframe(stream_dict)

    df.rename(columns={'id': 'stream_id', 'type': 'stream_type'}, inplace=True)
    curr_time = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    df['time_logged'] = curr_time

    engine = create_engine('postgresql://twitch.caampywfg0rz.us-east-1.rds.amazonaws.com:5432/Twitch?user=GaTech_team_96&password=i-love-my-coffee-without-milk-and-sugar-at-800AM')
    df.to_sql('stream_data', engine, if_exists='append', index=False)
    print("finished")



