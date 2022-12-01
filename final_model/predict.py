# Librairies
import pandas as pd
import numpy as np
from fetch_data import DB
from process_data import ProcessStreamData 
from MAB import MAB
import time
from sqlalchemy import create_engine
from datetime import timedelta

class Inference:
    def __init__(self) -> None:
        self.MIN_STREAM_TIME_THRESHOLD = 0.5 # Hours
        self.MAX_STREAM_TIME_THRESHOLD = 7 # Hours
        self.SHIFT_DURATION = 4 #Days
        self.data_processsor = ProcessStreamData(
            self.MIN_STREAM_TIME_THRESHOLD,
            self.MAX_STREAM_TIME_THRESHOLD,
            self.SHIFT_DURATION
        )

    

    def preprocess_data(self, stream_data):
        
        stream_data = self.data_processsor.clean_time_fields(stream_data)
        stream_data = self.data_processsor.convert_maturity_ratings_to_float(stream_data)
        stream_data = self.data_processsor.filter_for_language(stream_data)
        
        
        stream_data = self.data_processsor.create_time_chunks(stream_data)
        stream_data = self.data_processsor.clean_time_fields(stream_data)
        stream_data = self.data_processsor.filter_for_stream_length(stream_data)
        stream_data = self.data_processsor.get_sentiment(stream_data)
        

        # Feature_engineering
        df_with_features = self.data_processsor.perform_feature_engineering(stream_data)

        return df_with_features
    
    
    def fetch_data(self, date_time):
        print('Running get data')
            # 2010-11-29 00:00:00

        date_lowrbound = date_time - timedelta(days=1, hours=4)
        date_upperboud = date_time - timedelta(days=1)

        SQL = """
            SELECT 
                * 
            FROM 
                stream_data
            WHERE
                time_logged >= '{0}'
            AND
                time_logged <= '{1}'
            """.format(str(date_lowrbound), str(date_upperboud))

        stream_data = DB().get_stream_data(SQL)
        print('Processing data')
        processed_data = self.preprocess_data(stream_data)

        print('Creating Targets')
        processed_data = self.data_processsor.get_prediction_dictionary(processed_data)

        return processed_data, date_time.hour//4
    

    def run(self, preference, data=None, timesplit=None):

        date_time = pd.to_datetime(np.datetime64('now'))
        if data is None:
            data, timesplit = self.fetch_data(date_time)
        else:
            timesplit = timesplit
        model = MAB()

        # timesplit = pd.to_datetime(np.datetime64("{0} {1}".format(date, time_of_day))).hour//4
        data = data[data['time_logged_encoded'] == timesplit].reset_index(drop=True)
        

        preference = np.array(preference)
        preference = preference/preference.sum()
        model_prediction = model.predict(data, timesplit=timesplit, preference=preference)

        df = self.create_writable_df(model_prediction, preference, date_time)
        self.write_to_DB(df)

        return model_prediction
    
    def create_writable_df(self, model_prediction, preference, date_time):
        game_names = [k for k, v in sorted(model_prediction.items(), key=lambda item: item[1])][-10:]
        game_values = np.array([v for k, v in sorted(model_prediction.items(), key=lambda item: item[1])][-10:])

        index_list = np.random.choice(len(game_values), 3, replace=False, p=game_values/game_values.sum())
        final_game_list = [game_names[i] for i in index_list]

        result_dict = {
            'GAME_1': final_game_list[0],
            'GAME_2': final_game_list[1],
            'GAME_3': final_game_list[2],
            'MeanViewer': preference[0],
            'MedianViewer': preference[1],
            'ShortStream': preference[2],
            'MediumStream': preference[3],
            'LongStream': preference[4],
            'MatureContent': preference[5],
            'DateTimeExecuted': str(date_time)
        }
        df = pd.DataFrame.from_dict(result_dict, orient='index').T
        return df

    def write_to_DB(self, df):
        print("push_gameids_to_SQL")
        game_df = df
        engine = create_engine('postgresql://twitch.caampywfg0rz.us-east-1.rds.amazonaws.com:5432/Twitch?user=GaTech_team_96&password=i-love-my-coffee-without-milk-and-sugar-at-800AM')
        game_df.to_sql('user_recommendation', engine, if_exists='append', index=False)
        engine.dispose()
    
    
        
    
    
    