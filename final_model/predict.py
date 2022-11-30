# Librairies
import pandas as pd
import numpy as np
from fetch_data import DB
from process_data import ProcessStreamData 
from MAB import MAB


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
    
    
    def fetch_data(self, date, time_of_day):
        print('Running get data')
            # 2010-11-29 00:00:00
        SQL = """
            SELECT 
                * 
            FROM 
                stream_data
            WHERE
                time_logged >= '{0} {1}'
            """.format(date, time_of_day)

        stream_data = DB().get_stream_data(SQL)
        print('Processing data')
        processed_data = self.preprocess_data(stream_data)

        print('Creating Targets')
        processed_data = self.data_processsor.get_prediction_dictionary(processed_data)

        return processed_data
    

    def run(self, date, time_of_day, preference):
        data = self.fetch_data(date, time_of_day)
        model = MAB()

        timesplit = pd.to_datetime(np.datetime64("{0} {1}".format(date, time_of_day))).hour//4
        data = data[data['time_logged_encoded'] == timesplit].reset_index(drop=True)
        return model.predict(data, timesplit=timesplit, preference=preference)
        
    
    
        
    
    
    