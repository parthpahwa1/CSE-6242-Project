# Librairies
import pandas as pd
import numpy as np
from fetch_data import DB
from process_data import ProcessStreamData 
import nltk
import pickle


class Train:
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
    
    
    

    def run(self):
        print('Running get data')
        stream_data = DB().get_stream_data()

        print('Processing data')
        processed_data = self.preprocess_data(stream_data)

        print('Creating Targets')
        training_data = self.data_processsor.get_training_dictionary(processed_data)

        print('Creating dictionary pickle')
        with open('./Resources/saved_dictionary.pkl', 'wb') as f:
            pickle.dump(training_data, f)

        return training_data
        
    
    
        
    
    
    