import pandas as pd
import numpy as np
from fetch_data import DB
import nltk

nltk.download('vader_lexicon')

class ProcessStreamData():
    def __init__(self, MIN_STREAM_TIME_THRESHOLD, MAX_STREAM_TIME_THRESHOLD, SHIFT_DURATION) -> None:
        self.MIN_STREAM_TIME_THRESHOLD = MIN_STREAM_TIME_THRESHOLD # Hours
        self.MAX_STREAM_TIME_THRESHOLD = MAX_STREAM_TIME_THRESHOLD # Hours
        self.SHIFT_DURATION = SHIFT_DURATION #Days

    def clean_time_fields(self, stream_data):
        # changing formatting from 
        stream_data.loc[:,"started_at"] = stream_data.loc[:,"started_at"].map(lambda x: x.rstrip("Z"))
        stream_data.loc[:,"started_at"] = stream_data.loc[:,"started_at"].map(lambda x: x.replace("T", " "))
        stream_data["stream_duration_hours"] = pd.to_datetime(stream_data["time_logged"])-pd.to_datetime(stream_data["started_at"])
        stream_data['stream_duration_hours'] = stream_data['stream_duration_hours']/np.timedelta64(1, 'h')

        return stream_data

    def convert_maturity_ratings_to_float(self, stream_data):
        # Changing is_mature with True =1 & False = 0
        stream_data.loc[stream_data["is_mature"] == True, "is_mature"] = 1
        stream_data.loc[stream_data["is_mature"] == False, "is_mature"] = 0
        
        return stream_data

    def filter_for_language(self, stream_data):
        stream_data = stream_data[stream_data["language"] == "en"]
        return stream_data.reset_index(drop=True)
            
    def create_time_chunks(self, stream_data):
        stream_data['log_date'] = pd.to_datetime(stream_data['time_logged']).dt.date
        stream_data['time_logged_encoded'] = pd.to_datetime(stream_data['time_logged']).dt.hour
        stream_data['time_logged_encoded'] =  stream_data['time_logged_encoded']//4

        return stream_data
    
    def filter_for_stream_length(self, stream_data):
        stream_data = stream_data[
            (stream_data['stream_duration_hours'] > self.MIN_STREAM_TIME_THRESHOLD) & 
            (stream_data['stream_duration_hours'] < self.MAX_STREAM_TIME_THRESHOLD)
        ].reset_index(drop=True)

        return stream_data

    def get_sentiment(self, df):
        from nltk.sentiment.vader import SentimentIntensityAnalyzer
        si = SentimentIntensityAnalyzer()

        def get_sentiment_row(row):
            sentiment = si.polarity_scores(row)
            return pd.Series([sentiment['pos'], sentiment['neg'], sentiment['neu']])

        df['positive_sentiment'] = 0.0
        df['negative_sentiment'] = 0.0
        df['neutral_sentiment'] = 0.0

        df[['positive_sentiment',
            'negative_sentiment',
            'neutral_sentiment']] =  df.title.apply(lambda row: get_sentiment_row(row))

        return df
    
    def get_training_dictionary(self, stream_data):
        """
        Dictionary with following hierarcy

        -time slot
            - game name
                - DataFrame with targets 

        """ 
        feauture_column_names = list(stream_data.columns)[3:-3]

        training_data_frame_dictionary = {
    
        }
        for time_slot in stream_data.time_logged_encoded.unique():
            df_filtered_on_timeslot = stream_data[stream_data['time_logged_encoded'] == time_slot].copy()
            
            training_data_frame_dictionary[time_slot] = {
                
            }
            
            for game in df_filtered_on_timeslot.game_name.unique():
                df_filtered_on_game = df_filtered_on_timeslot[df_filtered_on_timeslot['game_name'] == game].copy()
                df_filtered_on_game = df_filtered_on_game.sort_values(by='log_date').reset_index(drop=True)
                
                
                for col in feauture_column_names:
                    df_filtered_on_game['target_'+ col + '_' + str(self.SHIFT_DURATION)] = df_filtered_on_game[col].shift(-self.SHIFT_DURATION)
                
                training_data_frame_dictionary[time_slot][game] = df_filtered_on_game.dropna().reset_index(drop=True)
        return training_data_frame_dictionary
    
    def perform_feature_engineering(self, stream_data):
        stream_data = stream_data.copy()

        # Add mean, median, total viewership 
        df_with_features = stream_data.groupby(['time_logged_encoded', 'game_name', 'log_date'])[['viewer_count']].mean().reset_index()
        df_with_features = df_with_features.rename(columns={'viewer_count': 'mean_viewer_count'})

        temp_df = stream_data.groupby(['time_logged_encoded', 'game_name', 'log_date'])[['viewer_count']].median().reset_index()
        temp_df = temp_df.rename(columns={'viewer_count': 'median_viewer_count'})
        df_with_features = pd.merge(df_with_features, temp_df)

        temp_df = stream_data.groupby(['time_logged_encoded', 'game_name', 'log_date'])[['viewer_count']].sum().reset_index()
        temp_df = temp_df.rename(columns={'viewer_count': 'total_viewer_count'})
        df_with_features = pd.merge(df_with_features, temp_df)


        # Add mean, median, total stream time 
        temp_df = stream_data.groupby(['time_logged_encoded', 'game_name', 'log_date'])[['stream_duration_hours']].mean().reset_index()
        temp_df = temp_df.rename(columns={'stream_duration_hours': 'mean_stream_duration_hours'})
        df_with_features = pd.merge(df_with_features, temp_df)

        temp_df = stream_data.groupby(['time_logged_encoded', 'game_name', 'log_date'])[['stream_duration_hours']].median().reset_index()
        temp_df = temp_df.rename(columns={'stream_duration_hours': 'median_stream_duration_hours'})
        df_with_features = pd.merge(df_with_features, temp_df)

        temp_df = stream_data.groupby(['time_logged_encoded', 'game_name', 'log_date'])[['stream_duration_hours']].sum().reset_index()
        temp_df = temp_df.rename(columns={'stream_duration_hours': 'total_stream_duration_hours'})
        df_with_features = pd.merge(df_with_features, temp_df)


        # Add average matrure rating 
        temp_df = stream_data.groupby(['time_logged_encoded', 'game_name', 'log_date'])[['is_mature']].mean().reset_index()
        df_with_features = pd.merge(df_with_features, temp_df)


        # Add mean sentiment
        temp_df = stream_data.groupby(['time_logged_encoded', 'game_name', 'log_date'])[['positive_sentiment']].mean().reset_index()
        temp_df = temp_df.rename(columns={'positive_sentiment': 'mean_positive_sentiment'})
        df_with_features = pd.merge(df_with_features, temp_df)

        temp_df = stream_data.groupby(['time_logged_encoded', 'game_name', 'log_date'])[['negative_sentiment']].mean().reset_index()
        temp_df = temp_df.rename(columns={'negative_sentiment': 'mean_negative_sentiment'})
        df_with_features = pd.merge(df_with_features, temp_df)

        temp_df = stream_data.groupby(['time_logged_encoded', 'game_name', 'log_date'])[['neutral_sentiment']].mean().reset_index()
        temp_df = temp_df.rename(columns={'neutral_sentiment': 'mean_neutral_sentiment'})
        df_with_features = pd.merge(df_with_features, temp_df)

        return df_with_features