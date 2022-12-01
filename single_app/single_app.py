# Librairies
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from datetime import timedelta
import psycopg2
import sys
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import pickle
import os

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
    
    
        
    
import pandas as pd
import numpy as np
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
    
    def get_prediction_dictionary(self, stream_data):
        """
        Dictionary with following hierarcy

        -time slot
            - game name
                - DataFrame with targets 

        """ 
        feauture_column_names = list(stream_data.columns)[3:]
        return stream_data.groupby(['time_logged_encoded', 'game_name'])[feauture_column_names].mean().reset_index()


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



class DB:

    def __init__(self) -> None:
        self.co_param = {
            "host"      : "twitch.caampywfg0rz.us-east-1.rds.amazonaws.com",
            "database"  : "Twitch",
            "user"      : "GaTech_team_96",
            "password"  : "i-love-my-coffee-without-milk-and-sugar-at-800AM"
        }

    def connect(self, co_param):
        """
        Connect to the PostgreSQL database server
        """
        conn = None
        try:
            # connect to the PostgreSQL server
            print('Connecting to the PostgreSQL database...')
            conn = psycopg2.connect(**co_param)
        except (Exception, psycopg2.DatabaseError) as error:
            print(error)
            sys.exit(1) 
        print("Connection successful")
        return conn

    def postgresql_to_dataframe(self, conn, select_query, column_names):
        """
        Tranform a SELECT query into a pandas dataframe
        """
        cursor = conn.cursor()
        try:
            print('Executing Query')
            cursor.execute(select_query)
        except (Exception, psycopg2.DatabaseError) as error:
            print("Error: %s" % error)
            cursor.close()
            return 1
        
        # Naturally we get a list of tupples
        tupples = cursor.fetchall()
        print('Execution Successful')
        cursor.close()
        
        # We just need to turn it into a pandas dataframe
        print('Creating raw dataframe')
        df = pd.DataFrame(tupples, columns=column_names)
        return df

    def get_stream_data(self, sql_query=None):
        # SQL query
        if sql_query is None:
            sql_query = """SELECT * FROM stream_data"""

        # Column names
        stream_data_col_names = ["game_id","stream_id","language","started_at","title",
                                    "stream_type","user_id","user_name","viewer_count","user_login","game_name",
                                    "thumbnail_url","tag_ids","is_mature","time_logged"]

        # Retrieving the data
        stream_data = self.postgresql_to_dataframe(self.connect(self.co_param), sql_query, stream_data_col_names)
        return stream_data

class MAB:
    def __init__(self) -> None:
        self.SCALER_FILE_LOC = "./mab_std_sclaer.pkl"
        self.MODEL_FILE_LOC = "./MAB_WEIGHTS.pkl"
        self.GAME_NAMES_FILE_LOC = "./game_names.pkl"

        self.NUM_PREFERECES = 6
        self.NUM_SAMPLED_PREFERENCES = 2048

        self.context_features_list = [
            'mean_viewer_count',
            'median_viewer_count',
            'total_viewer_count',
            'mean_stream_duration_hours',
            'median_stream_duration_hours',
            'total_stream_duration_hours',
            'is_mature',
            'mean_positive_sentiment',
            'mean_negative_sentiment',
            'mean_neutral_sentiment'
        ]
        self.indx_to_reward_str_mapping = {
            0 :'target_mean_viewer_count_growth',
            1 :'target_median_viewer_count_growth',
            2 :'target_is_short_stream',
            3 : 'target_is_medium_stream',
            4 : 'target_is_long_stream',
            5 : 'target_is_mature'
        }

        self.SCALER_DICTIONRY = self.load_standard_scalers()
        self.GAME_NAMES = self.load_game_names()
        self.GAME_INDEX_NAME_MAP = self.load_game_index_to_name_mapping()

        self.MIN_TRIALS = {}
        self.alphas = [0.001]
    
    def load_game_index_to_name_mapping(self):
        if len(self.GAME_NAMES) == 0:
            return {}
        else:
            GAME_INDEX_NAME_MAP = {}
            for timeslot in self.GAME_NAMES.keys():
                GAME_INDEX_NAME_MAP[timeslot] = {}

                for i in range(0, len(self.GAME_NAMES[timeslot])):
                    GAME_INDEX_NAME_MAP[timeslot][i] = list(self.GAME_NAMES[timeslot])[i]
        return GAME_INDEX_NAME_MAP

    def load_game_names(self):
        if os.path.isfile(self.GAME_NAMES_FILE_LOC):
            return pickle.load(open(self.GAME_NAMES_FILE_LOC,'rb'))

        return { }

    def load_standard_scalers(self):
        if os.path.isfile(self.SCALER_FILE_LOC):
            return pickle.load(open(self.SCALER_FILE_LOC,'rb'))

        return { }
    
    def save_standard_scalers(self):
        pickle.dump(self.SCALER_DICTIONRY, open(self.SCALER_FILE_LOC, 'wb'))

    def save_game_names(self):
        pickle.dump(self.GAME_NAMES, open(self.GAME_NAMES_FILE_LOC,'wb'))


    def init_train_data(self, training_data):
        loaded_dict = training_data
        TRAIN_MATRIX = {}

        for timeslot in loaded_dict.keys():
            
            TRAIN_MATRIX[timeslot] = {}
            self.GAME_NAMES[timeslot] = set()
            self.MIN_TRIALS[timeslot] = 10e6
            
            for game in loaded_dict[timeslot].keys():
                if loaded_dict[timeslot][game].shape[0] > 6:
                    if loaded_dict[timeslot][game].shape[0] < self.MIN_TRIALS[timeslot]:
                        self.MIN_TRIALS[timeslot] = loaded_dict[timeslot][game].shape[0]

                    self.GAME_NAMES[timeslot].add(game)
                    TRAIN_MATRIX[timeslot][game] = loaded_dict[timeslot][game]
        
        self.save_game_names()
        self.GAME_INDEX_NAME_MAP = self.load_game_index_to_name_mapping()

        return TRAIN_MATRIX
    
    def predict(self, input, timesplit, preference, model=None):
        if model is None:
            model = pickle.load(open(self.MODEL_FILE_LOC, 'rb'))

        SCALER_DICTIONRY = self.load_standard_scalers()[timesplit]
        

        FINAL_MODEL = model[timesplit]
        p_vector = {}
        for i in range(0, FINAL_MODEL.shape[0]):
            game_name = self.GAME_INDEX_NAME_MAP[timesplit][i]
            arm = FINAL_MODEL[i]
            
            eval_data = input[input['game_name']==game_name][self.context_features_list]
            if eval_data.shape[0] > 0:
                INPUT = SCALER_DICTIONRY.transform(eval_data)
                INPUT = np.concatenate((INPUT, preference.reshape(1,-1).repeat(INPUT.shape[0], axis=0)), axis=1).reshape(-1)
                # print(INPUT)
                # print(arm)
                # print(INPUT.shape)
                # print(arm.shape)
                p_vector[game_name] = arm.dot(INPUT)
            else:
                p_vector[game_name] = 0
        
        return p_vector


    def train(self, training_data):
        TRAIN_MATRIX = self.init_train_data(training_data)
        results_dict = {}
        
        for time_split in TRAIN_MATRIX.keys():
            nArms = len(TRAIN_MATRIX[time_split])
            n_trials = (self.MIN_TRIALS[time_split] - 1)
            

            X = [None] * len(self.GAME_INDEX_NAME_MAP[time_split])
            REWARD = [None] * len(self.GAME_INDEX_NAME_MAP[time_split])
            games_index_to_name_mapping = self.GAME_INDEX_NAME_MAP[time_split]
            
            
            FINAL_DATA = np.empty(
                shape=(
                    n_trials*self.NUM_SAMPLED_PREFERENCES, 
                    nArms, 
                    self.NUM_PREFERECES + len(self.context_features_list)
                )
            ) * 0.0
            print(n_trials*self.NUM_SAMPLED_PREFERENCES, nArms)
            REWARD_MATRIX = np.empty(shape=(n_trials*self.NUM_SAMPLED_PREFERENCES, nArms)) * 0.0
            
            print(n_trials*self.NUM_SAMPLED_PREFERENCES, nArms)
            ORACLE = np.empty(shape=(n_trials*self.NUM_SAMPLED_PREFERENCES)) * 0.0
            
            df_for_train = pd.concat([TRAIN_MATRIX[time_split][name] for name in TRAIN_MATRIX[time_split]])

            print(df_for_train.shape)
            df_for_train = df_for_train.reset_index(drop=True)
            df_for_train = self.generate_metrics(df_for_train)
            
            self.SCALER_DICTIONRY[time_split] = StandardScaler()
            self.SCALER_DICTIONRY[time_split].fit(df_for_train[self.context_features_list])
            self.save_standard_scalers()
            
            for preference_num in range(self.NUM_SAMPLED_PREFERENCES):
                preference = self.generate_random_preference()

                df_for_train['reward'] = df_for_train.apply(lambda row: self.calculate_reward(row, preference, self.indx_to_reward_str_mapping), axis=1)
                df_for_train['oracle_value'] = df_for_train.groupby(['log_date'])['reward'].transform(max)
                
                for i in range(len(games_index_to_name_mapping)):
                    temp_data = df_for_train[df_for_train['game_name'] == games_index_to_name_mapping[i]][self.context_features_list]
                    REWARD[i] = df_for_train[df_for_train['game_name'] == games_index_to_name_mapping[i]][['reward']].to_numpy()
                    MAX_REWARD = df_for_train[df_for_train['game_name'] == games_index_to_name_mapping[i]][['oracle_value']].to_numpy()


                    if temp_data.shape[0] > 0:
                        X[i] = np.clip(self.SCALER_DICTIONRY[time_split].transform(temp_data), None, 2)
                        X[i] = np.concatenate((X[i], preference.reshape(1,-1).repeat(X[i].shape[0], axis=0)), axis=1)

                        for trails in range(n_trials):
                            FINAL_DATA[(preference_num+1)*trails][i] = X[i][trails]
                            REWARD_MATRIX[(preference_num+1)*trails][i] = REWARD[i][trails]
                            ORACLE[(preference_num+1)*trails] = MAX_REWARD[trails]

            results_dict[time_split] = {
                alpha: self.lin_ucb(
                    alpha=alpha,
                    X = FINAL_DATA,
                    n_arms = nArms,
                    REWARD_MATRIX = REWARD_MATRIX
                ) 
                for alpha in self.alphas
            }
            

            def plot_regrets(results, oracles, time_slot):
                [plt.plot(self.make_regret(payoffs=x['r_payoffs'], oracles=oracles), label="Time: " + str(time_slot) +
                        " alpha: " + str(alpha)) for (alpha, x) in results.items()]

            plt.figure(figsize=(12.5, 7.5))
            plot_regrets(results_dict[time_split], ORACLE, time_split)
            # plot also the random one
            plt.legend()
            plt.title("Regret for timesplit="+ str(time_split))
            plt.show()

        

        self.save_model_params(results_dict)
        
        
        return results_dict
    
    def save_model_params(self, result):
        model_parameters = {

        } 
        for timeslot in result.keys():
            model_parameters[timeslot] = {

            }
            for alpha in result[timeslot]:
                model_parameters[timeslot] = result[timeslot][alpha]['theta'][-1]
        pickle.dump(model_parameters, open(self.MODEL_FILE_LOC, 'wb'))


    def make_regret(self, payoffs, oracles):
        return np.cumsum(oracles - payoffs)
    
    def lin_ucb(self, alpha, X, n_arms, REWARD_MATRIX):        
        # Data storages
        n_trials, n_arms, n_features = X.shape
        arm_choice = np.empty(n_trials) # used to store agent's choices for each trial
        r_payoffs = np.empty(n_trials) # used to store the payoff for each trial (the payoff for the selected arm based on the true_theta)
        
        theta = np.empty(shape=(n_trials, n_arms, n_features)) + 1e-6 # used to store the predicted theta over each trial
        p = np.empty(shape=(n_trials, n_arms)) # used to store predictions for reward of each arm for each trial
        
        # Lin UCB Objects
        A = np.array([np.diag(np.ones(shape=n_features)) for _ in np.arange(n_arms)]) + 1e-6 # A is the matrix defined as :math:A_a = D_a^TD_a + I_d, and for the initialization it is I_d and will be updated after every trial
        b = np.array([np.zeros(shape=n_features) for _ in np.arange(n_arms)]) + 1e-6 # b is the matrix defined as response vectors (reward for each feature for each arm at each trial, initialized to zero for all features of all arms at every trial)
        
        # The algorithm
        for epoch in range(10):
            for t in range(n_trials):
                # compute the estimates (theta) and prediction (p) for all arms
                for a in range(n_arms):
                    if np.isnan(X[t, a]).any():
                        X[t, a] = np.nan_to_num(X[t, a], nan=1e-5)
                        p[t, a] = np.random.uniform(0, 1.0/(epoch+1))
                    else:
                        inv_A = np.linalg.inv(A[a])
                        theta[t, a] = inv_A.dot(b[a]) # estimate theta as from this formula :math:`\hat{\theta}_a = A_a^{-1}b_a`
                        p[t, a] = theta[t, a].dot(X[t, a]) + alpha * np.sqrt(X[t, a].dot(inv_A).dot(X[t, a])) + np.random.uniform(0, 0.8/(epoch+1))# predictions is the expected mean + the confidence upper bound

                # choosing the best arms
                chosen_arm = np.argmax(p[t])
                x_chosen_arm = X[t, chosen_arm]
                r_payoffs[t] = REWARD_MATRIX[t, chosen_arm] # This payoff is for the predicted chosen arm, and but the payoff is based on theoretical theta (true theta)
                arm_choice[t] = chosen_arm

                # Update intermediate objects (A and b)
                A[chosen_arm] += np.outer(x_chosen_arm, x_chosen_arm.T)


                b[chosen_arm] += r_payoffs[t]*x_chosen_arm # update the b values for each features corresponding to the pay off and the features of the chosen_arm

        return dict(theta=theta, p=p, arm_choice=arm_choice, r_payoffs = r_payoffs)

    def generate_metrics(self, df):
        df['target_mean_viewer_count_growth'] = (df['target_mean_viewer_count_4'] - df['mean_viewer_count']) / (df['mean_viewer_count'] + 1e-5)
        df['target_median_viewer_count_growth'] = (df['target_median_viewer_count_4'] - df['median_viewer_count']) / (df['median_viewer_count'] + 1e-5)
        
        df['target_is_short_stream'] = df['target_median_stream_duration_hours_4'].apply(lambda row: 1 if row <= 1 else 0)
        
        df['target_is_medium_stream'] = df['target_median_stream_duration_hours_4'].apply(lambda row: 1 if row <= 3 and row > 1 else 0)
        df['target_is_long_stream'] = df['target_median_stream_duration_hours_4'].apply(lambda row: 1 if row > 3 else 0)
        df['target_is_mature'] = df['target_is_mature_4']
        return df

    def calculate_reward(self, row, preference, indx_to_reward_str_mapping):
        reward_vector = np.array([0.0] * len(preference))

        for objective in indx_to_reward_str_mapping.keys():
            reward_vector[objective] = row[indx_to_reward_str_mapping[objective]]
            if reward_vector[objective] <= 0:
                reward_vector[objective] = 0
            if reward_vector[objective] >= 3:
                reward_vector[objective] = 3

        reward = np.dot(reward_vector, preference)

        if reward is None:
            reward = 0
        if reward > 3:
            reward = 3
        if reward < 0:
            reward = 0
            
        return reward

    def get_oracle_value_for_day(self, df):
        return np.max(df['reward'])
    
    def generate_random_preference(self):
        return np.random.dirichlet(np.ones(self.NUM_PREFERECES))


def fetch_preference():
    return np.array([1]*6)


inference_class = Inference()
preference = fetch_preference()
inference_class.run(preference)