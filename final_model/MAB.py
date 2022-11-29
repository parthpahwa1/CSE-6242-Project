import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle
import os

class MAB:
    def __init__(self) -> None:
        self.SCALER_FILE_LOC = "./Resources/mab_std_sclaer.pkl"
        self.MODEL_FILE_LOC = "./Resources/MAB_WEIGHTS.pkl"
        self.GAME_NAMES_FILE_LOC = "./Resources/game_names.pkl"

        self.NUM_PREFERECES = 6
        self.NUM_SAMPLED_PREFERENCES = 512

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
        p_vector = []
        for i in range(0, FINAL_MODEL.shape[0]):
            game_name = self.GAME_INDEX_NAME_MAP[timesplit][i]
            arm = FINAL_MODEL[i]

            INPUT = SCALER_DICTIONRY.transform(input[input['game_name']==game_name][self.context_features_list])
            INPUT = np.concatenate((INPUT, preference.reshape(1,-1).repeat(INPUT.shape[0], axis=0)), axis=1)
            p_vector.append(arm.dot(INPUT))
        
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
                        X[i] = self.SCALER_DICTIONRY[time_split].transform(temp_data)
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
        
        theta = np.empty(shape=(n_trials, n_arms, n_features)) # used to store the predicted theta over each trial
        p = np.empty(shape=(n_trials, n_arms)) # used to store predictions for reward of each arm for each trial
        
        # Lin UCB Objects
        A = np.array([np.diag(np.ones(shape=n_features)) for _ in np.arange(n_arms)]) # A is the matrix defined as :math:A_a = D_a^TD_a + I_d, and for the initialization it is I_d and will be updated after every trial
        b = np.array([np.zeros(shape=n_features) for _ in np.arange(n_arms)]) # b is the matrix defined as response vectors (reward for each feature for each arm at each trial, initialized to zero for all features of all arms at every trial)
        
        # The algorithm
        for epoch in range(5):
            for t in range(n_trials):
                # compute the estimates (theta) and prediction (p) for all arms
                for a in range(n_arms):
                    if np.isnan(X[t, a]).any():
                        X[t, a] = np.nan_to_num(X[t, a], nan=1e-5)
                        p[t, a] = np.random.uniform(0, 1.0/n_arms)
                    else:
                        inv_A = np.linalg.inv(A[a])
                        theta[t, a] = inv_A.dot(b[a]) # estimate theta as from this formula :math:`\hat{\theta}_a = A_a^{-1}b_a`
                        p[t, a] = theta[t, a].dot(X[t, a]) + alpha * np.sqrt(X[t, a].dot(inv_A).dot(X[t, a])) # predictions is the expected mean + the confidence upper bound

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
