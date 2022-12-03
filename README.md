# Twitch Game Recommander

In this project, the team proposes to develop a Twitch Game recommendation system based on sentiment analysis and multi-arm bandit in order to provide a point of reference to streamers.
Moreover, we are helping streaming players develop their channel by picking the game, the duration of the stream, and when they shall stream based on the streamer preferences to grow from zero to hero on Twitch.





## Table of Contents

 - [Description](#description)
    - [Twitch Developer console](#11-twitch-developer-console)
    - [Twitch API](#12-twitch-api)
    - [Google Cloud Platform - Scheduler](#13-google-cloud-platform---scheduler)
    - [Google Cloud Platform - Cloud Functions](#14-google-cloud-platform---cloud-functions)
    - [AWS - RDS](#15-aws---rds)
    - [AWS - PostgreSQL](#16-aws---postgresql)
    - [Tableau](#17-tableau)
    - [Code](#18-code)
    - [Inference](#19-inference)
    - [Website](#110-website)
 - [Installation](#installation)
    - [Tokens](#21-tokens)
    - [Environment](#22-environment)
    - [Tableau Dashboard](#23-tableau-dashboard)
 - [Execution](#execution)








## Description


### 1.1. Twitch Developer console

We created a [Twitch](https://www.twitch.tv) account and connected to the [Developer console](https://dev.twitch.tv/console).
Within the console, we have setup and application and defined the OAuth Redirect URLs and Category to retrieve the Client ID and the Client Secret code. 
We setup our console as as below:

```https
  https://dev.twitch.tv/console
```

| Parameter | Inputs     | Description                |
| :-------- | :------- | :------------------------- |
| `Name` | `GaTech_team_project096` | Unique project name to pick |
| `OAuth Redirect URLs` | `https://id.twitch.tv/oauth2/token` | choosen |
| `Category` | `Analytics Tool` | choosen |
| `Client ID` | `qvsl21co22jan49fwd6tn0mw5dohjb` | auto-generated by Twitch |
| `Client Secret` | `hidden` | auto-generated by Twitch |


*You may check the code within the folder > `CODE` > `api_connection` > run `Extracting_data.ipynb` to see an extract from the API.


### 1.2. Twitch API

After writing a script to connect to [Twitch API](https://dev.twitch.tv/docs/api/reference), we retrieve the raw data as below:


| Parameter | Type     | Description                |
| :-------- | :------- | :------------------------- |
| `game_id` | `integer` | Unique Game iD number |
| `stream_id` | `integer` | Unique Stream iD number |
| `language` | `string` | language of the stream set |
| `started_at` | `timestamp` | Date and time the stream started |
| `title` | `string` | Title of the stream provided by the Streamer |
| `stream_type` | `string` | Boolean if the stream is live or a recoded version |
| `user_id` | `integer` | Unique Streamer iD number |
| `user_name` | `string` | Streamer displayed pseudo |
| `viewer_count` | `integer` | Number of viewers at the extraction of the data |
| `user_login` | `string` | Streamer pseudo to log in Twitch |
| `game_name` | `string` | Game name |
| `thumbnail_url` | `string` | URL of a thumbnail image of the game |
| `tag_ids` | `string` | Tags provided by the Streamer |
| `is_mature` | `boolean` | True/False boolean if the streaming content is advised to a mature audience |
| `time_logged` | `timestamp` | Time of the extraction of the data from the Twitch API |

We organized data in the format that corresponds to a minimum of 5'000 rows based on the top 100 games with their top 100 streamers.


### 1.3. Google Cloud Platform - Scheduler

[Google Scheduler](https://console.cloud.google.com/products?supportedpurview=project) allows us to to run every hour for a cron job that sends a pub/sub command.

```bash
Bucket settings:
  • Name > any name
  • Location type > Multi-region
  • Create

Scheduler settings:
  • Create function > Select the function trigger type > Cloud Pub/Sub
  • Create a topic > Pub/Sub topic which will trigger this cloud function > Save
  • Give it an appropriate name and click create topic
  • Connections tab > Allow internal traffic only
```

### 1.4. Google Cloud Platform - Cloud Functions

[Google Cloud Functions](https://console.cloud.google.com/products?supportedpurview=project) helped us to runs the extraction script and saves the information to the AWS Relational Database Service (RDS) database.

```bash
Function settings:
  • Create function > Select the function trigger type > Cloud Pub/Sub
  • Create a topic > Pub/Sub topic which will trigger this cloud function > Save
  • Give it an appropriate name and click create topic
  • Connections tab > Allow internal traffic only
  • Runtime environement > Python 3.9 > Deploy
```

*You may check the code within the folder > `CODE` > `api_connection` > `Twitch_puller.py` to see how we extract the data and then push it to RDS.


### 1.5. AWS - RDS

[AWS RDS](https://github.com/awslabs/rds-support-tools) is allowing us to have a self-managed and scaled database in the cloud to retrieve this instant data from Twitch.

We used [AWS Academy Learner Labs](https://awsacademy.instructure.com/)

```bash
RDS settings:
	• select Standard create
	• Engine options > PostgreSQL
	• Template > Free tier
	• Settings
		master username: provided below
		master password: provided below
		database cluster: French,France
```

### 1.6. AWS - PostgreSQL

Last  but not least, we are pushing this temporary data to [PostgreSQL](https://github.com/awslabs/rds-support-tools/tree/main/postgres) which is accumulating this data to previous retrieval to form an entire database online and easily accessible. 

We used [AWS Academy Learner Labs](https://awsacademy.instructure.com/)

```bash
Postgres settings
	• master password: provided below
	• Add New Server
		General > Name > Twitch
		Connection > Host > AWS RDS instance end point > twitch.caampywfg0rz.us-east-1.rds.amazonaws.com
					 Port > default
					 Maintenance database > name of our database from AWS RDS > ex: Twitch
					 Username > Master username from AWS RDS
					 Password > Master password from AWS RDS
					 Save
```


### 1.7. Tableau

Finally, from this PostGreSQL database, we are able to connect to it with our [Tableau Dashboard](https://dub01.online.tableau.com/t/twitchgamerecommandations/views/Twitch_Game_Recommandation/GameStatisticsDashboard) to retrieve an overall view of the database and find some great insights on the extracted data and for our algorithm to generate the recommendations in a Jupyter notebook. The outcome of analysis will be also published on Tableau for visualization and user-friendly interactions.

Click here to view our [Tableau Dashboard](https://dub01.online.tableau.com/t/twitchgamerecommandations/views/Twitch_Game_Recommandation/GameStatisticsDashboard)

```bash
Tableau settings:
  • made locally with Student license
  • connected to the PostegreSQL database in "live" mode
  • push the dashboard into Tableau Cloud
  • published the dashboard > to be viewed only
```

### 1.8 Code


If you wish to see the code, you may check the three folders: `single_app` , `Tableau`, `website`

Application\
• Open the folder `single_app` in your python environment\
• Please run the following code in terminal before running the `single_app.py` notebook:
```bash
  pip install -r requirements.txt
```

Then you have an explanation of the code below:

```
from MAB import MAB
from train import Train

train_data = Train().run()
model.train(final_data)
```

The training consist of 3 parts
1. Fetching data from data base: fetch_data.py 
2. Cleaning and processing data: process_data.py
3. Training Multi Armed Bandits: MAB.py

### 1.8.1 fetch_data.py

class DB
- connect: establishes connection with the SQL DB
- postgresql_to_dataframe: accepts SQL query as input and fetches data from the SQL DB

### 1.8.2 process_data.py

class ProcessStreamData
- clean_time_fields: converts time fields to numpy dattime64
- convert_maturity_ratings_to_float: converts maturity boolean to float (1 mature content, 0 not mature)
- filter_for_language: filters for only english streams
- create_time_chunks: maps log time hour to a integer between 0 and 5 (inclusive)
- get_sentiment: maps the stream title to sentiment score
- perform_feature_engineering: aggregates stream level data to game level data and generates the 10 dim feature vector


### 1.8.3 MAB.py

class MAB
- init_train_data: accepts training data as input, generates the mapping from gamenames to arms and games applicable for each timeslot
- train: for each timeslot uses standard sclaer to scalarize data, samples preferences and calls the training algorithm
- lin_ucb: MAB trianing algorithm


### 1.9 Inference


```
import pandas as pd
from predict import Inference
from MAB import MAB
import numpy as np

date_time = pd.to_datetime(np.datetime64('now'))
data, timesplit = Inference().fetch_data(date_time)

preference=np.array([1./6.]*6)
prediction_dict = MAB().predict(data[data['time_logged_encoded'] == 1].reset_index(drop=True),
           timesplit = timesplit,
        preference=preference
)

```

The inference of 3 parts
1. Fetching data from data base: fetch_data.py 
2. Cleaning and processing data: process_data.py
3. Predicting: MAB.py

Parts 1 and 2 are similar to training.

### 1.9.1 Prediction MAB.py

class MAB
- predict: loads the pretrained models and metadata from Resources and generates the predictions 



### 1.10 Website

The website was coded under `CODE` > `website` > `uniDash` > `pages`  with all the `.cshtml` files

The dashboards were embeded directly into the `Stats.cshtml` and `Recommendations.cshtml`

`Recommendations.cshtml` will push the `main.py` to the server that will run the algorithm to push the outputs into the PostgreSQL within `user_recommendation` table.

## Installation

### 2.1. Tokens

The tokens have been provided in the file directly. They will be change after the grading period.


### 2.2. Environment

If you wish to run the codes, you would need the below specifications.

Requires Python 3.9
```bash
  python --version
```



### 2.3. Tableau Dashboard

Within the created website, we have embbed a trial tableau cloud account that will last until Dec 12th (after grading period).

However, if you are facing an issue view the dashboard, you can still connect to Tableau Cloud with your own account to see the two dashboard: `Twitch_Game_Statistics.twbx` & `Twitch_Game_Recommendations.twbx`

When opening the [Twitch_Game_Statistics](https://dub01.online.tableau.com/t/hugodupouy/views/Twitch_Game_Statistics/GameStatisticsDashboard) or [Twitch_Game_Recommendations](https://dub01.online.tableau.com/t/hugodupouy/views/Twitch_Game_Recommendations/GameRecommandationResults), if the PostgreSQL ask you for the credentials to refresh the database, please enter the following in the general tab:

| Parameter | Input     |
| :-------- | :------- | 
| `Server` | `twitch.caampywfg0rz.us-east-1.rds.amazonaws.com` | 
| `Port` | `5432` | 
| `Database` | `Twitch` | 
| `Authentication` | `Username and Password` |
| `Username` | `GaTech_team_96` | 
| `Password` | `i-love-my-coffee-without-milk-and-sugar-at-800AM` | 

*if you wish to see the tables from PostgreSQL, you can use the same credentials in the software [pgAdmin 4 v6](https://www.pgadmin.org/download/) \
  • enter the credentials above
  • click on the `Twitch` arrow > `Schemas` > `Tables`\
  • otherwise, you can make SQL requires to the database by clicking on the `>_` logo in the Browser menu

## Execution

### 3.1. Website

1. Click on our [Website](http://unidash.thebatcave.click) \
  • disable any adblockers as it may affect the viewing and display of the dashboards  \
  • click on `login` > on the top right corner \
  • use the below credentials

      | Parameter | Input     |
      | :-------- | :------- | 
      | `username` | `grading@gatech.edu` | 
      | `Password` | `Fullmarks100%` | 


2. You can access to our `Game Statistics Dashboard` on our [Website - Game Stats](http://unidash.thebatcave.click/Stats)\
  • use the below credentials if Tableau prompt you
      | Parameter | Input     |
      | :-------- | :------- | 
      | `username` | `deangarmwork@gmail.com` | 
      | `Password` | `Fullmarks100%` | 

  • if you cannot view on the website you can see it in Tableau Cloud @ [GameStatisticsDashboard](https://dub01.online.tableau.com/t/hugodupouy/views/Twitch_Game_Statistics/GameStatisticsDashboard)\
  • Otherwise, you may view it in the folder > CODE > Tableau > Twitch_v1


3. You can access to our `Game Recommendation Dashboard` on our [Website - Game Recommendation](http://unidash.thebatcave.click/Recommendations)\
  • use the below credentials if Tableau prompt you
      | Parameter | Input     |
      | :-------- | :------- | 
      | `username` | `deangarmwork@gmail.com` | 
      | `Password` | `Fullmarks100%` | 
  • if you cannot view on the website you can see it in Tableau Cloud @ [GameRecommendationDashboard](https://dub01.online.tableau.com/t/hugodupouy/views/Twitch_Game_Recommendations/GameRecommandationResults)\
  • Otherwise, you may view it in the folder > CODE > Tableau > Twitch_v2

\
if you need to enter the logins for the database, please refer to [Tableau Dashboard](#23-tableau-dashboard)
if you wish the sithe code for the website, you may check the folder `website`






## Authors

- [@hugodup](https://github.com/hugodup)
- [@parthpahwa1](https://github.com/parthpahwa1)
- [@deangarmcode](https://github.com/deangarmcode)
- [@blackjacc](https://github.com/blackjacc)
- [@lliu442](https://github.com/lliu442)
