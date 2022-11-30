# Twitch Game Recommander

In this project, the team proposes to develop a Twitch Game recommendation system based on sentiment analysis and multi-arm bandit in order to provide a point of reference to streamers.
Moreover, we are helping streaming players develop their channel by picking the game, the duration of the stream, and when they shall stream based on the streamer preferences to grow from zero to hero on Twitch.





## Table of Contents

 - [Description](#description)
    - [Twitch Developer console](#1-twitch-developer-console)
    - [Twitch API](#2-twitch-api)
    - [Google Cloud Platform - Scheduler](#3-google-cloud-platform---scheduler)
    - [Google Cloud Platform - Cloud Functions](#4-google-cloud-platform---cloud-functions)
    - [AWS - RDS](#5-aws---rds)
    - [AWS - PostgreSQL](#6-aws---postgresql)
    - [Tableau](#7-tableau)
 - [Installation](#installation)
    - [Tokens](#tokens)
    - [Environment](#environment)
    - [Tableau Dashboard](#tableau-dashboard)
 - [Execution](#execution)






## Description


### 1. Twitch Developer console

We created a [Twitch](https://www.twitch.tv) account and connected to the [Developer console](https://dev.twitch.tv/console).
Within the console, we have setup and application and defined the OAuth Redirect URLs and Category to retrieve the Client ID and the Client Secret code. 
We setup our console as as below:

```http
  https://dev.twitch.tv/console
```

| Parameter | Inputs     | Description                |
| :-------- | :------- | :------------------------- |
| `Name` | `GaTech_team_project096` | Unique project name to pick |
| `OAuth Redirect URLs` | `https://id.twitch.tv/oauth2/token` | choosen |
| `Category` | `Analytics Tool` | choosen |
| `Client ID` | `qvsl21co22jan49fwd6tn0mw5dohjb` | auto-generated by Twitch |
| `Client Secret` | `hidden` | auto-generated by Twitch |


### 2. Twitch API

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

### 3. Google Cloud Platform - Scheduler

[Google Scheduler](https://console.cloud.google.com/products?supportedpurview=project) allows us to to run every hour for a cron job that sends a pub/sub command.

### 4. Google Cloud Platform - Cloud Functions

[Google Cloud Functions](https://console.cloud.google.com/products?supportedpurview=project) helped us to runs the extraction script and saves the information to the AWS Relational Database Service (RDS) database.


### 5. AWS - RDS

[AWS RDS](https://github.com/awslabs/rds-support-tools) is allowing us to have a self-managed and scaled database in the cloud to retrieve this instant data from Twitch.

We use [AWS Academy Learner Labs](https://awsacademy.instructure.com/)

```http
RDS settings:
	• select Standard create
	• Engine options > PostgreSQL
	• Template > Free tier
	• Settings
		master username: GaTech_team_96
		master password: i-love-my-coffee-without-milk-and-sugar-at-800AM
		database cluster: French,France
```

### 6. AWS - PostgreSQL

Last  but not least, we are pushing this temporary data to [PostgreSQL](https://github.com/awslabs/rds-support-tools/tree/main/postgres) which is accumulating this data to previous retrieval to form an entire database online and easily accessible. 

We use [AWS Academy Learner Labs](https://awsacademy.instructure.com/)

```http
Postgres settings
	• master password: i-love-my-coffee-without-milk-and-sugar-at-800AM
	• Add New Server
		General > Name > Twitch
		Connection > Host > AWS RDS instance end point > twitch.caampywfg0rz.us-east-1.rds.amazonaws.com
					 Port > default
					 Maintenance database > name of our database from AWS RDS > ex: Twitch
					 Username > Master username from AWS RDS
					 Password > Master password from AWS RDS
					 Save
```


### 7. Tableau

Finally, from this PostGreSQL database, we are able to connect to it with our [Tableau Dashboard](https://dub01.online.tableau.com/t/twitchgamerecommandations/views/Twitch_Game_Recommandation/GameStatisticsDashboard) to retrieve an overall view of the database and find some great insights on the extracted data and for our algorithm to generate the recommendations in a Jupyter notebook. The outcome of analysis will be also published on Tableau for visualization and user-friendly interactions.

Click here to view our [Tableau Dashboard](https://dub01.online.tableau.com/t/twitchgamerecommandations/views/Twitch_Game_Recommandation/GameStatisticsDashboard)

```http
Tableau settings:
  • made locally with Student license
  • connected to the PostegreSQL database in "live" mode
  • push the dashboard into Tableau Cloud
  • published the dashboard > to be viewed only
```



## Installation

### Tokens

The tokens have been provided in the file directly. They will be change after the grading period.


### Environment

Requires Python 3.11
```http
  python3 --version
```

Please run the following code in terminal before running the `main.ipynb` notebook:
```http
  pip install -r requirements.txt
```


### Tableau Dashboard
When opening the `Twitch_v1.twbx`, if the PostgreSQL ask you for the credentials to refresh the database, please enter the following in the general tab:

| Parameter | Input     |
| :-------- | :------- | 
| `Server` | `twitch.caampywfg0rz.us-east-1.rds.amazonaws.com` | 
| `Port` | `5432` | 
| `Database` | `Twitch` | 
| `Authentication` | `Username and Password` |
| `Username` | `GaTech_team_96` | 
| `Password` | `i-love-my-coffee-without-milk-and-sugar-at-800AM` | 

```http
Tableau:
  • Data Source tab > refresh the data on the top right corner
  • connected to the PostegreSQL database in "live" mode
  • push the dashboard into Tableau Cloud
  • published the dashboard > to be viewed only
```

## Execution

nothing yet






## Authors

- [@hugodup](https://github.com/hugodup)
- [@parthpahwa1](https://github.com/parthpahwa1)
- [@deangarmcode](https://github.com/deangarmcode)
- [@blackjacc](https://github.com/blackjacc)
- [@lliu442](https://github.com/lliu442)