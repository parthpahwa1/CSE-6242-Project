# Connection parameters to login
import psycopg2
import pandas as pd
import numpy as np
import psycopg2
import pandas as pd
import numpy as np
import sys

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