import requests 
import pandas as pd
import os
import json
import argparse
from tabulate import tabulate
import logging

logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)

URL = "https://bxgscnkych.execute-api.us-east-1.amazonaws.com/api/status"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', type=str, default="phase1", choices={"phase1", "phase2", "phase3"})
    args = parser.parse_args()

    auth = {
    "Team": "", # Add Team ID here
    "Token": "", # Add Token here
    "Content-Type": "application/json"
    }
    auth["Stage"] = args.phase
        
    response = requests.get(URL,headers=auth)

    if response.status_code == 401:
        logger.error("Authentication Error, Please check your team-id and key")
    elif response.status_code == 200:
        if len(response.json())==0:
            print("No Record found")
        else:
            df = pd.DataFrame(response.json())
            df["Submitted Time"] =pd.to_datetime(df["Submitted Time"], infer_datetime_format=True) + pd.to_timedelta(5, unit='h')
            df["Submitted Time"] = df["Submitted Time"].astype('datetime64[s]')
            df.rename(columns = {'Submitted Time':'Submitted Time (PKT)'}, inplace = True) 
            df.drop(["Submission Reference", "#"], axis=1, inplace=True)
            df["Team Id"] = auth["Team"]
            print(tabulate(df, headers = 'keys', tablefmt = 'psql', showindex=False))
            
    else:
        logger.error(f"Unknown Error occured with code {response.status_code}")



