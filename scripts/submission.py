import requests 
import pandas as pd
import os
import json
import argparse
from tabulate import tabulate
import logging

logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)

URL = "https://bxgscnkych.execute-api.us-east-1.amazonaws.com/api/submission_link"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', type=str, default="phase1", choices={"phase1", "phase2", "phase3"})
    parser.add_argument('--json_path', type=str, required=True, help="Path to submission json")
    args = parser.parse_args()

    auth = {
    "Team": "", # Add Team ID here
    "Token": "", # Add Token here
    "Content-Type": "application/json"
    }
    auth["Stage"] = args.phase

    logger.info("Requesting upload link")
    response = requests.get(URL,headers=auth)
    if response.status_code == 401:
        logger.error("Authentication Error, Please check your team-id and key")
        exit()
    logger.info("Upload link generated")

    request_body = response.json()['data']
    requests_url = response.json()['url']

    try:
        myfiles = {'file': open(args.json_path ,'rb')}
    except:
        logger.error("Failed to read the json file.")

    logger.info("uploading this may take a while.")
    response = requests.post(requests_url, data = request_body, files = myfiles)
    if response.status_code == 200 or response.status_code == 204:
        print("Submission successful. Use status.py to check the submission status.")
    else:
        logger.error(f"Unknown Error occured with code {response.status_code}")



        
