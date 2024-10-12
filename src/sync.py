import csv
import os
import time
import datetime
import json
import sys
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from tinkerforge.ip_connection import IPConnection

GDOCS_OAUTH_JSON = "../secret-keys/trash-project-2024-10b0ff6b629e.json"
GDOCS_SPREADSHEET_NAME = "Data Backup"
# FREQUENCY_SECONDS = 30
# HOST = "localhost"
# PORT = 4223

def login_open_sheet(oauth_key_file,spreadsheet):
    # try:
    scope = ['https://spreadsheets.google.com/feeds','https://www.googleapis.com/auth/drive']
    credentials = ServiceAccountCredentials.from_json_keyfile_name(oauth_key_file,scope)
    gc=gspread.authorize(credentials)
    worksheet=gc.open(spreadsheet).sheet1
    return worksheet

    # except Exception as ex:
    print ('Unable to login and get spreadsheet')
    print('Google sheet login failes with error',ex)
    sys.exit(1)

def upload_data():
    worksheet = None

    if worksheet is None:
        worksheet=login_open_sheet(GDOCS_OAUTH_JSON,GDOCS_SPREADSHEET_NAME)


    daten= datetime.datetime.now().strftime("%Y/%m/%d")
    timen= datetime.datetime.now().strftime("%H:%M:%S")
    data = []
    with open('data-temp.csv', newline='') as f:
        reader = csv.reader(f)
        data = list(reader)
     
    try:
        values = [daten + " " + timen]
        worksheet.append_row(values=values+data[0])

    except:
        print('failed')
        worksheet = None

if __name__ == "__main__":
    upload_data() 
