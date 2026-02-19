import pandas as pd 
from datetime import datetime
import csv


def add_data(query,input_type,classification):
    with open("data.csv","a") as file:
        writer = csv.writer(file)
        writer.writerow([query,input_type,classification])
        writer = csv.writer(file, quoting=csv.QUOTE_ALL)
    return 
add_data("I hate, league","text","safe")   

def view_data():
   df= pd.read_csv('data.csv',encoding="latin1")
   return df




