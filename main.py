import pandas as pd
import  matplotlib
import seaborn
import joblib
from sklearn.model_selection import train_test_split
print("Done")
import sqlite3

connection = sqlite3.connect("FPA_FOD_20170508.sqlite")
cursor = connection.cursor()

cursor.execute("SELECT * FROM Fires LIMIT 5")
print(cursor.fetchall())

connection.close()