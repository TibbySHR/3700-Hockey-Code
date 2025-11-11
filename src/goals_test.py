import pandas as pd
df = pd.read_csv("data/clean/events_2016_2019_with_gsec.csv")
print(df["eventType"].value_counts())
