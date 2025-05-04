import pandas as pd

train = pd.read_csv("labels.csv")
dev = pd.read_csv("full_test_split.csv")

df = pd.concat([train, dev], axis=0)
df = df[["Participant_ID", "PHQ8_Binary"]]
df.columns = ["session_id", "label"]
df.to_csv("final-labels.csv", index=False)
