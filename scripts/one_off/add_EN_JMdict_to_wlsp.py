import pandas as pd
import pickle


with open("data/processed/JMdict.pkl", "rb") as f:
    JMdict = pickle.load(f)

wlsp = pd.read_parquet("data/processed/wlsp.parquet")

def translate(lemma):
    return JMdict.get(lemma, [])

wlsp["EN_JMdict"] = wlsp["lemma"].apply(translate)

wlsp.to_pickle("data/processed/wlsp.pkl")

print(wlsp["EN_JMdict"].head())

