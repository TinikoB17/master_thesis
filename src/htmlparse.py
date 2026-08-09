import pandas as pd
import requests


ht = pd.read_html("mirgenedb.html", flavor="bs4")
ht[0].columns = ht[0].columns.get_level_values(-1)

resp = requests.get("https://www.mirbase.org/cgi-bin/mature.pl", params={"acc": "MIMAT0000101"})

resp = requests.get('https://www.mirbase.org/results/?query=MIMAT0000101')

breakpoint()
df = ht[0][['MiRBase ID', 'Family']]

df.to_csv('families.csv', sep='\t', index=False)
