
import pandas as pd
import subprocess
from bs4 import BeautifulSoup
# soup = BeautifulSoup(html_doc, 'html.parser')

accessions_fa = "original_input/high-conf-mirnas.fa"
accessions = []

with open(accessions_fa) as f:
    content = f.readlines()
    for line in content:
        if line.startswith(">"):
            accessions.append(line.split(">")[1].split(" ")[1].strip())

    # print(accessions)



mature_mirs = []
with open("test_output/high_confidence_mirnas.txt", "w") as file:
    for ac in accessions:
        command = ["curl", f"https://www.mirbase.org/hairpin/{ac}"]
        res = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        html_content = res.stdout

        soup = BeautifulSoup(html_content, 'html.parser')

        elements = soup.find_all(class_="mature-name")

        for el in elements:
            mir = el.text
            print(mir)
            mature_mirs.append(mir)
            file.write(mir)
            file.write('\n')
        
