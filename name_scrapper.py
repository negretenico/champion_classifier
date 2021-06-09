from bs4 import BeautifulSoup as bs
import requests
import os


s = f"https://na.leagueoflegends.com/en-us/champions/"
page = requests.get(s)
soup = bs(page.content, 'html.parser')
with open('champs.txt', 'w') as f:
    for champ in soup.find_all('span',attrs={'class': 'style__Text-sc-12h96bu-3 gPUACV'}):
        f.write(champ.text)
        f.write("\n")