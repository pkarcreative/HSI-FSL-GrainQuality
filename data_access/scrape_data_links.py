# -*- coding: utf-8 -*-
"""
Created on Sun Aug 20 17:24:29 2023

@author: pkarmakar
"""

from selenium import webdriver
from bs4 import BeautifulSoup
import pandas as pd

# URL of the web page where HSI data is listed
url = 'https://erda.ku.dk/archives/89a8b2b044d458e487fc2ce56927f420/published-archive.html#'

# Initialize the WebDriver # You may need to have Chrome browser installed on your computer
driver = webdriver.Chrome()

# Load the webpage
driver.get(url)

import time
time.sleep(5)  # Wait for 5 seconds

# Get the page source after the dynamic content has loaded
page_source = driver.page_source

driver.quit()

soup = BeautifulSoup(page_source, 'html.parser')

tbody = soup.find('tbody')

anchor_tags = tbody.find_all('a', href=True)
links_all=[]

for anchor in anchor_tags:
    link = anchor['href']
    links_all.append(link)
    print(link)

df=pd.DataFrame(links_all, columns=["source"])
df.to_csv("HSI_GT_Source.csv",index=True)
