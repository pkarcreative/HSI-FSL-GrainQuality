# -*- coding: utf-8 -*-
"""
Created on Sun Aug 20 17:44:13 2023

@author: pkarmakar
"""

import requests
import pandas as pd
import os


df = pd.read_csv('HSI_GT_Source.csv')
source = df['source']

store_path = " "  # provide path to store the HSI data


"""
#Uncomment this section if file downloading stopped and you need to resume downloading
#This code section will read whatever files are already downloaded and continue downloading the rest of files

file_list = [f for f in os.listdir(store_path) if os.path.isfile(os.path.join(store_path, f))]

file_original_names=[]
for f in file_list:
    f=f.split("_")
    modified_f=[]
    for i, word in enumerate(f):
        if i == 3 or i == 6 or i==8 or i==9 or i==11:
            modified_f.append(word + "/")
        elif i ==len(f)-1:
            modified_f.append(word)
        else:
            modified_f.append(word + "_")
    f_original = result_string = "".join(modified_f)
    file_original_names.append(f_original)


source_left=source[~source.isin(file_original_names)]
source=source_left
"""

for s in source:

    file_url = s
    # prefix need to be embedded before the file_url (file names) to obtain the actual source of the data
    prefix = 'https://erda.ku.dk/archives/89a8b2b044d458e487fc2ce56927f420/'
    prefixed_file_url = prefix + file_url

    # Send a GET request to download the file
    response = requests.get(prefixed_file_url)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # In the file_url, name consist of "/", it can cause error so replace it with "_"
        file_name = file_url.replace("/", "_")

        save_file_name = store_path + file_name

        # Save the content of the response to a local file
        with open(save_file_name, 'wb') as file:
            file.write(response.content)
        print(f"File '{save_file_name}' downloaded successfully.")
    else:
        print(f"Failed to download the file. Status code: {response.status_code}")
