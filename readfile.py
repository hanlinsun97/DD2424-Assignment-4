import json
import numpy as np
import os

def readfile():

    Text = []
    for filename in os.listdir("tweet/"):
        filename_new = "tweet/" + filename
        with open(filename_new,'r') as f:
            list_dictionary = json.load(f)
            for i in range(len(list_dictionary)):
                dictionary = list_dictionary[i]
                text = dictionary['text']
                Text.append(text)
    return Text
