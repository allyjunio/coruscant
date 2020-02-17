#!/usr/bin/env python


# call user.lookup api to query a list of user ids.
import sys
import json



# Contains the output json file
resultfile = open('data.json', 'wt')

data = []
with open('tweet.json') as f:
    for line in f:
        data.append(json.loads(line))

resultfile.write(json.dumps(data))
resultfile.close()
