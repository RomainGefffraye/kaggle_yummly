import json
from pprint import pprint
from sklearn import tree
import operator
import csv

with open('train.json') as outfile :
    data = json.load(outfile)

with open('test.json') as outfile :
    test = json.load(outfile)

stat = {}
numberOfReceip = {}
relatedCountry = {}
result = {}

for receip in data:
    numberOfReceip[receip["cuisine"]] = numberOfReceip.get(receip["cuisine"], 0) + 1

for receip in data:
    for ingredient in receip["ingredients"]:
        try:
            stat[ingredient]
        except:
            stat[ingredient] = {}
        stat[ingredient][receip["cuisine"]] = stat[ingredient].get(receip["cuisine"], 0.) + 1./numberOfReceip[receip["cuisine"]]

for ingredient in stat :
    sorted_x = sorted(stat[ingredient].items(), key=operator.itemgetter(1))
    relatedCountry[ingredient] = {'country' : str(sorted_x[-1][0]), 'coef': sorted_x[-1][1] }

for receip in test:
    target = {}
    for ingredient in receip["ingredients"]:
        try:
            country = relatedCountry[ingredient]
        except:
            county = "NONE"
        target[country['country']] = target.get(country['country'], 0) + 1
    sorted_x = sorted(target.items(), key=operator.itemgetter(1))
    result[receip["id"]] = sorted_x[-1][0]
    #print str(sorted_x[-1]) + " ID = " + str(receip["id"])



'''
for ingredient in test[1]["ingredients"]:
    try:
        country = relatedCountry[ingredient]
    except:
        county = "NONE"
    target[country['country']] = target.get(country['country'], 0.) + country['coef']
print target
'''


writer = csv.writer(open('dict.csv', 'wb'))
for key, value in result.items():
   writer.writerow([key, value])
