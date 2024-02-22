from transformers import MLukeTokenizer, LukeModel
import sentencepiece as spm
import torch
import csv
import scipy.spatial
import pandas as pd

csv_filename = "gamedata4_df.csv"
target_people = 4

results2 = []

def extract_people_range(people):
    if '~' in people:
        return [int(value) for value in people.replace('人', '').split('~')]
    else:
        return [int(people.replace('人', ''))]

def is_people_in_range(target_people, people_range):
    target_people = int(target_people)

    if len(people_range) == 1:
        return target_people == people_range[0]
    elif len(people_range) == 2:
        return target_people >= people_range[0] and target_people <= people_range[1]
    else:
        return False

with open(csv_filename, 'r', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    
    for row in reader:
        print(row['people'])
        people_range = extract_people_range(row['people'])
        if is_people_in_range(target_people, people_range):
            result_dict = {key: row[key] for key in ['title','people']}
            results2.append(result_dict)

print(results2)
print(type(results2))