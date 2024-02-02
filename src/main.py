import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk
import json

file = open("./dataset/glassdoor_reviews.json", 'r', encoding='utf8')
dataset = json.load(file)

print("Dataset Size: %d" % (len(dataset)))
print(f"Sample Item: {dataset[0]}")
