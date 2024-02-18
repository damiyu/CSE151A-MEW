import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk
import json

# Remove data with 4 or more missing features.

file = open("./dataset/glassdoor_reviews.json", 'r', encoding='utf8')
dataset = json.load(file)