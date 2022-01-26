import os
import numpy as np
from statistics import mean

directory_actual = 'Actual_Anns/footage2'

directory_predicted = 'Predicted_Anns_Faster/footage2'

actual_count = []
pred_count = []

for filename in os.listdir(directory_actual):
    f = os.path.join(directory_actual, filename)
    with open(f, "r") as infile:
        actual_count.append(len(infile.readlines()))
        
for filename in os.listdir(directory_predicted):
    f = os.path.join(directory_predicted, filename)
    with open(f, "r") as infile:
        pred_count.append(len(infile.readlines()))

avg = []

for i in range(len(actual_count)):
    if actual_count[i] == 0:
        continue
    diff = abs(pred_count[i] - actual_count[i])
    avg.append(float(diff/actual_count[i]))
    
score = 1 - mean(avg)

print(score)