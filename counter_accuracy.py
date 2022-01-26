import os
import numpy as np

directory_actual = ''

directory_predicted = ''

actual_count = []
pred_count = []

for filename in os.listdir(directory_actual):
    f = os.path.join(directory_actual, filename)
    with open(f, "r") as infile:
        actual_count.append(infile.readlines())
        
for filename in os.listdir(directory_predicted):
    f = os.path.join(directory_predicted, filename)
    with open(f, "r") as infile:
        pred_count.append(infile.readlines())
        
actual_count = np.array(actual_count)
pred_count = np.array(pred_count)

scores = np.divide(pred_count, actual_count)
score = np.mean(scores)

print(score)