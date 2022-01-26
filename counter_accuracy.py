import os

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
        
