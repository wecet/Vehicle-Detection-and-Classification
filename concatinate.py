import os

directory = 'annotations/rainTest'

with open('annotations/rainTest.txt', 'w') as outfile:
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        with open(f) as infile:
            outfile.write(infile.read())