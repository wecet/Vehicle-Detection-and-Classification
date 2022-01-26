import os

directory = 'Annotations/footage2'

count = 0
with open('Full_Annotation/footage2.txt', 'w') as outfile:
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        with open(f) as infile:
            outfile.write(infile.read())