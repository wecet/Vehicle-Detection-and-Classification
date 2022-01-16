import random

n = 14500
n_20 = int(n * 0.2)

train = []

for i in range(1, n+1):
    train.append("data/obj/img"+str(i)+".jpg\n")

test = []
r = n
for i in range(n_20):
    c = random.randint(1,r)
    test.append(train.pop(c))
    r -= 1
    
with open("train.txt", "w") as file:
    for t in train:
        file.write(t)
        
with open("test.txt", "w") as file:
    for t in test:
        file.write(t)
