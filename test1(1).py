import numpy as np
import random
import matplotlib

def generate_bernulli(p):
    random_nums=random.random()
    if random_nums<=p:
        return 1
    else:
        return 0
sum=0
nums=10000
for _ in range(nums):
    #print(generate_bernulli(0.5)) 
    random_bonulli= np.random.binomial(1,0.5)
    print(random_bonulli)
    
    sum+=random_bonulli
print(sum/nums)
