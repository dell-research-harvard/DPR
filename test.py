import json

# file = open("C:/Users/Emily/Downloads/newspapers210917.json",)

# data = json.load(file)

# with open("C:/Users/Emily/Downloads/newspapers210917.json") as data:
#    for i in range(0, 20):
#        print(data.readline(), end='')

n_contexts = 50
n_strata = 50

strata_size = round(n_contexts / n_strata)
for l in range(n_strata):
    strata_range = range(l * strata_size, ((l+1) * strata_size) -1)
    print(strata_range)
