import pandas as pd

train1_file = pd.read_csv('retail50k_train_1.csv')
links = train1_file['ImageUrl']
print(len(set(links)))
with open("train1.txt", 'w') as f:
    [f.write(photo_url + '\n') for photo_url in set(links)]
