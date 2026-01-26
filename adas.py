import numpy as np
import pandas as pd


cars = [f'Car_{i + 1}' for i in range(74 * 5)]
trains = [f'Train_{i + 1}' for i in range(11 * 5)]
pedestrians = [f'Pedestrain_{i + 1}' for i in range(69 * 5)]
feature_list = np.concatenate((cars, trains, pedestrians))

# df = pd.DataFrame(np.nan, index=np.arange(7), columns=feature_list) 
table_data = np.empty((7, len(feature_list)))
table_data[:] = np.nan

a1 = np.asarray([[1, 2, 3, 4, 5],
      [11, 22, 33, 44, 55],
      [111, 222, 333, 444, 555]])

a2 = np.asarray([[-1, -2, -3, -4, -5]])

a3 = np.asarray([[9, 9, 9, 9, 9],
      [8, 8, 8, 8, 8]])

# df['Car_1'][0] = 0
atr1 = a1.T.reshape(-1) 
atr2 = a2.T.reshape(-1)
print(atr1.shape)

table_data[0, 0:len(atr1)] = atr1
table_data[0, 74 * 5:74 * 5 + len(atr2)] = atr2

print(table_data)
df = pd.DataFrame(table_data, columns=feature_list)
df.to_csv("foo.csv")