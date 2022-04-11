from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import matplotlib.pyplot as plt



# a = sns.load_dataset('titanic')
a = pd.read_excel('/workspace/ai_test.xlsx')
# print(a)

clean_data = raw_data.dropna(axis=1, thresh=500)