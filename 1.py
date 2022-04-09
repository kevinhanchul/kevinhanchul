import pandas as pd
import seaborn as sns

import numpy as np


a = {10:[1,2,3], 20:[4,5,6]}

a = pd.DataFrame(a)
a.columns = [100, 200]


print(a)