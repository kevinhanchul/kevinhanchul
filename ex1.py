import pandas as pd

a = [1,2,3]
a = pd.DataFrame(a)

# print(a)
# print(a.shape)

#위의 리스트와 타이틀만 틀리고 똑같음
a = {1:[1,2,3]}
a = pd.DataFrame(a)

# print(a)
# print(a.shape)

a = [[1,2],[3,4]]
a = pd.DataFrame(a)

print(a)
print(a.shape)
