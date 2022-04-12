# 파일에서 리스트 확인

import pandas as pd

column_name = ['a','b','c','d','e','f','g','h']

a = pd.read_excel('/workspace/ai_test.xlsx',
                         header=None, names=column_name)

a = a['a']

# for i in a:
#     print(i)

# print(a[0])
if a[0]>a[1] and a[0]>a[2]:
    print(a[0])
if a[1]>a[0] and a[1]>a[2]:
    print(a[1])
if a[2]>a[0] and a[2]>a[1]:
    print(a[2])

tmp = 0
if a[0]<a[1]:
    tmp = a[0]
    a[0] = a[1]
    a[1] = tmp

if a[1]<a[2]:
    tmp = a[1]
    a[1] = a[2]
    a[2] = tmp

if a[2]<a[3]:
    tmp = a[2]
    a[2] = a[3]
    a[3] = tmp

for i in a:
    print(a)



