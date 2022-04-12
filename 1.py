#네이버 영화 리뷰

import requests
from bs4 import BeautifulSoup

url = 'https://movie.naver.com/movie/sdb/browsing/bmovie.naver?open=2022'
headers = {"User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/98.0.4758.82 Safari/537.36"}
res = requests.get(url, headers=headers)
res.raise_for_status()


a = BeautifulSoup(res.text, "lxml")
# a = a.find('ul',{'class':'directory_list'}).a['href']

# print(a)
# /movie/bi/mi/basic.naver?code=176037

a1 = a.findAll('ul',{'class':'directory_list'})

for i in a1:
    k = i

for i in k:
    if i.find('a')!=-1:
        print(i.find('a')['href'][-6:])