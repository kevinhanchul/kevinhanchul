import requests

from bs4 import BeautifulSoup

​

# url = "https://comic.naver.com/webtoon/genre?genre=omnibus"

url = "http://localhost:8080/aaa.html"

​

​

headers = {"User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/98.0.4758.82 Safari/537.36"}

res = requests.get(url, headers=headers)

​

res.raise_for_status()

res.encoding='UTF-8'

​

soup = BeautifulSoup(res.text, 'lxml')

​

ab = soup.find_all('li')

​

print(ab)