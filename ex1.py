import requests
from bs4 import BeautifulSoup


url = 'https://kin.naver.com/qna/list.naver?dirId=402'
# url = 'https://comic.naver.com/webtoon/weekday'



headers = {"User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/98.0.4758.82 Safari/537.36"}
res = requests.get(url, headers=headers)
res.raise_for_status()
soup = BeautifulSoup(res.text, "lxml")
# print(soup)

a = soup.findAll('td',{'class':'title'})
# a = a.find('a').get_text()
print(a)