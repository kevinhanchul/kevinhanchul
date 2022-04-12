#네이버 영화 리뷰 안에 들어가서 리뷰들의 시퀀스 가져오기

import requests
from bs4 import BeautifulSoup

#clickcr(this, 'rli.uid', '', '', event); showReviewListByNid('4801966');
url = 'https://movie.naver.com/movie/bi/mi/review.naver?code=176037'
headers = {"User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/98.0.4758.82 Safari/537.36"}
res = requests.get(url, headers=headers)
res.raise_for_status()


a = BeautifulSoup(res.text, "lxml")
# print(a)

a = a.find('span',{'class':'user'})
a = a.a['onclick'][-10:-3]
print(a)
