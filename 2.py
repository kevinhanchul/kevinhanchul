#마트 장보기 프로젝트 (클래스 사용)

class Mart:
    def __init__(self, no, item, price, cnt):
        self.no = no
        self.item = item
        self.price = price
        self.cnt = cnt

    def __str__(self):
        return "순번 : {0}, 품목 : {1}, 가격 : {2}, 개수 : {3}".format(self.no, self.item, self.price, self.cnt)

Mart1 = []

# Mart1.append(5)

Mart1.append(Mart(1, '아이템', 1500, 3))
Mart1.append(Mart(2, '아이템2', 2500, 2))

print(Mart1)



