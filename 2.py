#마트 장보기 프로젝트 (클래스 사용)

class Mart:
    def __init__(self, no, item, price, cnt):
        self.no = no
        self.item = item
        self.price = price
        self.cnt = cnt

    def __str__(self):
        return "{}{}{}{}".format(self.no, self.item, self.price, self.cnt)

Mart1 = [0,1,2,3]

Mart1[0] = Mart(1, '아이템', 1500, 3)
print(Mart1[0].no)
