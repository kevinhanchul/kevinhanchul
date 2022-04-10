#도서관리 프로젝트

book_nm=[0,1,2,3,4,5,6,7]

book_nm[0] = {'제목':['Cooking Light'], '분류':['living', 'cooking'],
            '가격':[15000], '비고':['America Cooking']}
book_nm[1]= {'제목':['Auto Bild'], '분류':['scientce','car'],
            '가격':[16000], '비고':['Gemany Car']}
book_nm[2] = {'제목':['The Confession'], '저자':['Grisham', 'John'],
            '가격':[10500]}
book_nm[3] = {'제목':['Les Miserables'], '저자':['Hugo','Victor'],
              '가격':[17500]}



a = 1

if a == 1 :
    for i in range(3):
        print(book_nm[i])
#

# if a == 2 :
#     print(book_nm1)
