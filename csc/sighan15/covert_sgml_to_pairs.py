#coding=utf-8
from bs4 import BeautifulSoup
file='sample.txt'
with open(file) as f:
    data=f.read()
    # data=data.replace('\n',' ')
    soup = BeautifulSoup(data,features="html.parser")
    print(soup.essay['title'])
    for item in soup.find_all('passage'):
        print(item)
    