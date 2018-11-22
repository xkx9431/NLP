import requests
import re
from bs4 import BeautifulSoup
url = 'http://baike.baidu.com/item/%E5%8C%97%E4%BA%AC%E5%9C%B0%E9%93%81/408485'
resp=requests.get(url)
html=resp.content
bs=BeautifulSoup(html,'html5lib')
#print(bs.prettify())
def test():
    pass
test()