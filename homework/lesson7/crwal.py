#coding for crwal
# xkx-2018-12-18

import requests
from requests.exceptions import RequestException
import re
from bs4 import BeautifulSoup

def get_page(url):
    try:
        headers={
                 "User-Agent": "User-Agent:Mozilla/5.0 (compatible; MSIE 9.0;Windows NT 6.1; Trident/5.0;"}
        r = requests.get(url, headers=headers)
        if r.status_code==200:
            return r.content
        return None
    except RequestException :
        return None

url1='http://www.cmiw.cn/forum-60-1.html'
url2='http://www.cmiw.cn/thread-889554-1-1.html'
page1=BeautifulSoup(get_page(url1), 'html5lib')

#获取帖子标题 和 标题的链接
def get_urls(html):
    result=[]
    page=BeautifulSoup(html, 'html5lib')
    urls_raw=page.find_all(class_='s xst')

    for url_raw in urls_raw:
        a=re.findall('.*href=\"(.*.html)\" .*>(.*)</a>',str(url_raw))
        result+=a
    for item in result:
        yield [item[1], get_content(get_page(item[0])),item[0]]


#对帖子内容进行抓取
def get_content(html):
    result=' '
    page = BeautifulSoup(html, 'html5lib')
    topic = page.find_all(class_='t_f')
    next_page=page.find_all(class_="pgbtn")# 判断帖子是否有下一页
    for a in topic:
        b = a.get_text().strip()
        if b :
            result+='\n'.join([e for e in b.split('\n')if e ])
    return result if not next_page \
        else result + get_content(get_page(re.findall('.*href=\"(.*.html)\"',str(next_page))[0]))

#写入文件
def write_to_file(item):
    with open(item[2][26:36]+'.txt','w',encoding='utf-8') as f:
        f.write('topic---'+item[0]+':'+'\n'+item[1])

def main(offset):
    url='http://www.cmiw.cn/forum-60-'+str(offset)+'.html'
    html=get_page(url)
    for item in get_urls(html):
        write_to_file(item)


def test():
    for i in range(1000):
        main(offset=i)
test()


