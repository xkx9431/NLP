import requests
from bs4 import BeautifulSoup
import re

def get_url(url):
    # get 网页
    headers = {
        "User-Agent": "User-Agent:Mozilla/5.0 (compatible; MSIE 9.0; Windows NT 6.1; Trident/5.0;"}
    r = requests.get(url, headers=headers).content.decode('utf8')
    return r

def parse_line(re_style,raws):
    # 提取有用信息，by re
    result = set()
    for raw in raws:
        temp = re.findall(re_style, str(raw))
        if temp:
            if temp[0] not in result:
                result.add(temp[0])
    return result

def parse_station(raws,re_pattern):
    stations_pairs=[]
    stations=[]
    for raw in raws:
        stations_pair=re.findall(re_pattern,str(raw))
        if stations_pair:
            stations_pairs+=stations_pair

    for a,b in stations_pairs:
        stations.append(a)
    if stations_pairs:
        stations.append(stations_pairs[-1][-1])
    return stations


url = 'https://baike.baidu.com/item/%E5%8C%97%E4%BA%AC%E5%9C%B0%E9%93%81/408485'
line_bs = BeautifulSoup(get_url(url), 'html5lib')

# 爬取线路信息
lines_raw = line_bs.find_all('a')
# <a href="/item/%E5%8C%97..." target="_blank">北京地铁1号线</a> '<a .*href="(.*)">(北京地铁\d+|\w+线)</a>'
lines_re_style = '.*href=\"(.*)\" .*>(北京地铁\d+|\w+线)</a>'
lines_url = parse_line(lines_re_style,lines_raw)


# 爬取车站信息
BJ_sub={}

re_str1='<th.*>(\w+)——(\w+)</th>'
for line_url,linename in lines_url:
    line_url_raw = get_url('https://baike.baidu.com'+line_url)
    stations_bs = BeautifulSoup(line_url_raw, 'html5lib')
    stations_raw = stations_bs.find_all('th')
    stations= parse_station(stations_raw,re_str1)
    BJ_sub[linename]=stations



def test():
    fp=open('bj_sub_info.txt','w',encoding='utf8')
    for lineurl,line in lines_url:
        print(line,linename)
        fp.write(line+' : '+ lineurl+'\n')

    for e in BJ_sub:
        print(e,BJ_sub[e],len(BJ_sub[e]))
        fp.write(e+' : '+str(BJ_sub[e])+'\n')
    fp.close()
test()
#/item/%E5%8C%97%E4%BA%AC%E5%9C%B0%E9%93%815%E5%8F%B7%E7%BA%BF
#/item/%E5%8C%97%E4%BA%AC%E5%9C%B0%E9%93%815%E5%8F%B7%E7%BA%BF