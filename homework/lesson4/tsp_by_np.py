import random
import math
import collections
from functools import wraps

def memo(func):
    cache={}
    @wraps(func)
    def wapper(*args,**kwargs):
        key=str(args)+str(kwargs)
        if key not in cache:
            cache[key]=func(*args,**kwargs)
        return cache[key]
    return wapper

# x = [ random.randint(1,100) for _ in range(10)]
# y = [ random.randint(1,100) for _ in range(10)]
# citys=list(zip(x,y))
cities=[(34, 83), (31, 62), (46, 1),(54, 44),
        (90, 7), (15, 74), (86, 44), (90, 36), (16, 71), (36, 42)]

def dist(a,b):
    return math.sqrt((a[0]-b[0])**2+(a[1]-b[1])**2)

def city_remove(cities,k):
    if k not in cities: return 'error'
    new_cities=[]
    for i in cities:
        if i!=k:
            new_cities=new_cities+[i]
    return new_cities

'''
s # 起点
cities   # 需要到达地方的坐标点集合
distance # k 到 s 的距离 
递推方程  # cost（s,cities）= min{cost(k,cities-k) + distance(k,s)} 其中 cities-k 表示从集合cities 移除 k
'''
@memo
def tsp_by_np(s,cities):
    while len(cities)>1:
        return min([tsp_by_np(k,city_remove(cities,k))+ dist(s,k) for k in cities])
    return dist(s,cities[0])

def test():
    print(tsp_by_np((0,0),cities))
test()




