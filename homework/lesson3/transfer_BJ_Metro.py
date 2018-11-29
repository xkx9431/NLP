# coding for ‘Transfer of Beijing Metro’ with different agent
import re
import collections
import matplotlib
import networkx as nx

## get line and station informations from 'bj_raw.txt'by crwal
bj_sub=open('bj_raw.txt',encoding='utf-8').read()
bj_sub=bj_sub.split('\n')
count=0
counts=[]
for e in bj_sub:
    if e.find('线')!=-1:
        counts.append(count)
    count+=1
BJ_sub_dict={} # dict{linename:[stations1,stations2],....}
for i in range(len(counts)-1):
    BJ_sub_dict[bj_sub[counts[i]]]= bj_sub[counts[i]+1:counts[i+1]]

def successor(subway_dict):
    successors=collections.defaultdict(dict)
    for linename,stations in subway_dict.items():
        for a,b in overlapping(stations):
            successors[a][b]= linename
            successors[b][a]= linename
    return successors
def overlapping(items):
    return [items[i:i+2] for i in range(len(items)-1)]

BJ_SUB=successor(BJ_sub_dict)

def count_trans_line_num(path):
    #count the num of total transfer lines,for paths=[[linename,station,],...]
    count=1
    temp=path[1]
    for linename in path[1::2]:
        if temp!=linename:
            temp=linename
            count+=1
    return count

#不同换乘策略
def sort_paths(paths,func):
    return sorted(paths,key=func)

def shortest_trans(paths):# 最少车站数
    return sorted(paths,key=lambda p:len(p))

def less_trans(paths):# 最少换乘
    return sort_paths(paths,count_trans_line_num)

def comprehensive_trans(paths):#综合优先
    return sort_paths(paths,lambda p:(2*len(p)-6*count_trans_line_num(p)))


def count_by_way(a,b): # 经过站[,] 在线路中，使该线路靠前排序（count 越小）
    count = 0
    for i in a:
        if i in b:
            count += 1
    return count

def by_way_trans(paths,stations):#经过站换乘
    return sort_paths(paths,lambda p:(len(p)-10000*count_by_way(stations,p)))

def is_goal(destination,pathes,by_way):
    if by_way:
        for e in by_way:
            if e not in pathes:
                return False
    elif pathes[-1] == destination:
        return True
    else:
        return False




def transfer(start,destination,successor,trans_agent,by_way=[]):
    frontier=[[start]]
    explored=set()
    chosen_paths=[]
    while frontier:
        paths=frontier.pop(0)
        s=paths[-1]
        for (station,line) in successor[s].items():
            if station not in explored:
                explored.add(station)
                new_paths = paths + [line, station]
                if station==destination:
                    return new_paths
                else:
                    frontier.append(new_paths)
        if not by_way:
            frontier = trans_agent(frontier)
        else:
            frontier = trans_agent(frontier,by_way)
    return chosen_paths


def parse_results(r):
    current=r[1]
    stations=[]
    result=[]
    a=r[::2]
    b=r[1::2]
    for i in range(len(b)):
        if current==b[i]:
            stations=stations+[a[i]]
        else:
            result=result+[[current]+stations+[a[i]]+[b[i]]]
            stations=[a[i]]
            current=b[i]
    result = result + [[current]+stations+[r[-1],'到达目的地']]

    for i in result[:-1]:
        print('{}:经过站{}，换乘{}'.format(i[0], i[1:-1], i[-1]))
    print('{}:经过站{},{}'.format(result[-1][0], result[-1][1:-1], result[-1][-1]))
    print('\n')
    return

def test():
    result1 = transfer('古城', '安定门', BJ_SUB, less_trans)
    result2 = transfer('古城', '安定门', BJ_SUB, shortest_trans)
    result3 = transfer('古城', '安定门', BJ_SUB, comprehensive_trans)
    result4 = transfer('古城', '安定门', BJ_SUB, by_way_trans,['西单','大钟寺'])

    print('###start=\'古城\',destination=\'安定门\',换乘策略：最少换乘数###')
    parse_results(result1)
    print('###start=\'古城\',destination=\'安定门\',换乘策略：最短距离（最小车站数目）###')
    parse_results(result2)
    print('###start=\'古城\',destination=\'安定门\',换乘策略：综合优先###')
    parse_results(result3)
    print('###start=\'古城\',destination=\'安定门\',换乘策略：经过站[\'西单\',\'大钟寺\']###')
    parse_results(result4)
test()










