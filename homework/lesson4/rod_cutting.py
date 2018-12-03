from collections import defaultdict
import functools
import random

prices = defaultdict(lambda: -float('inf'))
for i, v in enumerate([1, 5, 8, 9, 10, 17, 17, 20, 24, 30]):
    prices[i+1] = v

def memo(func):
    cache={}
    @functools.wraps(func)
    def _wrap(*args,**kwargs):
        str_key=str(args)+str(kwargs)
        if str_key not in cache:
            result=func(*args,**kwargs)
            cache[str_key]=result
        return cache[str_key]
    return _wrap

solution={}

@memo
def revenue(r):
    split, r_star = max([(0, prices[r])] + [(i, revenue(r - i) + revenue(i))
                                            for i in range(1, r)], key=lambda x: x[1])
    solution[r]=(split,r-split)
    return r_star

def parse_solution(solution,r):
    return [solution[r][1]] if solution[r][0]==0 else [solution[r][0]] + parse_solution(solution,solution[r][1])

def how_to_cut(r):
    revenue(r)
    raw=parse_solution(solution,r)
    print('the best strategy to cut the rod of {} is:'.format(r))
    print('-->'.join(map(str,raw)))


def test():
    how_to_cut(79)
    how_to_cut(17)
    how_to_cut(99)
test()