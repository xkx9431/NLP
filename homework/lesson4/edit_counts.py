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

@memo
def edit_counts(str1,str2):
    other={str1:str2,str2:str1}
    for str in other:
        if len(str)==0:
            return len(other[str])
    return min([edit_counts(str1[:-1],str2)+1,
                edit_counts(str1,str2[:-1])+1,
                edit_counts(str1[:-1],str2[:-1])+(0 if str1[-1]==str2[-1] else 2)])

def test():
    print(edit_counts('beijing','beijin'))
    print(edit_counts('xkx','xukaixuan'))
    print(edit_counts('what\'s up','whats up'))


test()