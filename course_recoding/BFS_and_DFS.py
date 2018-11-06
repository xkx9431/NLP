
# Search Based Intelligence
# BFS 广度搜索 DFS 深度优先搜索
# BFS Breadth First Search
# DFS Deep First Search


graph_long = {
    '1': '2 7',
    '2': '3',
    '3': '4',
    '4': '5',
    '5': '6 10',
    '7': '8',
    '6': '5',
    '8': '9',
    '9': '10',
    '10': '5 11',
    '11': '12',
    '12': '11',
}

for n in graph_long:
    graph_long[n] = graph_long[n].split()
#


def search(graph, search_verson):
    found = set()
    to_search = ['1']
    while to_search:
        node = to_search.pop(0)
        if node in found:
            continue
        print(' I am looking at :{}'.format(node))
        found.add(node)
        new_nodes = graph[node]
        to_search = search_verson(new_nodes, to_search)


def LFS(new_nodes, to_search):
    return new_nodes + to_search


def BFS(new_nodes, to_search):
    return to_search + new_nodes


def test():
    print(search(graph_long, LFS))
    print(search(graph_long, BFS))


test()
