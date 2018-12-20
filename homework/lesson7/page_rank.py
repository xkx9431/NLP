import networkx as nx
import random
from string import ascii_letters
import matplotlib
from matplotlib.pyplot import show
def generate_name():
    return ''.join([random.choice(ascii_letters.upper())for _ in range(3)])

soical_gragh={

    "Yao": ['Guo', 'Wang', 'Tian', 'Tim'] + [generate_name() for _ in range(4)],
    "Guo": ['Li'] + [generate_name() for _ in range(5)],
    "Wang": ["Li_2"] + [generate_name() for _ in range(5)],
    "Li_2": [generate_name() for _ in range(5)],
    "Li": [generate_name() for _ in range(1)]
}

def sorted_graph_with_page_rank(graph):
    return sorted(nx.pagerank(nx.Graph(graph)).items(),key=lambda x:x[1],reverse=True)


def test():
    print(generate_name())
    social_network = nx.draw(nx.Graph(soical_gragh),with_labels = True)
    show()
    print(sorted_graph_with_page_rank(soical_gragh))

test()
