import random
grammar = """
sentence = adj noun verb adv noun2
adj = adj_single adj_single 的 | null
adj_single = 漂亮  | 蓝色 | 好看
adv = 安静地 | 静静
noun = 猫 | 女人 | 男人
verb = adv 看着 | adv 坐着
noun2 = 桌子 | 皮球
"""


def build_grammar(grammar):
    grammar_pattern = {}
    for line in grammar.split('\n'):
        if not line:
            continue
        statement, expr = line.split('=')
        grammar_pattern[statement.strip()] = [e.split()
                                              for e in expr.split('|')]
    return grammar_pattern


# print(build_grammar(grammar))


def generate(grammar_pattern, target):
    if target not in grammar_pattern:
        return target
    expr = random.choice(grammar_pattern[target])
    tokens = [generate(grammar_pattern, e) for e in expr]
    return ''.join([t for t in tokens if t != 'null'])


grammar_pattern = build_grammar(grammar)
sen1=generate(grammar_pattern, 'sentence')
print(sen1)
