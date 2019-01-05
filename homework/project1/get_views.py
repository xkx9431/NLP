# coding: utf-8
import os
from pyltp import SentenceSplitter
from pyltp import Segmentor
from pyltp import Postagger
from pyltp import NamedEntityRecognizer
from pyltp import Parser
from pyltp import SementicRoleLabeller
import re
# import processHandler
import pyltp

# pyltp官方文档http://pyltp.readthedocs.io/zh_CN/develop/api.html#id15
# http://blog.csdn.net/MebiuW/article/details/52496920
# http://blog.csdn.net/lalalawxt/article/details/55804384

LTP_DATA_DIR = 'E:\\NLP\\LTP\\LTP-project\\ltp_data_v3.4.0'  # ltp模型目录的路径
cws_model_path = os.path.join(LTP_DATA_DIR, 'cws.model')  # 分词模型路径，模型名称为`cws.model`
pos_model_path = os.path.join(LTP_DATA_DIR, 'pos.model')  # 词性标注模型路径，模型名称为`pos.model`
ner_model_path = os.path.join(LTP_DATA_DIR, 'ner.model')  # 命名实体识别模型路径，模型名称为`pos.model`
par_model_path = os.path.join(LTP_DATA_DIR, 'parser.model')  # 依存句法分析模型路径，模型名称为`parser.model`
srl_model_path = os.path.join(LTP_DATA_DIR, 'pisrl.model')  # 语义角色标注模型目录路径，

# 分句，也就是将一片文本分割为独立的句子
def sentence_splitter(news='你好，你觉得这个例子从哪里来的？'):
    news = news.replace('\n','')
    sents = SentenceSplitter.split(news)  # 分句
    SentenceSplitter
    #print('\n'.join(sents))
    return sents

def split_sents( content):
    return [sentence for sentence in re.split(r'[？?！!。；;：:\n\r]', content) if sentence]
# 分词
def segmentor(sentence=None):
    segmentor = Segmentor()  # 初始化实例
    segmentor.load(cws_model_path)  # 加载模型
    words = segmentor.segment(sentence)  # 分词
    # 转换成List 输出
    words_list = list(words)
    segmentor.release()  # 释放模型
    return words_list
#词性标注
def posttagger(words):
    postagger = Postagger()  # 初始化实例
    postagger.load(pos_model_path)  # 加载模型
    postags = postagger.postag(words)  # 词性标注
    # for word, tag in zip(words, postags):
    #     print(word + '/' + tag)
    postagger.release()  # 释放模型
    return postags

#命名实体
def ner(words, postags):
    recognizer = NamedEntityRecognizer()
    recognizer.load(ner_model_path)  # 加载模型
    netags = recognizer.recognize(words, postags)  # 命名实体识别
    # for word, ntag in zip(words, netags):
    #     print(word + '/' + ntag)
    recognizer.release()  # 释放模型
    nerttags = list(netags)
    return nerttags

#依存句法分析
def parse(words, postags):
    parser = Parser()  # 初始化实例
    parser.load(par_model_path)  # 加载模型
    arcs = parser.parse(words, postags)  # 句法分析
    # print("\t".join("%d:%s" % (arc.head, arc.relation) for arc in arcs))
    parser.release()  # 释放模型
    return arcs

# # 语义角色标注
def role_label(words, postags,arcs):
    labeller = SementicRoleLabeller()  # 初始化实例
    labeller.load(srl_model_path)  # 加载模型
    roles = labeller.label(words, postags, arcs)  # 语义角色标注
    for role in roles:
        print(role.index + "".join(
            ["%s:(%d,%d)" % (arg.name, arg.range.start, arg.range.end) for arg in role.arguments]))
    labeller.release()  # 释放模型
    return roles

#创建 说的词典
with open('shuo.txt',encoding='utf-8') as f:
    shuo_dict = f.read().splitlines()
    f.close()


def build_parse_child_dict(words, postags, arcs):
    child_dict_list = []
    format_parse_list = []
    for index in range(len(words)):
        child_dict = dict()
        for arc_index in range(len(arcs)):
            if arcs[arc_index].head == index + 1:  # arcs的索引从1开始
                if arcs[arc_index].relation in child_dict:
                    child_dict[arcs[arc_index].relation].append(arc_index)
                else:
                    child_dict[arcs[arc_index].relation] = []
                    child_dict[arcs[arc_index].relation].append(arc_index)
        child_dict_list.append(child_dict)
    rely_id = [arc.head for arc in arcs]  # 提取依存父节点id
    relation = [arc.relation for arc in arcs]  # 提取依存关系
    heads = ['Root' if id == 0 else words[id - 1] for id in rely_id]  # 匹配依存父节点词语
    for i in range(len(words)):
        # ['ATT', '李克强', 0, 'nh', '总理', 1, 'n']
        a = [relation[i], words[i], i, postags[i], heads[i], rely_id[i] - 1, postags[rely_id[i] - 1]]
        format_parse_list.append(a)
    return  child_dict_list, format_parse_list

'''对找出的主语或者宾语进行扩展'''
def complete_e( words, postags, child_dict_list, word_index):
    child_dict = child_dict_list[word_index]
    prefix = ''
    if 'ATT' in child_dict:
        for i in range(len(child_dict['ATT'])):
            prefix += complete_e(words, postags, child_dict_list, child_dict['ATT'][i])
    postfix = ''
    if postags[word_index] == 'v':
        if 'VOB' in child_dict:
            postfix += complete_e(words, postags, child_dict_list, child_dict['VOB'][0])
        if 'SBV' in child_dict:
            prefix = complete_e(words, postags, child_dict_list, child_dict['SBV'][0]) + prefix

    return prefix + words[word_index] + postfix

#
def get_sbv1_(corpus,dict):
    words = segmentor(corpus)
    postags = posttagger(words)
    arcs = parse(words,postags)
    #
    rely_id = [arc.head for arc in arcs]  # 提取依存父节点id
    relation = [arc.relation for arc in arcs]  # 提取依存关系
    heads = ['Root' if id == 0 else words[id - 1] for id in rely_id]  # 匹配依存父节点词语
    #找到谓语
    for i in range(len(words)):
        if relation[i]=='SBV' and heads[i] in dict:
            return[words[i],heads[i]]
    return None
#
def get_sbv2_(corpus,dict):
    words = segmentor(corpus)
    postags = posttagger(words)
    arcs = parse(words,postags)
    child_dict_list, format_parse_list = build_parse_child_dict(words, postags, arcs)
    #找到谓语
    for i in range(len(words)):
        if format_parse_list[i][0]=='SBV' and format_parse_list[i][4] in dict:
            full_sub= complete_e(words, postags, child_dict_list, i)
            shuo_verb = format_parse_list[i][4]
            return [full_sub,shuo_verb]
    return None



def get_views(corpus):
    dict = shuo_dict
    sents=sentence_splitter(corpus)
    views=[]
    for sent in sents:
        if sent and get_sbv1_(sent,dict):
            view = get_sbv1_(sent,dict)+[sent]
            views.append(view)
    return views

corpus1="""
昨日，雷先生说，交警部门罚了他 16 次，他只认了一次，交了一次罚款，拿到法
院的判决书后，会前往交警队，要求撤销此前的处罚。
律师：不依法粘贴告知单
有谋取罚款之嫌。
陕西金镝律师事务所律师骆裕德说，这起案件中，交警部门在处理交通违法的程
序上存在问题。司机违停了，交警应将处罚单张贴在车上，并告知不服可以行使申请
复议和提起诉讼的权利。这既是交警的告知义务，也是司机的知情权利。交警如果这
么做了，本案司机何以被短时间内处罚 16 次后才知晓被罚？程序违法，为罚而罚，没
有起到教育的目的。
"""

corpus2="""
中新网6月23日电 (记者潘旭临) 意大利航空首席商务官乔治先生22日在北京接受中新网记者专访时表示，
意航确信中国市场对意航的重要性，目前意航已将发展中国市场提升到战略层级的高度，
未来，意航将加大在华布局，提升业务水平。
到意大利航空履职仅7个月的乔治，主要负责包括中国市场在内的亚太业务。
乔治称，随着对华业务不断提升，意航明年可能会将每周4班提高到每天一班。同时，意航会借罗马新航站楼启用之际，
吸引更多中国旅客到意大利旅游和转机。
此外，还将加大对北京直飞航线的投资，如翻新航班座椅，
增加电视中有关中国内容的娱乐节目、提高机上中文服务、餐饮服务、完善意航中文官方网站，
提升商务舱和普通舱的舒适度等。
"""

def test(corpus):
    views = get_views(corpus)
    for view in views:
        print(view)
test(corpus2)
