from SPARQLWrapper import SPARQLWrapper, JSON
from utils import *

SPARQLPATH = "http://192.168.80.12:8890/sparql"  # depend on your own internal address and port, shown in Freebase folder's readme.md

# pre-defined sparqls
sparql_head_relations = """\nPREFIX ns: <http://rdf.freebase.com/ns/>\nSELECT ?relation\nWHERE {\n  ns:%s ?relation ?x .\n}"""
sparql_tail_relations = """\nPREFIX ns: <http://rdf.freebase.com/ns/>\nSELECT ?relation\nWHERE {\n  ?x ?relation ns:%s .\n}"""
sparql_tail_entities_extract = """PREFIX ns: <http://rdf.freebase.com/ns/>\nSELECT ?tailEntity\nWHERE {\nns:%s ns:%s ?tailEntity .\n}"""
sparql_head_entities_extract = """PREFIX ns: <http://rdf.freebase.com/ns/>\nSELECT ?tailEntity\nWHERE {\n?tailEntity ns:%s ns:%s  .\n}"""
sparql_id = """PREFIX ns: <http://rdf.freebase.com/ns/>\nSELECT DISTINCT ?tailEntity\nWHERE {\n  {\n    ?entity ns:type.object.name ?tailEntity .\n    FILTER(?entity = ns:%s)\n  }\n  UNION\n  {\n    ?entity <http://www.w3.org/2002/07/owl#sameAs> ?tailEntity .\n    FILTER(?entity = ns:%s)\n  }\n}"""


def check_end_word(s):
    # 检查字符串s是否以给定的单词之一结尾
    # 参数：
    #   s (str): 要检查的字符串
    # 返回值：
    #   bool: 如果字符串s以给定的单词之一结尾，则返回True，否则返回False
    words = [" ID", " code", " number", "instance of", "website", "URL", "inception", "image", " rate", " count"]
    return any(s.endswith(word) for word in words)


def abandon_rels(relation):
    """
    判断给定的关系是否需要被放弃。

    参数：
    relation (str): 关系字符串。

    返回值：
    bool: 如果需要放弃关系，则返回True；否则返回False。
    """
    if relation == "type.object.type" or relation == "type.object.name" or relation.startswith(
            "common.") or relation.startswith("freebase.") or "sameAs" in relation:
        return True


def execurte_sparql(sparql_query):
    # 创建SPARQLWrapper对象
    sparql = SPARQLWrapper(SPARQLPATH)
    # 设置查询语句
    sparql.setQuery(sparql_query)
    # 设置返回格式为JSON
    sparql.setReturnFormat(JSON)
    # 执行查询并转换结果
    results = sparql.query().convert()
    # 返回结果中的bindings部分
    return results["results"]["bindings"]


def replace_relation_prefix(relations):
    """
    替换关系前缀的函数

    参数：
    relations (list): 包含关系字典的列表

    返回值：
    list: 替换前缀后的关系列表
    """
    return [relation['relation']['value'].replace("http://rdf.freebase.com/ns/", "") for relation in relations]


def replace_entities_prefix(entities):
    """
    替换实体前缀的函数

    参数：
    entities (list): 包含实体的列表

    返回值：
    list: 替换前缀后的实体列表
    """
    return [entity['tailEntity']['value'].replace("http://rdf.freebase.com/ns/", "") for entity in entities]


def id2entity_name_or_type(entity_id):
    """
    根据实体ID查询实体名称或类型

    参数：
    entity_id (str): 实体ID

    返回值：
    str: 实体名称或类型

    """
    # 构建SPARQL查询语句
    sparql_query = sparql_id % (entity_id, entity_id)

    # 创建SPARQLWrapper对象
    sparql = SPARQLWrapper(SPARQLPATH)

    # 设置查询语句
    sparql.setQuery(sparql_query)

    # 设置返回格式为JSON
    sparql.setReturnFormat(JSON)

    # 执行查询并转换结果
    results = sparql.query().convert()

    # 如果查询结果为空，则返回"UnName_Entity"
    if len(results["results"]["bindings"]) == 0:
        return "UnName_Entity"
    else:
        # 返回第一个绑定的实体名称或类型
        return results["results"]["bindings"][0]['tailEntity']['value']


from freebase_func import *
from prompt_list import *
import json
import time
import openai
import re
from prompt_list import *
from rank_bm25 import BM25Okapi
from sentence_transformers import util
from sentence_transformers import SentenceTransformer


def clean_relations(string, entity_id, head_relations):
    # 定义正则表达式模式，用于匹配关系和分数
    pattern = r"{\s*(?P<relation>[^()]+)\s+\(Score:\s+(?P<score>[0-9.]+)\)}"
    # 初始化关系列表
    relations = []
    # 遍历匹配结果
    for match in re.finditer(pattern, string):
        # 获取关系和分数
        relation = match.group("relation").strip()
        if ';' in relation:
            continue
        score = match.group("score")
        # 如果关系或分数为空，则返回False和"output uncompleted.."
        if not relation or not score:
            return False, "output uncompleted.."
        # 尝试将分数转换为浮点数
        try:
            score = float(score)
        # 如果转换失败，则返回False和"Invalid score"
        except ValueError:
            return False, "Invalid score"
        # 如果关系在head_relations中，则将关系添加到列表中，并标记为head=True
        if relation in head_relations:
            relations.append({"entity": entity_id, "relation": relation, "score": score, "head": True})
        # 否则，将关系添加到列表中，并标记为head=False
        else:
            relations.append({"entity": entity_id, "relation": relation, "score": score, "head": False})
    # 如果没有找到关系，则返回False和"No relations found"
    if not relations:
        return False, "No relations found"
    # 如果成功找到关系，则返回True和关系列表
    return True, relations


def if_all_zero(topn_scores):
    """
    判断给定的topn_scores列表中的所有元素是否都为0。
    :param topn_scores: 一个包含分数的列表
    :return: 如果所有分数都为0，则返回True；否则返回False
    """
    return all(score == 0 for score in topn_scores)


def clean_relations_bm25_sent(topn_relations, topn_scores, entity_id, head_relations):
    """
    清理BM25句子关系

    参数：
    topn_relations: list，前n个关系列表
    topn_scores: list，前n个关系得分列表
    entity_id: str，实体ID
    head_relations: list，头部关系列表

    返回值：
    True: 成功清理关系
    relations: list，清理后的关系列表
    """
    relations = []  # 初始化关系列表
    if if_all_zero(topn_scores):  # 如果所有得分都为0
        topn_scores = [float(1 / len(topn_scores))] * len(topn_scores)  # 将得分设置为1/关系数量
    i = 0  # 初始化索引
    for relation in topn_relations:  # 遍历前n个关系
        if relation in head_relations:  # 如果关系在头部关系列表中
            relations.append({"entity": entity_id, "relation": relation, "score": topn_scores[i],
                              "head": True})  # 添加关系到列表中，头部标记为True
        else:
            relations.append({"entity": entity_id, "relation": relation, "score": topn_scores[i],
                              "head": False})  # 添加关系到列表中，头部标记为False
        i += 1  # 索引加1
    return True, relations  # 返回成功清理关系和清理后的关系列表


def construct_relation_prune_prompt(question, entity_name, total_relations, args):
    """
    构建关系剪枝提示

    参数：
    question (str): 问题
    entity_name (str): 实体名称
    total_relations (list): 总共的关系列表
    args (object): 参数对象

    返回：
    str: 构建的关系剪枝提示字符串
    """
    return extract_relation_prompt % (
        args.width, args.width) + question + '\nTopic Entity: ' + entity_name + '\nRelations: ' + '; '.join(
        total_relations) + "\nA: "


def construct_entity_score_prompt(question, relation, entity_candidates):
    """
    构建实体分数提示函数

    参数：
    question (str): 问题
    relation (str): 关系
    entity_candidates (list): 实体候选列表

    返回：
    str: 构建好的实体分数提示字符串
    """
    return score_entity_candidates_prompt.format(question, relation) + "; ".join(entity_candidates) + '\nScore: '


def relation_search_prune(entity_id, entity_name, pre_relations, pre_head, question, args):
    """
    关系剪枝搜索函数
    参数：
    entity_id (str): 实体ID
    entity_name (str): 实体名称
    pre_relations (list): 预设关系列表
    pre_head (bool): 预设头部关系
    question (str): 问题
    args (object): 参数对象
    返回：
    True: 成功搜索关系
    relations: list，搜索到的关系列表
    False: 失败搜索关系
    """
    # 通过SPARQL查询获取实体的头部关系
    sparql_relations_extract_head = sparql_head_relations % entity_id
    head_relations = execurte_sparql(sparql_relations_extract_head)
    head_relations = replace_relation_prefix(head_relations)

    # 通过SPARQL查询获取实体的尾部关系
    sparql_relations_extract_tail = sparql_tail_relations % entity_id
    tail_relations = execurte_sparql(sparql_relations_extract_tail)
    tail_relations = replace_relation_prefix(tail_relations)

    # 如果需要移除不必要的关系，则移除
    if args.remove_unnecessary_rel:
        head_relations = [relation for relation in head_relations if not abandon_rels(relation)]
        tail_relations = [relation for relation in tail_relations if not abandon_rels(relation)]

    # 如果预设头部关系，则移除预设头部关系
    if pre_head:
        tail_relations = list(set(tail_relations) - set(pre_relations))
    else:
        head_relations = list(set(head_relations) - set(pre_relations))

    # 将头部关系和尾部关系合并，并去重
    head_relations = list(set(head_relations))
    tail_relations = list(set(tail_relations))
    total_relations = head_relations + tail_relations
    total_relations.sort()  # 确保提示中的顺序始终相同

    # 根据指定的工具进行关系修剪
    if args.prune_tools == "llm":
        prompt = construct_relation_prune_prompt(question, entity_name, total_relations, args)

        result = run_llm(prompt, args.temperature_exploration, args.max_length, args.opeani_api_keys, args.LLM_type)
        flag, retrieve_relations_with_scores = clean_relations(result, entity_id, head_relations)

    elif args.prune_tools == "bm25":
        topn_relations, topn_scores = compute_bm25_similarity(question, total_relations, args.width)
        flag, retrieve_relations_with_scores = clean_relations_bm25_sent(topn_relations, topn_scores, entity_id,
                                                                         head_relations)
    else:
        model = SentenceTransformer('sentence-transformers/msmarco-distilbert-base-tas-b')
        topn_relations, topn_scores = retrieve_top_docs(question, total_relations, model, args.width)
        flag, retrieve_relations_with_scores = clean_relations_bm25_sent(topn_relations, topn_scores, entity_id,
                                                                         head_relations)

    # 如果修剪成功，则返回修剪后的关系及其分数
    if flag:
        return retrieve_relations_with_scores
    else:
        return []  # 格式错误或最大长度太小


def entity_search(entity, relation, head=True):
    """
    根据给定的实体和关系进行实体搜索

    参数：
    entity (str): 实体名称
    relation (str): 关系名称
    head (bool, 可选): 是否搜索实体的头部，默认为True

    返回：
    list: 包含以“m.”为前缀的实体列表
    """

    if head:
        tail_entities_extract = sparql_tail_entities_extract % (entity, relation)
        entities = execurte_sparql(tail_entities_extract)
    else:
        head_entities_extract = sparql_head_entities_extract % (entity, relation)
        entities = execurte_sparql(head_entities_extract)

    entity_ids = replace_entities_prefix(entities)
    new_entity = [entity for entity in entity_ids if entity.startswith("m.")]
    return new_entity


def entity_score(question, entity_candidates_id, score, relation, args):
    # 将实体候选者的id转换为实体名称或类型
    entity_candidates = [id2entity_name_or_type(entity_id) for entity_id in entity_candidates_id]

    # 如果所有实体候选者都是未知实体，则返回平均分数
    if all_unknown_entity(entity_candidates):
        return [1 / len(entity_candidates) * score] * len(entity_candidates), entity_candidates, entity_candidates_id

    # 删除未知实体
    entity_candidates = del_unknown_entity(entity_candidates)

    # 如果只剩下一个实体候选者，则返回该实体的分数
    if len(entity_candidates) == 1:
        return [score], entity_candidates, entity_candidates_id

    # 如果实体候选者列表为空，则返回0.0分数
    if len(entity_candidates) == 0:
        return [0.0], entity_candidates, entity_candidates_id

    # 确保id和实体名称或类型按照相同的顺序排列
    zipped_lists = sorted(zip(entity_candidates, entity_candidates_id))
    entity_candidates, entity_candidates_id = zip(*zipped_lists)
    entity_candidates = list(entity_candidates)
    entity_candidates_id = list(entity_candidates_id)

    # 如果使用LLM进行分数计算
    if args.prune_tools == "llm":
        # 构建LLM提示
        prompt = construct_entity_score_prompt(question, relation, entity_candidates)

        # 运行LLM模型
        result = run_llm(prompt, args.temperature_exploration, args.max_length, args.opeani_api_keys, args.LLM_type)

        # 清理分数并返回
        return [float(x) * score for x in
                clean_scores(result, entity_candidates)], entity_candidates, entity_candidates_id

    # 如果使用BM25进行分数计算
    elif args.prune_tools == "bm25":
        # 计算BM25相似度
        topn_entities, topn_scores = compute_bm25_similarity(question, entity_candidates, args.width)
    # 如果使用SentenceTransformer进行分数计算
    else:
        # 加载模型
        model = SentenceTransformer('sentence-transformers/msmarco-distilbert-base-tas-b')

        # 检索相关文档并返回分数和实体候选者
        topn_entities, topn_scores = retrieve_top_docs(question, entity_candidates, model, args.width)

        # 如果所有分数都为0，则将分数设置为1 / len(topn_scores)
        if if_all_zero(topn_scores):
            topn_scores = [float(1 / len(topn_scores))] * len(topn_scores)

        # 返回分数和实体候选者
        return [float(x) * score for x in topn_scores], topn_entities, entity_candidates_id


def update_history(entity_candidates, entity, scores, entity_candidates_id, total_candidates, total_scores,
                   total_relations, total_entities_id, total_topic_entities, total_head):
    # 如果实体候选列表为空，则添加一个表示完成的标记
    if len(entity_candidates) == 0:
        entity_candidates.append("[FINISH]")
        entity_candidates_id = ["[FINISH_ID]"]

    # 将实体的relation属性复制给candidates_relation列表
    candidates_relation = [entity['relation']] * len(entity_candidates)

    # 将实体的entity属性复制给topic_entities列表
    topic_entities = [entity['entity']] * len(entity_candidates)

    # 将实体的head属性复制给head_num列表
    head_num = [entity['head']] * len(entity_candidates)

    # 将entity_candidates列表添加到total_candidates列表中
    total_candidates.extend(entity_candidates)

    # 将scores列表添加到total_scores列表中
    total_scores.extend(scores)

    # 将candidates_relation列表添加到total_relations列表中
    total_relations.extend(candidates_relation)

    # 将entity_candidates_id列表添加到total_entities_id列表中
    total_entities_id.extend(entity_candidates_id)

    # 将topic_entities列表添加到total_topic_entities列表中
    total_topic_entities.extend(topic_entities)

    # 将head_num列表添加到total_head列表中
    total_head.extend(head_num)

    return total_candidates, total_scores, total_relations, total_entities_id, total_topic_entities, total_head


def half_stop(question, cluster_chain_of_entities, depth, args):
    # 打印搜索深度达到的提示信息
    print("搜索深度 %d 达到，停止搜索。" % depth)
    # 生成答案
    answer = generate_answer(question, cluster_chain_of_entities, args)
    # 将问题、答案、实体链和文件名保存到jsonl文件中
    save_2_jsonl(question, answer, cluster_chain_of_entities, file_name=args.dataset)


def generate_answer(question, cluster_chain_of_entities, args):
    # 生成回答的提示
    prompt = answer_prompt + question + '\n'
    # 生成知识三元组的提示
    chain_prompt = '\n'.join(
        [', '.join([str(x) for x in chain]) for sublist in cluster_chain_of_entities for chain in sublist])
    prompt += "\nKnowledge Triplets: " + chain_prompt + 'A: '
    # 运行LLM模型生成结果
    result = run_llm(prompt, args.temperature_reasoning, args.max_length, args.opeani_api_keys, args.LLM_type)
    return result


def entity_prune(total_entities_id, total_relations, total_candidates, total_topic_entities, total_head, total_scores,
                 args):
    # 将所有实体ID、关系、候选实体、主题实体、头部实体和分数打包成一个列表
    zipped = list(
        zip(total_entities_id, total_relations, total_candidates, total_topic_entities, total_head, total_scores))

    # 根据分数对打包好的列表进行排序，从高到低
    sorted_zipped = sorted(zipped, key=lambda x: x[5], reverse=True)

    # 从排序好的列表中提取出实体ID、关系、候选实体、主题实体、头部实体和分数
    sorted_entities_id, sorted_relations, sorted_candidates, sorted_topic_entities, sorted_head, sorted_scores = [x[0]
                                                                                                                  for x
                                                                                                                  in
                                                                                                                  sorted_zipped], [
        x[1] for x in sorted_zipped], [x[2] for x in sorted_zipped], [x[3] for x in sorted_zipped], [x[4] for x in
                                                                                                     sorted_zipped], [
        x[5] for x in sorted_zipped]

    # 根据参数args的宽度截取排序好的实体ID、关系、候选实体、主题实体、头部实体和分数
    entities_id, relations, candidates, topics, heads, scores = sorted_entities_id[:args.width], sorted_relations[
                                                                                                 :args.width], sorted_candidates[
                                                                                                               :args.width], sorted_topic_entities[
                                                                                                                             :args.width], sorted_head[
                                                                                                                                           :args.width], sorted_scores[
                                                                                                                                                         :args.width]

    # 将截取好的实体ID、关系、候选实体、主题实体、头部实体和分数打包成一个列表
    merged_list = list(zip(entities_id, relations, candidates, topics, heads, scores))

    # 从打包好的列表中过滤掉分数为0的实体
    filtered_list = [(id, rel, ent, top, hea, score) for id, rel, ent, top, hea, score in merged_list if score != 0]

    # 如果过滤后的列表为空，则返回False和空列表
    if len(filtered_list) == 0:
        return False, [], [], [], []

    # 将过滤后的列表中的实体ID、关系、候选实体、主题实体、头部实体和分数分别打包成列表
    entities_id, relations, candidates, tops, heads, scores = map(list, zip(*filtered_list))

    # 将主题实体转换为实体名称或类型
    tops = [id2entity_name_or_type(entity_id) for entity_id in tops]

    # 构建实体链表，每个实体链表包含主题实体、关系和候选实体
    cluster_chain_of_entities = [[(tops[i], relations[i], candidates[i]) for i in range(len(candidates))]]

    # 返回True和实体链表、实体ID、关系、头部实体
    return True, cluster_chain_of_entities, entities_id, relations, heads


def reasoning(question, cluster_chain_of_entities, args):
    # 构建提示文本
    prompt = prompt_evaluate + question
    # 构建知识三元组链的字符串
    chain_prompt = '\n'.join(
        [', '.join([str(x) for x in chain]) for sublist in cluster_chain_of_entities for chain in sublist])
    # 将知识三元组链添加到提示文本中
    prompt += "\nKnowledge Triplets: " + chain_prompt + 'A: '

    # 运行语言模型并获取响应
    response = run_llm(prompt, args.temperature_reasoning, args.max_length, args.opeani_api_keys, args.LLM_type)

    # 从响应中提取答案
    result = extract_answer(response)
    # 判断答案是否为真
    if if_true(result):
        return True, response
    else:
        return False, response
