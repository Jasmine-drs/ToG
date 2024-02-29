from prompt_list import *
import json
import openai
import re
import time
from utils import *


def transform_relation(relation):
    """
    将关系转换为无前缀的关系名称。
    参数：
        relation (str): 带有前缀的关系名称。
    返回：
        str: 无前缀的关系名称。
    """
    relation_without_prefix = relation.replace("wiki.relation.", "").replace("_", " ")
    return relation_without_prefix


def clean_relations(string, entity_id, head_relations):
    # 定义正则表达式模式，用于匹配关系和分数
    pattern = r"{\s*(?P<relation>[^()]+)\s+\(Score:\s+(?P<score>[0-9.]+)\)}"
    # 初始化关系列表
    relations = []
    # 遍历匹配结果
    for match in re.finditer(pattern, string):
        # 获取关系和分数
        relation = match.group("relation").strip()
        relation = transform_relation(relation)
        if ';' in relation:
            continue
        score = match.group("score")
        # 如果关系或分数为空，则返回False和"output uncompleted.."
        if not relation or not score:
            return False, "output uncompleted.."
        # 尝试将分数转换为浮点数
        try:
            score = float(score)
        except ValueError:
            return False, "Invalid score"
        # 如果关系在head_relations中，则将关系添加到列表中，并标记为head=True
        if relation in head_relations:
            relations.append({"entity": entity_id, "relation": relation, "score": score, "head": True})
        else:
            # 否则将关系添加到列表中，并标记为head=False
            relations.append({"entity": entity_id, "relation": relation, "score": score, "head": False})
    # 如果没有找到关系，则返回False和"No relations found"
    if not relations:
        return False, "No relations found"
    # 如果成功找到关系，则返回True和关系列表
    return True, relations


def construct_relation_prune_prompt(question, entity_name, total_relations, args):
    # 构建关系修剪提示
    # 参数：
    #   question: 问题
    #   entity_name: 实体名称
    #   total_relations: 总关系列表
    #   args: 参数对象
    # 返回值：修剪提示字符串
    return extract_relation_prompt_wiki % (
        args.width, args.width) + question + '\nTopic Entity: ' + entity_name + '\nRelations:\n' + '\n'.join(
        [f"{i}. {item}" for i, item in enumerate(total_relations, start=1)]) + 'A:'


def check_end_word(s):
    # 检查字符串s是否以给定的单词之一结尾
    # 参数：
    #   s (str): 要检查的字符串
    # 返回值：
    #   bool: 如果字符串s以给定的单词之一结尾，则返回True，否则返回False
    words = [" ID", " code", " number", "instance of", "website", "URL", "inception", "image", " rate", " count"]
    return any(s.endswith(word) for word in words)


def abandon_rels(relation):
    # 定义一个函数，用于判断是否放弃关系
    useless_relation_list = ["category's main topic", "topic\'s main category", "stack exchange site", 'main subject',
                             'country of citizenship', "commons category", "commons gallery", "country of origin",
                             "country", "nationality"]
    # 定义一个无用关系列表
    if check_end_word(
            relation) or 'wikidata' in relation.lower() or 'wikimedia' in relation.lower() or relation.lower() in useless_relation_list:
        # 如果关系的末尾单词是无用关系、关系中包含'wikidata'或'wikimedia'，或者关系在无用关系列表中，则返回True
        return True
    # 否则返回False
    return False


def construct_entity_score_prompt(question, relation, entity_candidates):
    """
    构建实体评分提示函数

    参数：
    question (str): 问题
    relation (str): 关系
    entity_candidates (list): 实体候选列表

    返回：
    str: 构建好的实体评分提示字符串
    """
    return score_entity_candidates_prompt_wiki.format(question, relation) + "; ".join(entity_candidates) + '\nScore: '


def relation_search_prune(entity_id, entity_name, pre_relations, pre_head, question, args, wiki_client):
    # 查询给定实体的所有关系
    relations = wiki_client.query_all("get_all_relations_of_an_entity", entity_id)
    head_relations = [rel['label'] for rel in relations['head']]
    tail_relations = [rel['label'] for rel in relations['tail']]

    # 如果需要移除不必要的关系，则移除
    if args.remove_unnecessary_rel:
        head_relations = [relation for relation in head_relations if not abandon_rels(relation)]
        tail_relations = [relation for relation in tail_relations if not abandon_rels(relation)]

    # 如果有预设的关系，则移除预设关系中的关系
    if pre_head:
        tail_relations = list(set(tail_relations) - set(pre_relations))
    else:
        head_relations = list(set(head_relations) - set(pre_relations))

    # 将关系列表转换为集合，然后转换回列表，以确保唯一性
    head_relations = list(set(head_relations))
    tail_relations = list(set(tail_relations))

    # 将所有关系合并，并按字母顺序排序
    total_relations = head_relations + tail_relations
    total_relations.sort()  # make sure the order in prompt is always equal

    # 构建关系修剪提示
    prompt = construct_relation_prune_prompt(question, entity_name, total_relations, args)

    # 运行语言模型
    result = run_llm(prompt, args.temperature_exploration, args.max_length, args.opeani_api_keys, args.LLM_type)

    # 清理结果并获取带有分数的关系
    flag, retrieve_relations_with_scores = clean_relations(result, entity_id, head_relations)

    # 如果清理成功，则返回带有分数的关系
    if flag:
        return retrieve_relations_with_scores
    else:
        return []  # format error or too small max_length


def del_all_unknown_entity(entity_candidates_id, entity_candidates_name):
    """
    删除所有未知实体的候选实体列表

    Args:
        entity_candidates_id (list): 候选实体的ID列表
        entity_candidates_name (list): 候选实体的名称列表

    Returns:
        tuple: 包含更新后的候选实体ID列表和名称列表的元组
    """
    if len(entity_candidates_name) == 1 and entity_candidates_name[0] == "N/A":
        return entity_candidates_id, entity_candidates_name

    new_candidates_id = []
    new_candidates_name = []
    for i, candidate in enumerate(entity_candidates_name):
        if candidate != "N/A":
            new_candidates_id.append(entity_candidates_id[i])
            new_candidates_name.append(candidate)

    return new_candidates_id, new_candidates_name


def all_zero(topn_scores):
    """
    判断给定的topn_scores列表中的所有元素是否都为0。

    参数：
    topn_scores (list): 包含分数的列表。

    返回值：
    bool: 如果所有元素都为0，则返回True；否则返回False。
    """
    return all(score == 0 for score in topn_scores)


def entity_search(entity, relation, wiki_client, head):
    # 根据关系查询相关实体的id
    rid = wiki_client.query_all("label2pid", relation)
    # 如果没有找到相关实体或者找到的是"Not Found!"，则返回空列表
    if not rid or rid == "Not Found!":
        return [], []

    # 取出最后一个相关实体的id
    rid_str = rid.pop()

    # 根据实体和相关实体的id查询相关实体的信息
    entities = wiki_client.query_all("get_tail_entities_given_head_and_relation", entity, rid_str)

    # 如果需要查询的是实体的尾部，则将结果赋值给尾部实体集合，否则赋值给头部实体集合
    if head:
        entities_set = entities['tail']
    else:
        entities_set = entities['head']

    # 如果没有找到相关实体，则查询相关实体的值，并返回空列表和值的列表
    if not entities_set:
        values = wiki_client.query_all("get_tail_values_given_head_and_relation", entity, rid_str)
        return [], list(values)

    # 将相关实体的id和名称存储在列表中
    id_list = [item['qid'] for item in entities_set]
    name_list = [item['label'] if item['label'] != "N/A" else "Unname_Entity" for item in entities_set]

    return id_list, name_list


def entity_score(question, entity_candidates_id, entity_candidates, score, relation, args):
    # 如果候选实体列表只有一个，直接返回该实体的分数
    if len(entity_candidates) == 1:
        return [score], entity_candidates, entity_candidates_id
    # 如果候选实体列表为空，返回0.0的分数
    if len(entity_candidates) == 0:
        return [0.0], entity_candidates, entity_candidates_id

    # 确保id和实体的顺序一致
    zipped_lists = sorted(zip(entity_candidates, entity_candidates_id))
    entity_candidates, entity_candidates_id = zip(*zipped_lists)
    entity_candidates = list(entity_candidates)
    entity_candidates_id = list(entity_candidates_id)

    # 构造用于LLM的提示
    prompt = construct_entity_score_prompt(question, relation, entity_candidates)

    # 运行LLM模型
    result = run_llm(prompt, args.temperature_exploration, args.max_length, args.opeani_api_keys, args.LLM_type)
    # 清理LLM模型返回的分数
    entity_scores = clean_scores(result, entity_candidates)
    # 如果所有分数都为0，返回平均分数
    if all_zero(entity_scores):
        return [1 / len(entity_candidates) * score] * len(entity_candidates), entity_candidates, entity_candidates_id
    else:
        # 返回每个实体的分数乘以原始分数
        return [float(x) * score for x in entity_scores], entity_candidates, entity_candidates_id


def update_history(entity_candidates, entity, scores, entity_candidates_id, total_candidates, total_scores,
                   total_relations, total_entities_id, total_topic_entities, total_head, value_flag):
    # 如果value_flag为真，则将entity['score']除以entity_candidates长度并赋值给scores列表
    if value_flag:
        scores = [1 / len(entity_candidates) * entity['score']]
    # 将entity['relation']重复len(entity_candidates)次并赋值给candidates_relation列表
    candidates_relation = [entity['relation']] * len(entity_candidates)
    # 将entity['entity']重复len(entity_candidates)次并赋值给topic_entities列表
    topic_entities = [entity['entity']] * len(entity_candidates)
    # 将entity['head']重复len(entity_candidates)次并赋值给head_num列表
    head_num = [entity['head']] * len(entity_candidates)
    # 将entity_candidates追加到total_candidates列表中
    total_candidates.extend(entity_candidates)
    # 将scores追加到total_scores列表中
    total_scores.extend(scores)
    # 将candidates_relation追加到total_relations列表中
    total_relations.extend(candidates_relation)
    # 将entity_candidates_id追加到total_entities_id列表中
    total_entities_id.extend(entity_candidates_id)
    # 将topic_entities追加到total_topic_entities列表中
    total_topic_entities.extend(topic_entities)
    # 将head_num追加到total_head列表中
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
    prompt = answer_prompt_wiki + question + '\n'
    # 生成知识三元组的提示
    chain_prompt = '\n'.join(
        [', '.join([str(x) for x in chain]) for sublist in cluster_chain_of_entities for chain in sublist])
    prompt += "\nKnowledge Triplets: " + chain_prompt + 'A: '
    # 运行LLM模型生成结果
    result = run_llm(prompt, args.temperature_reasoning, args.max_length, args.opeani_api_keys, args.LLM_type)
    return result


def entity_prune(total_entities_id, total_relations, total_candidates, total_topic_entities, total_head, total_scores,
                 args, wiki_client):
    # 将所有实体ID、关系、候选实体、主题实体、头部和分数打包成一个列表
    zipped = list(
        zip(total_entities_id, total_relations, total_candidates, total_topic_entities, total_head, total_scores))
    # 根据分数对列表进行排序，从高到低
    sorted_zipped = sorted(zipped, key=lambda x: x[5], reverse=True)
    # 提取排序后的实体ID、关系、候选实体、主题实体、头部和分数
    sorted_entities_id, sorted_relations, sorted_candidates, sorted_topic_entities, sorted_head, sorted_scores = [x[0] for x in sorted_zipped], \
        [x[1] for x in sorted_zipped], \
        [x[2] for x in sorted_zipped], \
        [x[3] for x in sorted_zipped], \
        [x[4] for x in sorted_zipped], \
        [x[5] for x in sorted_zipped]

    # 提取前width个实体ID、关系、候选实体、主题实体、头部和分数
    entities_id, relations, candidates, topics, heads, scores = sorted_entities_id[:args.width], sorted_relations[
                                                                                                 :args.width], sorted_candidates[
                                                                                                               :args.width], sorted_topic_entities[
                                                                                                                             :args.width], sorted_head[
                                                                                                                                           :args.width], sorted_scores[
                                                                                                                                                         :args.width]
    # 将提取的实体ID、关系、候选实体、主题实体、头部和分数打包成一个列表
    merged_list = list(zip(entities_id, relations, candidates, topics, heads, scores))
    # 过滤掉分数为0的实体
    filtered_list = [(id, rel, ent, top, hea, score) for id, rel, ent, top, hea, score in merged_list if score != 0]
    # 如果过滤后的列表为空，则返回False和空列表
    if len(filtered_list) == 0:
        return False, [], [], [], []
    # 提取过滤后的实体ID、关系、候选实体、主题实体、头部和分数
    entities_id, relations, candidates, tops, heads, scores = map(list, zip(*filtered_list))
    # 使用wiki_client查询每个主题实体的实体ID，并将查询结果赋值给tops列表
    tops = [wiki_client.query_all("qid2label", entity_id).pop() if (entity_name := wiki_client.query_all("qid2label",
                                                                                                         entity_id)) != "Not Found!" else "Unname_Entity"
            for entity_id in tops]
    # 将过滤后的实体ID、关系、候选实体、主题实体、头部和分数打包成一个列表
    cluster_chain_of_entities = [[(tops[i], relations[i], candidates[i]) for i in range(len(candidates))]]
    # 返回True和打包后的列表
    return True, cluster_chain_of_entities, entities_id, relations, heads


def reasoning(question, cluster_chain_of_entities, args):
    # 生成提示
    prompt = prompt_evaluate_wiki + question
    # 生成知识三元组的提示
    chain_prompt = '\n'.join(
        [', '.join([str(x) for x in chain]) for sublist in cluster_chain_of_entities for chain in sublist])
    prompt += "\nKnowledge Triplets: " + chain_prompt + 'A: '

    # 运行LLM模型生成回答
    response = run_llm(prompt, args.temperature_reasoning, args.max_length, args.opeani_api_keys, args.LLM_type)

    # 提取回答
    result = extract_answer(response)

    # 判断回答是否为真
    if if_true(result):
        return True, response
    else:
        return False, response