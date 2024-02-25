from prompt_list import *
import json
import time
import openai
import re
from prompt_list import *
from rank_bm25 import BM25Okapi
from sentence_transformers import util
from sentence_transformers import SentenceTransformer

def retrieve_top_docs(query, docs, model, width=3):
    """
    获取给定查询的前n个最相关的文档。

    参数：
    - query (str)：输入的查询。
    - docs (list of str)：要搜索的文档列表。
    - model_name (str)：要使用的SentenceTransformer模型的名称。
    - width (int)：要返回的前n个文档的数量。

    返回：
    - list of float：前n个文档的分数列表。
    - list of str：前n个文档的列表。
    """

    # 将查询和文档编码为嵌入向量
    query_emb = model.encode(query)
    doc_emb = model.encode(docs)

    # 计算查询和文档嵌入向量之间的点积得分
    scores = util.dot_score(query_emb, doc_emb)[0].cpu().tolist()

    # 根据分数对文档进行排序
    doc_score_pairs = sorted(list(zip(docs, scores)), key=lambda x: x[1], reverse=True)

    # 获取前n个文档和分数
    top_docs = [pair[0] for pair in doc_score_pairs[:width]]
    top_scores = [pair[1] for pair in doc_score_pairs[:width]]

    return top_docs, top_scores



def compute_bm25_similarity(query, corpus, width=3):
    """
    计算BM25相似度，返回与输入问题相似度最高的前n个关系及其分数。

    Args:
    - question (str): 输入问题。
    - relations_list (list): 关系列表。
    - width (int): 返回的关系数量。

    Returns:
    - list, list: 相似度最高的前n个关系及其分数。
    """

    # 将语料库分词
    tokenized_corpus = [doc.split(" ") for doc in corpus]
    # 创建BM25Okapi对象
    bm25 = BM25Okapi(tokenized_corpus)
    # 将查询分词
    tokenized_query = query.split(" ")

    # 计算文档分数
    doc_scores = bm25.get_scores(tokenized_query)

    # 获取与查询最相似的前n个文档
    relations = bm25.get_top_n(tokenized_query, corpus, n=width)
    # 获取文档分数的前n个
    doc_scores = sorted(doc_scores, reverse=True)[:width]

    return relations, doc_scores



def clean_relations(string, entity_id, head_relations):
    """
    清理关系字符串，提取关系和分数，并根据关系是否在head_relations中确定是否为主关系。
    Args:
        string (str): 包含关系的字符串。
        entity_id (str): 实体ID。
        head_relations (list): 主关系列表。

    Returns:
        tuple: 包含两个元素的元组，第一个元素为布尔值，表示是否成功提取关系，第二个元素为关系列表。
    """
    pattern = r"{\s*(?P<relation>[^()]+)\s+\(Score:\s+(?P<score>[0-9.]+)\)}"
    relations=[]
    for match in re.finditer(pattern, string):
        relation = match.group("relation").strip()
        if ';' in relation:
            continue
        score = match.group("score")
        if not relation or not score:
            return False, "output uncompleted.."
        try:
            score = float(score)
        except ValueError:
            return False, "Invalid score"
        if relation in head_relations:
            relations.append({"entity": entity_id, "relation": relation, "score": score, "head": True})
        else:
            relations.append({"entity": entity_id, "relation": relation, "score": score, "head": False})
    if not relations:
        return False, "No relations found"
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
    relations = []
    if if_all_zero(topn_scores):
        topn_scores = [float(1/len(topn_scores))] * len(topn_scores)
    i=0
    for relation in topn_relations:
        if relation in head_relations:
            relations.append({"entity": entity_id, "relation": relation, "score": topn_scores[i], "head": True})
        else:
            relations.append({"entity": entity_id, "relation": relation, "score": topn_scores[i], "head": False})
        i+=1
    return True, relations


def run_llm(prompt, temperature, max_tokens, opeani_api_keys, engine="gpt-3.5-turbo"):
    # 如果引擎中包含"llama"，则使用本地的llama服务器端口和模型ID
    if "llama" in engine.lower():
        openai.api_key = "EMPTY"  # 设置openai的api_key为空
        openai.api_base = "http://localhost:8000/v1"  # 设置openai的api_base为本地llama服务器端口
        engine = openai.Model.list()["data"][0]["id"]  # 获取本地llama服务器的模型ID
    else:
        openai.api_key = opeani_api_keys  # 设置openai的api_key为传入的api_keys

    # 创建消息列表
    messages = [{"role":"system","content":"You are an AI assistant that helps people find information."}]
    message_prompt = {"role":"user","content":prompt}
    messages.append(message_prompt)

    # 初始化变量f为0
    f = 0
    while(f == 0):
        try:
            # 使用openai的ChatCompletion.create方法进行聊天
            response = openai.ChatCompletion.create(
                    model=engine,  # 使用指定的模型进行聊天
                    messages = messages,  # 消息列表
                    temperature=temperature,  # 温度参数
                    max_tokens=max_tokens,  # 最大令牌数
                    frequency_penalty=0,  # 频率惩罚参数
                    presence_penalty=0)  # 存在惩罚参数
            result = response["choices"][0]['message']['content']  # 获取聊天结果
            f = 1  # 设置f为1，表示聊天成功
        except:
            # 如果出现openai错误，打印错误信息并等待2秒后重试
            print("openai error, retry")
            time.sleep(2)
    return result


    
def all_unknown_entity(entity_candidates):
    """
    判断给定的实体候选列表中是否全部为未知实体

    参数：
    entity_candidates (list): 实体候选列表

    返回值：
    bool: 如果所有实体候选都是未知实体，则返回True，否则返回False
    """
    return all(candidate == "UnName_Entity" for candidate in entity_candidates)


def del_unknown_entity(entity_candidates):
    # 如果实体候选列表中只有一个元素，并且该元素为"UnName_Entity"，则直接返回该列表
    if len(entity_candidates)==1 and entity_candidates[0]=="UnName_Entity":
        return entity_candidates
    # 过滤掉"UnName_Entity"，得到新的实体候选列表
    entity_candidates = [candidate for candidate in entity_candidates if candidate != "UnName_Entity"]
    # 返回过滤后的实体候选列表
    return entity_candidates



import re

def clean_scores(string, entity_candidates):
    """
    从字符串中提取分数，并将其转换为浮点数列表。
    如果提取的分数数量与实体候选数量相等，则返回分数列表。
    否则，打印"All entities are created equal."并返回一个长度为实体候选数量的分数列表，其中每个分数都为1/实体候选数量。
    """
    scores = re.findall(r'\d+\.\d+', string)  # 使用正则表达式提取字符串中的分数
    scores = [float(number) for number in scores]  # 将分数转换为浮点数列表
    if len(scores) == len(entity_candidates):  # 如果提取的分数数量与实体候选数量相等
        return scores  # 返回分数列表
    else:
        print("All entities are created equal.")  # 打印"All entities are created equal."
        return [1/len(entity_candidates)] * len(entity_candidates)  # 返回一个长度为实体候选数量的分数列表，其中每个分数都为1/实体候选数量

    

def save_2_jsonl(question, answer, cluster_chain_of_entities, file_name):
    # 创建一个字典，包含问题、答案和实体簇链
    dict = {"question":question, "results": answer, "reasoning_chains": cluster_chain_of_entities}
    # 打开一个名为"ToG_文件名.jsonl"的文件，如果不存在则创建
    with open("ToG_{}.jsonl".format(file_name), "a") as outfile:
        # 将字典转换为JSON字符串
        json_str = json.dumps(dict)
        # 将JSON字符串写入文件，并在末尾添加换行符
        outfile.write(json_str + "\n")


    
def extract_answer(text):
    """
    从给定的文本中提取答案。

    参数：
    text (str): 包含答案的文本。

    返回：
    str: 提取的答案。
    """
    start_index = text.find("{")
    end_index = text.find("}")
    if start_index != -1 and end_index != -1:
        return text[start_index+1:end_index].strip()
    else:
        return ""

    

def if_true(prompt):
    """
    判断输入的prompt是否为"yes"，如果是则返回True，否则返回False。
    """
    if prompt.lower().strip().replace(" ","")=="yes":
        return True
    return False



def generate_without_explored_paths(question, args):
    """
    生成不包含已探索路径的回答

    参数:
    question (str): 问题
    args: 包含以下参数的字典:
        temperature_reasoning (float): 温度参数
        max_length (int): 最大长度
        opeani_api_keys (str): Opeani API密钥
        LLM_type (str): LLM类型

    返回:
    response (str): 回答
    """
    # 构建提示
    prompt = cot_prompt + "\n\nQ: " + question + "\nA:"
    # 运行LLM模型生成回答
    response = run_llm(prompt, args.temperature_reasoning, args.max_length, args.opeani_api_keys, args.LLM_type)
    # 返回回答
    return response


def if_finish_list(lst):
    """
    判断列表中是否全部元素为"[FINISH_ID]"，若是则返回True和一个空列表，否则返回False和去除"[FINISH_ID]"元素后的列表。
    """
    if all(elem == "[FINISH_ID]" for elem in lst):
        return True, []
    else:
        new_lst = [elem for elem in lst if elem != "[FINISH_ID]"]
        return False, new_lst


def prepare_dataset(dataset_name):
    # 根据传入的dataset_name参数，准备相应的数据集
    if dataset_name == 'cwq':
        # 如果dataset_name为'cwq'，则打开cwq.json文件，读取数据并将其存储在datas变量中
        with open('../data/cwq.json',encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'question'
    elif dataset_name == 'webqsp':
        # 如果dataset_name为'webqsp'，则打开WebQSP.json文件，读取数据并将其存储在datas变量中
        with open('../data/WebQSP.json',encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'RawQuestion'
    elif dataset_name == 'grailqa':
        # 如果dataset_name为'grailqa'，则打开grailqa.json文件，读取数据并将其存储在datas变量中
        with open('../data/grailqa.json',encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'question'
    elif dataset_name == 'simpleqa':
        # 如果dataset_name为'simpleqa'，则打开SimpleQA.json文件，读取数据并将其存储在datas变量中
        with open('../data/SimpleQA.json',encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'question'
    elif dataset_name == 'qald':
        # 如果dataset_name为'qald'，则打开qald_10-en.json文件，读取数据并将其存储在datas变量中
        with open('../data/qald_10-en.json',encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'question'
    elif dataset_name == 'webquestions':
        # 如果dataset_name为'webquestions'，则打开WebQuestions.json文件，读取数据并将其存储在datas变量中
        with open('../data/WebQuestions.json',encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'question'
    elif dataset_name == 'trex':
        # 如果dataset_name为'trex'，则打开T-REX.json文件，读取数据并将其存储在datas变量中
        with open('../data/T-REX.json',encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'input'
    elif dataset_name == 'zeroshotre':
        # 如果dataset_name为'zeroshotre'，则打开Zero_Shot_RE.json文件，读取数据并将其存储在datas变量中
        with open('../data/Zero_Shot_RE.json',encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'input'
    elif dataset_name == 'creak':
        # 如果dataset_name为'creak'，则打开creak.json文件，读取数据并将其存储在datas变量中
        with open('../data/creak.json',encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'sentence'
    else:
        # 如果dataset_name不是以上任何一种情况，则打印错误信息并退出程序
        print("dataset not found, you should pick from {cwq, webqsp, grailqa, simpleqa, qald, webquestions, trex, zeroshotre, creak}.")
        exit(-1)
    # 返回数据集和问题字符串
    return datas, question_string