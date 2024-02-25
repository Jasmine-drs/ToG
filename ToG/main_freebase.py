from tqdm import tqdm
import argparse
from utils import *
from freebase_func import *
import random
from client import *


if __name__ == '__main__':
    # 创建参数解析器
    parser = argparse.ArgumentParser()
    # 添加参数
    parser.add_argument("--dataset", type=str,
                        default="webqsp", help="选择数据集.")
    parser.add_argument("--max_length", type=int,
                        default=256, help="LLM输出的最大长度.")
    parser.add_argument("--temperature_exploration", type=float,
                        default=0.4, help="探索阶段的温度.")
    parser.add_argument("--temperature_reasoning", type=float,
                        default=0, help="推理阶段的温度.")
    parser.add_argument("--width", type=int,
                        default=3, help="ToG的搜索宽度.")
    parser.add_argument("--depth", type=int,
                        default=3, help="ToG的搜索深度.")
    parser.add_argument("--remove_unnecessary_rel", type=bool,
                        default=True, help="是否移除不必要的关系.")
    parser.add_argument("--LLM_type", type=str,
                        default="gpt-3.5-turbo", help="基础LLM模型.")
    parser.add_argument("--opeani_api_keys", type=str,
                        default="", help="如果LLM_type是gpt-3.5-turbo或gpt-4，则需要添加自己的OpenAI API密钥.")
    parser.add_argument("--num_retain_entity", type=int,
                        default=5, help="保留的实体数量.")
    parser.add_argument("--prune_tools", type=str,
                        default="llm", help="ToG的剪枝工具，可以是LLM_type（与LLM_type相同）、bm25或sentencebert.")
    # 解析参数
    args = parser.parse_args()

    # 准备数据集
    datas, question_string = prepare_dataset(args.dataset)
    # 打印运行信息
    print("开始在%s数据集上运行ToG." % args.dataset)
    # 遍历数据集
    for data in tqdm(datas):
        # 获取问题和主题实体
        question = data[question_string]
        topic_entity = data['topic_entity']
        # 初始化链式实体列表
        cluster_chain_of_entities = []
        # 如果主题实体为空，则直接生成结果
        if len(topic_entity) == 0:
            results = generate_without_explored_paths(question, args)
            save_2_jsonl(question, results, [], file_name=args.dataset)
            continue
        # 初始化前一个关系和前一个头部
        pre_relations = []
        pre_heads= [-1] * len(topic_entity)
        # 标记是否已打印结果
        flag_printed = False
        # 遍历搜索深度
        for depth in range(1, args.depth+1):
            # 初始化当前实体关系列表
            current_entity_relations_list = []
            i=0
            # 遍历主题实体
            for entity in topic_entity:
                # 如果不是结束标识，则获取当前实体的关系和得分
                if entity!="[FINISH_ID]":
                    retrieve_relations_with_scores = relation_search_prune(entity, topic_entity[entity], pre_relations, pre_heads[i], question, args)  # 最佳实体三元组，实体id
                    current_entity_relations_list.extend(retrieve_relations_with_scores)
                i+=1
            # 初始化候选列表、得分列表、关系列表、实体id列表、主题实体列表和头部列表
            total_candidates = []
            total_scores = []
            total_relations = []
            total_entities_id = []
            total_topic_entities = []
            total_head = []

            # 遍历当前实体关系列表
            for entity in current_entity_relations_list:
                # 如果是头部，则获取当前实体的候选id
                if entity['head']:
                    entity_candidates_id = entity_search(entity['entity'], entity['relation'], True)
                else:
                    entity_candidates_id = entity_search(entity['entity'], entity['relation'], False)

                # 如果剪枝工具是LLM，则根据保留数量随机采样候选id
                if args.prune_tools == "llm":
                    if len(entity_candidates_id) >=20:
                        entity_candidates_id = random.sample(entity_candidates_id, args.num_retain_entity)

                # 如果候选id为空，则继续下一个循环
                if len(entity_candidates_id) ==0:
                    continue
                # 获取候选id的得分、候选实体、候选实体id
                scores, entity_candidates, entity_candidates_id = entity_score(question, entity_candidates_id, entity['score'], entity['relation'], args)

                # 更新候选列表、得分列表、关系列表、实体id列表、主题实体列表和头部列表
                total_candidates, total_scores, total_relations, total_entities_id, total_topic_entities, total_head = update_history(entity_candidates, entity, scores, entity_candidates_id, total_candidates, total_scores, total_relations, total_entities_id, total_topic_entities, total_head)

            # 如果候选列表为空，则进行半停止
            if len(total_candidates) ==0:
                half_stop(question, cluster_chain_of_entities, depth, args)
                flag_printed = True
                break

            # 进行实体剪枝
            flag, chain_of_entities, entities_id, pre_relations, pre_heads = entity_prune(total_entities_id, total_relations, total_candidates, total_topic_entities, total_head, total_scores, args)
            # 更新链式实体列表
            cluster_chain_of_entities.append(chain_of_entities)
            # 如果剪枝成功，则进行推理
            if flag:
                stop, results = reasoning(question, cluster_chain_of_entities, args)
                # 如果推理成功，则打印结果并保存
                if stop:
                    print("ToG在深度%d停止." % depth)
                    save_2_jsonl(question, results, cluster_chain_of_entities, file_name=args.dataset)
                    flag_printed = True
                    break
                else:
                    print("深度%d仍未找到答案." % depth)
                    flag_finish, entities_id = if_finish_list(entities_id)
                    # 如果已结束，则进行半停止
                    if flag_finish:
                        half_stop(question, cluster_chain_of_entities, depth, args)
                        flag_printed = True
                    else:
                        # 更新主题实体
                        topic_entity = {entity: id2entity_name_or_type(entity) for entity in entities_id}
                        continue
            else:
                half_stop(question, cluster_chain_of_entities, depth, args)
                flag_printed = True

        # 如果未打印结果，则进行无探索路径生成
        if not flag_printed:
            results = generate_without_explored_paths(question, args)
            save_2_jsonl(question, results, [], file_name=args.dataset)
