import itertools
import xmlrpc.client
import typing as tp
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import requests
from bs4 import BeautifulSoup


class WikidataQueryClient:
    def __init__(self, url: str):
        self.url = url
        self.server = xmlrpc.client.ServerProxy(url)

    def label2qid(self, label: str) -> str:
        """
        根据标签获取实体的QID
        :param label: 标签
        :return: 实体的QID
        """
        return self.server.label2qid(label)

    def label2pid(self, label: str) -> str:
        """
        根据标签获取实体的PID
        :param label: 标签
        :return: 实体的PID
        """
        return self.server.label2pid(label)

    def pid2label(self, pid: str) -> str:
        """
        根据PID获取实体的标签
        :param pid: PID
        :return: 实体的标签
        """
        return self.server.pid2label(pid)

    def qid2label(self, qid: str) -> str:
        """
        根据QID获取实体的标签
        :param qid: QID
        :return: 实体的标签
        """
        return self.server.qid2label(qid)

    def get_all_relations_of_an_entity(
            self, entity_qid: str
    ) -> tp.Dict[str, tp.List]:
        """
        获取实体的所有关系
        :param entity_qid: 实体的QID
        :return: 实体的所有关系
        """
        return self.server.get_all_relations_of_an_entity(entity_qid)

    def get_tail_entities_given_head_and_relation(
            self, head_qid: str, relation_pid: str
    ) -> tp.Dict[str, tp.List]:
        """
        根据头部实体、关系获取尾部实体
        :param head_qid: 头部实体的QID
        :param relation_pid: 关系的PID
        :return: 尾部实体
        """
        return self.server.get_tail_entities_given_head_and_relation(
            head_qid, relation_pid
        )

    def get_tail_values_given_head_and_relation(
            self, head_qid: str, relation_pid: str
    ) -> tp.List[str]:
        """
        根据头部实体、关系获取尾部值
        :param head_qid: 头部实体的QID
        :param relation_pid: 关系的PID
        :return: 尾部值
        """
        return self.server.get_tail_values_given_head_and_relation(
            head_qid, relation_pid
        )

    def get_external_id_given_head_and_relation(
            self, head_qid: str, relation_pid: str
    ) -> tp.List[str]:
        """
        根据头部实体、关系获取外部ID
        :param head_qid: 头部实体的QID
        :param relation_pid: 关系的PID
        :return: 外部ID
        """
        return self.server.get_external_id_given_head_and_relation(
            head_qid, relation_pid
        )

    def get_wikipedia_page(self, qid: str, section: str = None) -> str:
        """
        获取维基百科页面
        :param qid: QID
        :param section: 部分
        :return: 维基百科页面
        """
        wikipedia_url = self.server.get_wikipedia_link(qid)
        if wikipedia_url == "Not Found!":
            return "Not Found!"
        else:
            response = requests.get(wikipedia_url)
            if response.status_code != 200:
                raise Exception(f"Failed to retrieve page: {wikipedia_url}")

            soup = BeautifulSoup(response.content, "html.parser")
            content_div = soup.find("div", {"id": "bodyContent"})

            # Remove script and style elements
            for script_or_style in content_div.find_all(["script", "style"]):
                script_or_style.decompose()

            if section:
                header = content_div.find(
                    lambda tag: tag.name == "h2" and section in tag.get_text()
                )
                if header:
                    content = ""
                    for sibling in header.find_next_siblings():
                        if sibling.name == "h2":
                            break
                        content += sibling.get_text()
                    return content.strip()
                else:
                    # If the specific section is not found, return an empty string or a message.
                    return f"Section '{section}' not found."

            # Fetch the header summary (before the first h2)
            summary_content = ""
            for element in content_div.find_all(recursive=False):
                if element.name == "h2":
                    break
                summary_content += element.get_text()

            return summary_content.strip()

    def mid2qid(self, mid: str) -> str:
        """
        根据MID获取QID
        :param mid: MID
        :return: QID
        """
        return self.server.mid2qid(mid)


import time
import typing as tp
from concurrent.futures import ThreadPoolExecutor


class MultiServerWikidataQueryClient:
    def __init__(self, urls: tp.List[str]):
        """
        初始化MultiServerWikidataQueryClient类的实例。

        参数：
        - urls: str列表，包含多个Wikidata查询服务器的URL。
        """
        self.clients = [WikidataQueryClient(url) for url in urls]
        self.executor = ThreadPoolExecutor(max_workers=len(urls))

    def test_connections(self):
        """
        测试所有客户端的连接。

        返回：
        - 无返回值。
        """

        def test_url(client):
            try:
                # 检查服务器是否提供system.listMethods函数。
                client.server.system.listMethods()
                return True
            except Exception as e:
                print(f"Failed to connect to {client.url}. Error: {str(e)}")
                return False

        start_time = time.perf_counter()
        futures = [
            self.executor.submit(test_url, client) for client in self.clients
        ]
        results = [f.result() for f in futures]
        end_time = time.perf_counter()
        # print(f"Connection testing took {end_time - start_time} seconds")

        # 移除连接失败的客户端
        self.clients = [
            client for client, result in zip(self.clients, results) if result
        ]
        if not self.clients:
            raise Exception("Failed to connect to all URLs")

    def query_all(self, method, *args):
        """
        向所有客户端发送查询请求，并返回结果。

        参数：
        - method: str，要调用的方法名。
        - *args: 可变参数，传递给方法的参数。

        返回：
        - 实际结果，如果结果为空则返回"Not Found!"。
        """
        start_time = time.perf_counter()
        futures = [
            self.executor.submit(getattr(client, method), *args)
            for client in self.clients
        ]
        # Retrieve results and filter out 'Not Found!'
        is_dict_return = method in [
            "get_all_relations_of_an_entity",
            "get_tail_entities_given_head_and_relation",
        ]
        results = [f.result() for f in futures]
        end_time = time.perf_counter()
        # print(f"HTTP Queries took {end_time - start_time} seconds")

        start_time = time.perf_counter()
        real_results = (
            set() if not is_dict_return else {"head": [], "tail": []}
        )
        for res in results:
            if isinstance(res, str) and res == "Not Found!":
                continue
            elif isinstance(res, tp.List):
                if len(res) == 0:
                    continue
                if isinstance(res[0], tp.List):
                    res_flattened = itertools.chain(*res)
                    real_results.update(res_flattened)
                    continue
                real_results.update(res)
            elif is_dict_return:
                real_results["head"].extend(res["head"])
                real_results["tail"].extend(res["tail"])
            else:
                real_results.add(res)
        end_time = time.perf_counter()
        # print(f"Querying all took {end_time - start_time} seconds")

        return real_results if len(real_results) > 0 else "Not Found!"


if __name__ == "__main__":
    import argparse

    # 创建参数解析器
    parser = argparse.ArgumentParser()
    # 添加命令行参数
    parser.add_argument(
        "--addr_list",
        type=str,
        required=True,
        help="路径到服务器地址列表",
    )
    # 解析命令行参数
    args = parser.parse_args()

    # 打开服务器地址列表文件
    with open(args.addr_list, "r") as f:
        # 读取服务器地址列表
        server_addrs = f.readlines()
        # 去除每个地址的换行符
        server_addrs = [addr.strip() for addr in server_addrs]
    # 打印服务器地址
    print(f"服务器地址：{server_addrs}")
    # 创建多服务器Wikidata查询客户端
    client = MultiServerWikidataQueryClient(server_addrs)
    # 打印MSFT的股票代码
    print(
        f'MSFT的股票代码是{client.query_all("get_tail_values_given_head_and_relation", "Q2283", "P249", )}'
    )
    # 退出程序
    # exit(0)
    # 存储交易所的QID
    exchange_qids = {
        "NYSE": "Q13677",
        "NASDAQ": "Q82059",
        "XSHG": "Q739514",
        "XSHE": "Q517750",
        "AMEX": "Q846626",
        "Euronext Paris": "Q2385849",
        "HKEX": "Q496672",
        "Tokyo": "Q217475",
        "Osaka": "Q1320224",
        "London": "Q171240",
    }

    # 遍历交易所
    for xchg in exchange_qids:
        xchg_name = xchg
        # 查询交易所的股票
        stocks = client.query_all(
            "get_tail_entities_given_head_and_relation",
            exchange_qids[xchg_name],
            "P414",
        )
        stocks = stocks["head"]

        # 存储股票信息
        acs = {}
        # 遍历股票
        for stock_record in tqdm(stocks):
            # 查询股票的股票代码
            ticker_id = client.query_all(
                "get_tail_values_given_head_and_relation",
                stock_record["qid"],
                "P249",
            )
            acs[stock_record["qid"]] = {
                "label": stock_record["label"],
                "ticker": ticker_id,
            }
        # 导出股票信息到文件
        import yaml

        with open(f"{xchg_name}.yaml", "w") as f:
            yaml.safe_dump(acs, f)

    # am = {}
    # # 遍历股票
    # for s in stocks:
    #     am[s] = client.query_all("qid2label", s)

    # # 导出股票信息到文件
    # import yaml
    # with open(f"{xchg_name}.yaml", "w") as f:
    #     yaml.safe_dump(am, f)

    # # 打印所有关系
    # print(client.query_all("get_all_relations_of_an_entity", "Q312",))
    # # 打印MSFT的Freebase MID
    # print(
    #     f'MSFT的Freebase MID是{client.query_all("get_external_id_given_head_and_relation", "Q2283", "P646")}'
    # )
    # # 打印MID对应的QID
    # print(
    #     f'MID /m/0k8z对应QID是{client.query_all("mid2qid", "/m/0k8z")}'
    # )
    # # 打印pid对应的label
    # print(client.query_all("label2pid", 'spouse'))  # P26

    # # 打印Carrollton对应的QID
    # print(f'Carrollton => {client.query_all("label2qid", "Carrollton")}')
    # # 打印crosses对应的pid
    # print(client.query_all("label2pid", "crosses"))

    # # 打印查询给定头部和关系的所有尾部实体
    # print(client.query_all(
    #         "get_tail_entities_given_head_and_relation", "Q6792298", "P106"
    #     )
    # )
    # # 打印查询给定头部和关系的所有尾部值
    # print(
    #     client.query_all("get_tail_values_given_head_and_relation", "Q42869", "P161")
    # )  # (Q507306, 'NASDAQ-100'), (Q180816, DJIA), ...
    # # 打印查询给定头部的所有值
    # print(
    #     client.query_all("get_all_values_of_an_entity", "Q2283")
    # )  # (P249, 'MSFT'), (P31, 'instance of'), ...
    # # 打印查询给定头部的所有关系
    # print(
    #     client.query_all("get_all_relations_of_an_entity", "Q2283")
    # )  # (P31, 'instance of'), (P361, 'part of'), ...
    # # 打印查询给定头部和关系的所有尾部实体
    # print(
    #     client.query_all("get_tail_entities_given_head_and_relation", "Q2283", "P361")
    # )  # (Q507306, 'NASDAQ-100'), (Q180816, DJIA), ...
    # # 打印查询给定头部和关系的所有尾部值
    # print(
    #     client.query_all("get_tail_values_given_head_and_relation", "Q2283", "P2139")
    # )  # MS revenue

    # # 速度测试
    # for i in tqdm(range(1000)):
    #     # 打印查询给定头部的label
    #     client.query_all("label2qid", "Microsoft")  # Q2283
    #     # 打印查询给定头部的qid
    #     client.query_all("qid2label", "Q2283")  # Microsoft
    #     # 打印查询给定头部的所有关系
    #     client.query_all("get_all_relations_of_an_entity", "Q2283")
    #     # 打印查询给定头部和关系的所有尾部实体
    #     client.query_all("get_tail_entities_given_head_and_relation", "Q2283", "P361")
    #     # 打印查询给定头部和关系的所有尾部值
    #     client.query_all("get_tail_values_given_head_and_relation", "Q2283", "P2139")
