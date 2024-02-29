import json
from collections import OrderedDict

# 以utf-8编码方式打开文件A.json，读取数据并加载为json格式
with open("A.json", "r", encoding="utf-8") as file:
    data = json.load(file)

# 使用OrderedDict创建一个有序字典，以问题为键，以问题对应的字典为值
result = list(OrderedDict((item['question'], item) for item in data).values())

# 以utf-8编码方式打开文件B.json，将result以json格式写入文件
with open("B.json", "w", encoding="utf-8") as file:
    json.dump(result, file, ensure_ascii=False)
