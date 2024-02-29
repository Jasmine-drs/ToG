# Datasets

当前文件夹中保存了我们使用的所有数据集，论文中使用的数据集统计如下表所示：

| Dataset             | Answer Format | Train     | Test  | Licence     | Mid (For Freebase) | Qid (For Wikidata) |
|---------------------|---------------|-----------|-------|-------------|--------------------|--------------------|
| ComplexWebQuestions | Entity        | 27,734    | 3,531 | -           | √                  | √                  |
| WebQSP              | Number        | 3,098     | 1,639 | CC License  | √                  | √                  |
| GrailQA*            | Entity/Number | 44,337    | 1,000 | -           | √                  |                    |
| QALD-10             | Entity/Number | -         | 333   | MIT License |                    | √                  |
| Simple Question*    | Number        | 14,894    | 1,000 | CC License  | √                  |                    |
| WebQuestions        | Entity/Number | 3,778     | 2,032 | -           | √                  | √                  |
| T-REx               | Entity        | 2,284,168 | 5,000 | MIT License |                    | √                  |
| Zero-Shot RE        | Entity        | 147,909   | 3,724 | MIT License |                    | √                  |
| Creak               | Bool          | 10,176    | 1,371 | MIT License |                    | √                  |

其中*表示由于测试样本丰富，我们从GrailQA和Simple Questions测试集中随机选择了1000个样本来组成测试集。
如果用户想使用不同的KG源进行搜索，请查看简单wikidata db文件夹的“mid2qid”和“qid2mid”API。我们已经将实验中使用的mid和qid放入数据集中。