# ToG

成功安装所有必要的配置后，可以使用以下命令直接执行ToG：
```sh
python main_freebase.py \  # if you wanna use Wikidata as KG source, run main_wiki.py
--dataset cwq \ # dataset your wanna test, see ToG/data/README.md
--max_length 256 \ 
--temperature_exploration 0.4 \ # the temperature in exploration stage.
--temperature_reasoning 0 \ # the temperature in reasoning stage.
--width 3 \ # choose the search width of ToG, 3 is the default setting.
--depth 3 \ # choose the search depth of ToG, 3 is the default setting.
--remove_unnecessary_rel True \ # whether removing unnecessary relations.
--LLM_type gpt-3.5-turbo \ # the LLM you choose
--opeani_api_keys sk-xxxx \ # your own api keys, if LLM_type == llama, this parameter would be rendered ineffective.
--num_retain_entity 5 \ # Number of entities retained during entities search.
--prune_tools llm \ # prune tools for ToG, can be llm (same as LLM_type), bm25 or sentencebert.
```

实验中使用的所有修剪和推理提示都在“prompt_list.py”文件中。
有关eval，请参阅`eval/README.md`文件。