import re
import sys
import json
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Sequence, Union, Callable, Collection, Optional

# 将项目根目录添加到Python路径
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from llm_pool.llm import get_model_response_sync
from tool_pool.serpapi import serpapi_search

class WebSearch:
    def __init__(self, name:str='web_search', threshold:int=8000):
        self.system_prompt = """
你是一位洞察研究员。

1. 为用户查询寻找详细信息，
并尽可能简单地将内容总结为一句话
2. 如果用户的问题是关于具体数值的，
只返回数值结果，不需要任何额外解释。
"""
        self.name = name
        self.description = "用于网络搜索的工具"
        self.threshold = threshold

    def __call__(self, query:str):
        results = serpapi_search(query)
        msg = [{"role":"system","content":self.system_prompt},
               {"role":"user", "content": f"The search query {query}\nThe search results are {results}"}]
        
        answer = get_model_response_sync(model_name="deepseek-chat", messages=msg)
        return answer

class AbsDifferenceCode:
    """
    This tool is used to calculate the absolute difference between two numerical values.

    """
    def __init__(self, name:str="math_code"):
        self.system_prompt = """从用户提供的被减数和减数字符串中提取数值。"""

        self.name = name
        self.description = "使用Python代码计算减法的数学工具，比LLM更准确。"
        
    def __call__(self, concat_content:str) -> float:
        bracket_content = concat_content.split('[')[1].split(']')[0]  # 先分割得到中括号部分
        items = bracket_content.split("',")  # 再分割得到两个内容
        result1 = items[0].strip().strip("'")  # 清理字符串并赋值
        result2 = items[1].strip().strip("'")
        user_content = f"""
input parameters
<minuend>
{result1}
</minuend>

<subtrahend>
{result2}
</subtrahend>

output parameters in json format
```josn
{{
    "minuend":"the value of minuend in float",
    "subtrahend": "the value of subtrahend in float"
}}
```

"""
        messages = [
            {"role":"system", "content":self.system_prompt},
            {"role":"user", "content":user_content}
        ]
        response = get_model_response_sync(model_name="deepseek-chat", messages=messages, is_json=True)
        result_dict = json.loads(response)
        
        # calculate the absolute difference
        minuend = result_dict["minuend"]
        subtrahend = result_dict["subtrahend"]
        if minuend >= subtrahend:
            return minuend - subtrahend
        else:
            return subtrahend - minuend

class AbsDifferenceLLM:
    def __init__(self, name:str="math_llm"):
        self.system_prompt = "你是一位计算助手。"
        self.name = name
        self.description = "使用LLM计算减法的数学工具。"
        
    def __call__(self, question:str) -> str:
        user_content = f"""
根据上下文回答问题。当你写下表达式时，它必须只包含数字和运算符。以下是一些你必须遵守的指导原则：

  - 当被要求计算差值时，你要考虑差值的绝对值。两个数的差永远是正数。例如，1和2的差是1，而不是-1。
  - 当你对上下文中的多个值进行运算（如差值、求和、比率等）时，你必须统一这些数字的单位。例如，你不能将1米加到1英尺上。
     - 如果所有值都以相同单位提供，你必须选择相同单位的值。
     - 如果不是，你必须在进行运算前将它们转换为相同的单位。
  - 你必须严格遵守所要求的单位（如米、公里、百万等）。
     - 如果上下文中的数字与问题要求的单位相同，你可以直接使用它们。
     - 如果上下文中的数字与问题要求的单位不同，你必须将其转换为问题要求的单位。例如，如果问题要求以公里为单位表示两个城市之间的距离，但上下文中的距离是以英里为单位，你必须将距离转换为公里。
  - 如果要求以百万、十亿或任何其他单位表示特定数字，该数字应该不指定单位。例如，如���要求是1亿，应该写作100，而不是1亿或100,000,000。
 - 永远不要引入变量。例如，不允许使用"gazelle_max_speed * 1.4"。从给定上下文中选择正确的数字。

 问题: {question}
"""
        messages = [
            {"role":"system", "content":self.system_prompt},
            {"role":"user", "content":user_content}
        ]

        response = get_model_response_sync(model_name="deepseek-chat", messages=messages)
        return response

class Step:
    def __init__(self, 
                 idx: int, 
                 name: str, 
                 tool: callable,
                 args: Collection[Any],
                 dependencies: Collection[int]):
        self.idx = idx
        self.name = name
        self.tool = tool
        self.args = args
        self.dependencies = dependencies
        self.observation = None

    def exec(self):
        self.observation = self.tool(self.args)
        return self.observation

def plan(question: str, tools: List[Callable]) -> str:
    system_prompt = f"""
让我们首先理解问题并制定解决方案。
请以"计划："为标题输出计划，然后是编号的步骤列表。
请制定完成任务所需的最少步骤数。如果任务是一个问题，
最后一步几乎总是"根据上述采取的步骤��请回答用户的原始问题"。

每个计划应该包含以下{len(tools) + 1}种类型的动作：
"""

    for i, tool in enumerate(tools):
        system_prompt += f"{i+1}. {tool.name}: {tool.description}\n"

    # few shot
    EXAMPLE_PROMPT = (
        "问题：如果猎豹速度慢1.3倍，灵缘犬速度快1.5倍，猎鹰速度慢2.3倍，最快和最慢动物的速度比是多少？\n"
        '1. search("猎豹")\n'
        '2. math_llm("如果猎豹最大速度慢1.3倍，速度是多少公里/小时？", ["$1"])\n'
        '3. search("灵缘犬")\n'
        '4. math_llm("如果灵缘犬最大速度快1.5倍，速度是多少公里/小时？", ["$3"])\n'
        '5. search("猎鹰")\n'
        '6. math_llm("如果猎鹰最大速度慢2.3倍，速度是多少公里/小时？", ["$5"])\n'
        '7. math_llm("max($2, $4, $6) / min($2, $4, $6)")\n'
        "思考：我现在可以回答这个问题了。\n"
        "###\n"
        "\n"
        "问题：如果珠穆朗玛峰的高度减半，乞力马扎罗山的高度翻倍，它们的高度差是多少？\n"
        '1. search("珠穆朗玛峰")\n'
        '2. math_llm("珠穆朗玛峰高度的一半是多少米？", ["$1"])\n'
        '3. search("乞力马扎罗山")\n'
        '4. math_llm("乞力马扎罗山高度的两倍是多少米？", ["$3"])\n'
        '5. math_llm("abs($2 - $4)")\n'
        "思考：我现在可以回答这个问题了。\n"
        "###\n"
        "\n"
        "问题：如果撒哈拉沙漠的面积减少33%，而卡拉哈里沙漠的面积增加52%，哪个面积更大？\n"
        '1. search("撒哈拉沙漠")\n'
        '2. math_llm("如果撒哈拉沙漠面积减少33%，面积是多少平方公里？", ["$1"])\n'
        '3. search("卡拉哈里沙漠")\n'
        '4. math_llm("如果卡拉哈里沙漠面积增加52%，面积是多少平方公里？", ["$3"])\n'
        "思考：我可以直接比较这些数字而不需要调用数学工具。\n"
        "###\n"
        "\n"
        "问题：确定哪个值更小：马里亚纳海沟和波多黎各海沟的深度差（米），还是南桑威奇海沟和巽他海沟的深度差（米）？\n"
        '1. search("马里亚纳海沟")\n'
        '2. search("波多黎各海沟")\n'
        '3. math_llm("马里亚纳海沟和波多黎各海沟的深度差的绝对值是多少米？", ["$1", "$2"])\n'
        '4. search("南桑威奇海沟")\n'
        '5. search("巽他海沟")\n'
        '6. math_llm("南桑威奇海沟和巽他海沟的深度差的绝对值��多少米？", ["$4", "$5"])\n'
        '7. math_llm("min($3, $6)")\n'
        "思考：我现在可以回答这个问题了。\n"
        "###\n"
        "\n"
        "问题：珠穆朗玛峰和乞力马扎罗山的高度比是多少？\n"
        '1. search("珠穆朗玛峰")\n'
        '2. search("乞力马扎罗山")\n'
        '3. math_llm("珠穆朗玛峰高度除以乞力马扎罗山高度", ["$1", "$2"])\n'
        "思考：我现在可以回答这个问题了。\n"
        "###\n"
    )

    # the ahove prompt is refer to llm compiler
    system_prompt = system_prompt + '\n' + EXAMPLE_PROMPT

    messages = [
        {"role":"system", "content":system_prompt},
        {"role":"user", "content":question}
    ]

    response = get_model_response_sync(model_name="deepseek-chat", messages=messages)
    return response

def plan_preset(question: str) -> str:
    """
    预先设置的plan，用于测试
    """
    plans_math_code= """
计划：
1. search('多伦多2023年���口')
2. search('纽约市2023年人口')
3. math_code('纽约市和多伦多的人口差值', ['$1', '$2'])
思考：我现在可以回答这个问题了。
"""
    plans_math_llm = """
计划：
1. search('多伦多2023年人口')
2. search('纽约市2023年人口')
3. math_llm('纽约市和多伦多的人口差值', ['$1', '$2'])
思考：我现在可以回答这个问题了。
"""
    return plans_math_code

def execute(plans: str, tools: List[Callable]) -> List[str]:
    ACTION_PATTERN = r"\n*(\d+)\. (\w+)\((.*)\)(\s*#\w+\n)?"
    # 这个正则表达式包含四个捕获组：
    # 1. (\d+) 捕获步骤编号
    # 2. (\w+) 捕获函数名
    # 3. (.*) 捕获函数参数
    # 4. (\s*#\w+\n)? 是可选的，用于捕获可能存在的注释
    ID_PATTERN = r"\$\{?(\d+)\}?"

    def default_dependency_rule(idx, args: str):
        """
        检查给定的索引是否存在于参数字符串中的数字引用中。

        此函数用于确定某个动作或步骤（由idx表示）是否被其他步骤引用或依赖。
        它可能是规划和执行系统的一部分，用于确定任务之间的依赖关系。

        参数：
            idx (int): 要检查的整数索引
            args (str): 包含潜在数字引用的字符串

        返回：
            bool: 如果在args中找到idx的数字引用则返回True，否则返回False
        """
        matches = re.findall(ID_PATTERN, args)
        numbers = [int(match) for match in matches]
        return idx in numbers

    def find_tool(tool_name:str, tools: Sequence[Callable]=tools):
        for tool in tools:
            if tool.name == tool_name:
                return tool

    matches = re.findall(ACTION_PATTERN, plans)
    steps = []

    for item in matches:
        idx, tool_name, tool_args, _ = item
        idx = int(idx)
        tool = find_tool(tool_name)
        dependencies = [i for i in range(1, idx) if default_dependency_rule(i, tool_args)]
        step = Step(
            idx=idx,
            name=tool_name,
            tool=tool,
            args=tool_args,
            dependencies=dependencies
        )
        steps.append(step)

    results = []
    for step in steps:
        for dependency in sorted(step.dependencies, reverse=True):
            for arg_mask in ["${" + str(dependency) + "}", "$" + str(dependency)]:
                if arg_mask in step.args:
                    if steps[dependency-1].observation is not None:
                        step.args = step.args.replace(
                            arg_mask, str(steps[dependency-1].observation)
                        )
        result = step.exec()
        results.append(result)
        print(f"步骤 {step.idx} 执行: {result}")
    return results

def synthesize(question: str, execution_results: List[str]) -> str:
    synthesis_system_prompt = "你是一位能够分析执行结果并回答问题的助手。"
    synthesis_user_prompt = f"""
    基于以下执行结果和原始问题，请提供简洁的答案：

    原始问题：{question}

    执行结果：
    {execution_results}

    请综合信息并回答原始问题。
    """

    breakpoint()
    synthesis_messages = [
        {'role': 'system', 'content': synthesis_system_prompt},
        {'role': 'user', 'content': synthesis_user_prompt}
    ]

    final_answer = get_model_response_sync(model_name="deepseek-chat", messages=synthesis_messages)
    return final_answer

def plan_and_execute(question: str, need_synthesize: bool=True) -> str:
    # 定义工具
    search_tool = WebSearch(name="search")
    math_llm_tool = AbsDifferenceLLM(name="math_llm")
    math_code_tool = AbsDifferenceCode(name="math_code")
    tools = [search_tool, math_llm_tool, math_code_tool]

    # 1. 规划
    # plans = plan(question, tools)
    plans = plan_preset(question)
    print(f"计划：\n{plans}")

    # 2. 执行
    execution_results = execute(plans, tools)

    # 3. 综合结果
    if need_synthesize:
        final_answer = synthesize(question, execution_results)
    else:
        final_answer = execution_results[-1]
    return final_answer

def agent_search_and_absdiff(question: str) -> str:
    result = plan_and_execute(question, need_synthesize=False)
    print("最终答案:")
    print(result)

if __name__ == "__main__":
    question = '多伦多和纽约市之间的人口差距是多少？'
    agent_search_and_absdiff(question)
