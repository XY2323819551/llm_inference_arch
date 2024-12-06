import re
import sys
import json
import asyncio
from pathlib import Path
from typing import List, Dict, Optional, Union, Callable

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
               {"role":"user", "content": f"查询内容是：{query}，搜索结果是：{results}"}]
        
        answer = get_model_response_sync(model_name="deepseek-chat", messages=msg)
        return answer


def format_message(messages: List[Dict], last_content_length: int = 0) -> int:
    """
    格式化打印新增的消息内容
    
    Args:
        messages: 消息列表
        last_content_length: 上次打印的内容长度
        
    Returns:
        int: 当前内容的总长度
    """
    latest_content = messages[-1]["content"]
    # 只获取新增的内容
    new_content = latest_content[last_content_length:]
    if new_content:
        print(new_content, end='', flush=True)
    return len(latest_content)

def react(question: str, tools: List[Callable]) -> str:
    react_prompt = """
你是一个中文AI助手。除了格式关键词外，请始终使用中文回复。
尽可能好地回答以下问题。你可以使用以下工具：

{tools}

使用以下格式：

Question: 你必须回答的输入问题
Thought: 你应该始终用中文思考下一步该做什么
Action: 要采取的行动，必须是[{tool_names}]其中之一
Action Input: 行动的输入内容
Observation: 行动的结果
...（这个Thought/Action/Action Input/Observation可以重复N次）
Thought: 我现在知道最终答案了
Final Answer: 对原始输入问题的最终答案

Begin!

Question: {input}
"""

    tool_descriptions = "\n".join([f"{tool.name}: {tool.description}" for tool in tools])
    tool_names = ", ".join([tool.name for tool in tools])
    
    user_prompt = react_prompt.format(
        tools=tool_descriptions,
        tool_names=tool_names,
        input=question
    )

    messages = [{"role": "user", "content": user_prompt}]
    last_content_length = 0
    
    while True:
        last_content_length = format_message(messages, last_content_length)
        response = get_model_response_sync(model_name="deepseek-chat", messages=messages, stop=["Observation", " Observation"])
        messages[-1]["content"] += response
        
        if "Final Answer:" in response:
            break
        
        regex = r"Action: \[(.*?)\][\s]*Action Input: (.*?)(?:\n|$)"
        action_match = re.search(regex, response, re.DOTALL)
        
        if action_match:
            action = action_match.group(1)
            action_input = action_match.group(2).strip()
            
            tool = next((t for t in tools if t.name == action), None)
            if tool:
                observation = tool(action_input)
                messages[-1]["content"] += f"\nObservation: {observation}\nThought: "
    
    # 打印最后的新内容
    format_message(messages, last_content_length)
    
    final_answer = re.search(r"Final Answer:(.*)", response, re.DOTALL)
    return final_answer.group(1).strip() if final_answer else "未找到最终答案。"

def main():
    query = "2024年欧洲杯和2024年美洲杯冠军"
    print("\n🚀 Starting new query:", query)
    
    search_tool = WebSearch()
    tools = [search_tool]
    
    result = react(query, tools)
    print("最终答案：")
    print(result)


if __name__ == "__main__":
    main()

