import asyncio
import sys
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

# 将项目根目录添加到Python路径
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from llm_pool.llm import get_model_response


@dataclass
class TestConfig:
    """测试配置类"""
    models: List[str] = (
        "deepseek-chat",
        "mixtral-8x7b-32768", 
        "Qwen/Qwen2-72B-Instruct",
        "gpt-4o"
    )
    system_prompt: str = "You are a helpful assistant"


@dataclass
class TestCase:
    """测试用例类"""
    question: str
    use_cot: bool

    def get_prompt(self) -> str:
        """获取完整的提示词"""
        if self.use_cot:
            return f"{self.question}\nA: Let's think step by step."
        return self.question


class ChainOfThoughtTester:
    """思维链测试器"""
    
    def __init__(self, config: Optional[TestConfig] = None):
        self.config = config or TestConfig()
        
    async def test_single_question(self, model_name: str, test_case: TestCase) -> str:
        """测试单个问题"""
        messages = [
            {"role": "system", "content": self.config.system_prompt},
            {"role": "user", "content": test_case.get_prompt()},
        ]
        
        return await get_model_response(model_name=model_name, messages=messages)
    
    async def test_model(self, model_name: str) -> Dict[str, str]:
        """测试单个模型的所有问题"""
        results = {}
        
        for i, test_case in enumerate(self.get_test_cases(), 1):
            result = await self.test_single_question(model_name, test_case)
            results[f"result{i}"] = result
            
        return results
    
    @staticmethod
    def get_test_cases() -> List[TestCase]:
        """获取所有测试用例"""
        return [
            TestCase(
                "Q: Roger has 5 tennis balls. He buys 2 more cans of tennis balls. Each can has 3 tennis balls. How many tennis balls does he have now?",
                False
            ),
            TestCase(
                "Q: Roger has 5 tennis balls. He buys 2 more cans of tennis balls. Each can has 3 tennis balls. How many tennis balls does he have now?",
                True
            ),
            TestCase(
                "The cafeteria had 23 apples. If they used 20 to make lunch and bought 6 more, how many apples do they have?",
                False
            ),
            TestCase(
                """Q: Roger has 5 tennis balls. He buys 2 more cans of tennis balls.
Each can has 3 tennis balls. How many tennis balls does he have now?
A: Roger started with 5 balls. 2 cans of 3 tennis balls
each is 6 tennis balls. 5 + 6 = 11. The answer is 11.
Q: The cafeteria had 23 apples.
If they used 20 to make lunch and bought 6 more, how many apples do they have?""",
                False
            ),
            TestCase(
                """Q: Roger has 5 tennis balls. He buys 2 more cans of tennis balls.
Each can has 3 tennis balls. How many tennis balls does he have now?
A: Let's think step by step. Roger started with 5 balls. 2 cans of 3 tennis balls
each is 6 tennis balls. 5 + 6 = 11. The answer is 11.
Q: The cafeteria had 23 apples.
If they used 20 to make lunch and bought 6 more, how many apples do they have?""",
                True
            ),
            TestCase("Q: 9.11 and 9.8, which is bigger?", False),
            TestCase("Q: 9.11 and 9.8, which is bigger?", True),
            TestCase(
                """Q: Roger has 5 tennis balls. He buys 2 more cans of tennis balls.
Each can has 3 tennis balls. How many tennis balls does he have now?
A: Roger started with 5 balls. 2 cans of 3 tennis balls
each is 6 tennis balls. 5 + 6 = 11. The answer is 11.
Q: 9.11 and 9.8, which is bigger?""",
                False
            ),
            TestCase(
                """Q: Roger has 5 tennis balls. He buys 2 more cans of tennis balls.
Each can has 3 tennis balls. How many tennis balls does he have now?
A: Let's think step by step. Roger started with 5 balls. 2 cans of 3 tennis balls
each is 6 tennis balls. 5 + 6 = 11. The answer is 11.
Q: 9.11 and 9.8, which is bigger?""",
                True
            )
        ]


class ResultManager:
    """结果管理器"""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
    
    def save_results(self, results: Dict[str, Dict[str, str]]):
        """保存测试结果到JSON文件"""
        output_file = self.output_dir / "cot_results.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"结果已保存到 {output_file}")


async def main():
    # 初始化测试器和结果管理器
    tester = ChainOfThoughtTester()
    result_manager = ResultManager(Path(__file__).parent)
    all_results = {}

    # 测试所有模型
    for model in tester.config.models:
        print(f"测试模型: {model}")
        try:
            model_results = await tester.test_model(model)
            all_results[model] = model_results
            print(f"完成测试 {model}")
        except Exception as e:
            print(f"测试 {model} 时出错: {str(e)}")
        print("-" * 50)

    # 保存结果
    result_manager.save_results(all_results)


if __name__ == "__main__":
    asyncio.run(main())
