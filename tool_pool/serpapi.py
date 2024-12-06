from dotenv import load_dotenv
from langchain_community.utilities import SerpAPIWrapper
from tool_pool import BaseTool
from typing import Optional, Dict
from dataclasses import dataclass

load_dotenv()  # 添加这行来加载 .env 文件


@dataclass
class SearchResult:
    """搜索结果的数据类"""
    success: bool
    message: str
    content: Optional[str] = None


class SearchTool(BaseTool):
    def __init__(self, api_key: Optional[str] = None, params: Optional[Dict] = None):
        """
        初始化搜索工具。

        Args:
            api_key (str, optional): SerpAPI的API密钥
            params (Dict, optional): 搜索参数配置
        """
        self.search = SerpAPIWrapper(
            serpapi_api_key=api_key,
            params=params or {
                "engine": "google",
                "gl": "us",
                "hl": "zh-cn"
            }
        )
        super().__init__()

    def search_web(self, query: str) -> SearchResult:
        """
        执行网络搜索。

        Args:
            query (str): 搜索查询字符串

        Returns:
            SearchResult: 搜索结果
        """
        try:
            result = self.search.run(query)
            return SearchResult(
                success=True,
                message="Search completed successfully",
                content=result
            )
        except Exception as e:
            return SearchResult(
                success=False,
                message=f"Error performing search: {str(e)}"
            )

def serpapi_search(query: str):
    search_tool = SearchTool()
    
    # 测试搜索
    print("web searching...")
    result = search_tool.search_web(query)
    return result.content


if __name__ == "__main__":
    # 测试搜索工具
    # 注意：需要设置 SERPAPI_API_KEY 环境变量或在初始化时传入
    search_tool = SearchTool()
    
    # 测试搜索
    print("Testing web search:")
    result = search_tool.search_web("法国的首都在哪里")
    print(result) 