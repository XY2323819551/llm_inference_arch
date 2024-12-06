import asyncio
import sys
import logging
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass

# 配置日志格式
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# 将项目根目录添加到Python路径
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from llm_pool.llm import get_model_response

@dataclass
class TranslationConfig:
    """翻译配置类"""
    model_name: str = "deepseek-chat"
    source_language: str = "英语"  # 源语言
    target_language: str = "中文"  # 目标语言
    
    @property
    def language_pair(self) -> str:
        """返回语言对描述"""
        return f"{self.source_language}到{self.target_language}"


class TranslationPrompts:
    """翻译提示词管理类"""
    
    @staticmethod
    def get_translator_system_prompt(config: TranslationConfig) -> str:
        return f"你是一位专业的语言学家，专门从事{config.language_pair}的翻译工作。"
    
    @staticmethod
    def get_reflection_system_prompt(config: TranslationConfig) -> str:
        return f"""你是一位专业的语言学家，专门从事{config.language_pair}的翻译工作。\
你的任务是改进给定的翻译。"""
    
    @staticmethod
    def get_refiner_system_prompt(config: TranslationConfig) -> str:
        return f"你是一位专业的语言学家，专门从事{config.target_language}翻译编辑工作。"
    
    @staticmethod
    def get_translation_prompt(source_text: str, config: TranslationConfig) -> str:
        return f"""这是一个{config.language_pair}的翻译任务，请为以下文本提供{config.target_language}翻译。\
请只提供翻译结果，不要包含任何解释或其他文本。
{config.source_language}: {source_text}

{config.target_language}:"""

    @staticmethod
    def get_reflection_prompt(source_text: str, translation: str, config: TranslationConfig) -> str:
        return f"""你的任务是仔细阅读源文本和其翻译，然后给出建设性的批评和有用的建议来改进翻译。

源文本和初始翻译用XML标签标记如下：

<SOURCE_TEXT>
{source_text}
</SOURCE_TEXT>

<TRANSLATION>
{translation}
</TRANSLATION>

在提供建议时，请注意是否有以下方面可以改进：
(i) 准确性（通过纠正添加、误译、遗漏或未翻译的文本），
(ii) 流畅性（通过应用{config.target_language}语法、拼写和标点规则，确保没有不必要的重复），
(iii) 风格（通过确保翻译反映源文本的风格并考虑文化背景），
(iv) 术语（通过确保术语使用的一致性并反映源文本领域；确保使用恰当的{config.target_language}习语）。

请列出具体的、有帮助的和建设性的建议来改进翻译。
每个建议应针对翻译的一个具体部分。
只输出建议，不要输出其他内容。"""

    @staticmethod
    def get_refinement_prompt(source_text: str, translation: str, reflection: str, config: TranslationConfig) -> str:
        return f"""你的任务是仔细阅读并编辑一个{config.language_pair}的翻译，同时考虑专家提供的建议和建设性批评。

源文本、初始翻译和专家语言学家的建议用XML标签标记如下：

<SOURCE_TEXT>
{source_text}
</SOURCE_TEXT>

<TRANSLATION>
{translation}
</TRANSLATION>

<EXPERT_SUGGESTIONS>
{reflection}
</EXPERT_SUGGESTIONS>

请在编辑翻译时考虑专家建议。编辑翻译时请确保：

(i) 准确性（通过纠正添加、误译、遗漏或未翻译的文本），
(ii) 流畅性（通过应用{config.target_language}语法、拼写和标点规则，确保没有不必要的重复），
(iii) 风格（通过确保翻译反映源文本的风格），
(iv) 术语（避免上下文不当、使用不一致），
(v) 其他错误。

只输出新的翻译，不要输出其他内容。"""


class SelfRefiningTranslator:
    """自优化翻译器类"""
    def __init__(self, config: Optional[TranslationConfig] = None):
        self.config = config or TranslationConfig()
        self.prompts = TranslationPrompts()
        logger.info(f"初始化翻译器 - 翻译方向: {self.config.language_pair}")
    
    async def _get_llm_response(self, system_prompt: str, user_prompt: str) -> str:
        """获取LLM响应的通用方法"""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        logger.debug(f"发送请求到模型: {self.config.model_name}")
        return await get_model_response(model_name=self.config.model_name, messages=messages)
    
    async def initial_translation(self, source_text: str) -> str:
        """执行初始翻译"""
        logger.info("开始第一阶段: 初始翻译")
        logger.debug(f"源文本: {source_text[:100]}...")
        
        translation = await self._get_llm_response(
            self.prompts.get_translator_system_prompt(self.config),
            self.prompts.get_translation_prompt(source_text, self.config)
        )
        
        logger.info("完成初始翻译")
        logger.debug(f"初始翻译结果: {translation[:100]}...")
        return translation
    
    async def get_translation_feedback(self, source_text: str, translation: str) -> str:
        """获取翻译反馈"""
        logger.info("开始第二阶段: 获取翻译反馈")
        
        feedback = await self._get_llm_response(
            self.prompts.get_reflection_system_prompt(self.config),
            self.prompts.get_reflection_prompt(source_text, translation, self.config)
        )
        
        logger.info("完成翻译反馈")
        logger.debug(f"反馈内容: {feedback[:100]}...")
        return feedback
    
    async def refine_translation(self, source_text: str, translation: str, feedback: str) -> str:
        """优化翻译"""
        logger.info("开始第三阶段: 优化翻译")
        
        refined_translation = await self._get_llm_response(
            self.prompts.get_refiner_system_prompt(self.config),
            self.prompts.get_refinement_prompt(source_text, translation, feedback, self.config)
        )
        
        logger.info("完成优化翻译")
        logger.debug(f"优化后的翻译: {refined_translation[:100]}...")
        return refined_translation
    
    async def translate(self, source_text: str) -> str:
        """执行完整的自优化翻译流程"""
        logger.info("开始完整翻译流程")
        logger.info("-" * 50)
        
        # 第一阶段：初始翻译
        initial_translation = await self.initial_translation(source_text)
        logger.info("\n初始翻译结果:")
        logger.info(f"{initial_translation}\n")
        logger.info("-" * 50)
        
        # 第二阶段：获取反馈
        feedback = await self.get_translation_feedback(source_text, initial_translation)
        logger.info("\n翻译反馈:")
        logger.info(f"{feedback}\n")
        logger.info("-" * 50)
        
        # 第三阶段：优化翻译
        final_translation = await self.refine_translation(source_text, initial_translation, feedback)
        logger.info("\n最终翻译结果:")
        logger.info(f"{final_translation}\n")
        logger.info("-" * 50)
        
        logger.info("翻译流程完成")
        return final_translation


async def main():
    # 设置更详细的日志级别（如果需要）
    logger.setLevel(logging.DEBUG)
    
    en_to_zh_config = TranslationConfig(
        source_language="英语",
        target_language="中文"
    )
    
    source_text = """
    Last week, I spoke about AI and regulation at the U.S. Capitol at an event that was attended by legislative and business leaders. I'm encouraged by the progress the open source community has made fending off regulations that would have stifled innovation. But opponents of open source are continuing to shift their arguments, with the latest worries centering on open source's impact on national security. I hope we'll all keep protecting open source!

    Based on my conversations with legislators, I'm encouraged by the progress the U.S. federal government has made getting a realistic grasp of AI's risks. To be clear, guardrails are needed. But they should be applied to AI applications, not to general-purpose AI technology.
    """




    en_to_zh_config = TranslationConfig(
        source_language="中文",
        target_language="英语"
    )
    
    source_text = """
昨夜雨疏风骤，浓睡不消残酒。试问卷帘人，却道海棠依旧，知否？知否？应是绿肥红瘦。
"""
    
    translator = SelfRefiningTranslator(en_to_zh_config)
    result = await translator.translate(source_text)
    
    # 最终结果已经在日志中打印，这里可以选择是否再次打印
    # print("\n最终翻译:")
    # print(result)


if __name__ == "__main__":
    asyncio.run(main())
