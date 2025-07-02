from ..io.file import readtxt_string, read_yaml
from tqdm import tqdm
import os
from openai import Openai
from typing import Optional, Union

"""
从你当前的项目里找到.key文件 获取url和key
"""


def call_once(
    client: Openai, input: Optional[Union[str, list]], model_name: str = "qwen3-0626-e4", max_tokens: int = 8192
) -> str:
    """
    调用LLM模型进行一次推理
    :param prompt: 输入的提示文本
    :param model_name: 模型名称
    :param max_tokens: 最大输出token数
    :return: 模型的输出文本
    """

    if isinstance(input, str):
        message = [{"role": "user", "content": input}]
    elif isinstance(input, list):
        message = input

    response = client.chat.completions.create(model=model_name, messages=message, max_tokens=max_tokens)

    return response.choices[0].message.content
