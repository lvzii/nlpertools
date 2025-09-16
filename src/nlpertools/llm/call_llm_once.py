from ..io.file import read_yaml
from tqdm import tqdm
import os
from typing import Optional, Union

"""
从你当前的项目里找到.key文件 获取url和key
"""


def call_once_stream(
    client, input: Optional[Union[str, list]], model_name: str = "qwen3-0626-e4", max_tokens: int = 8192, temperature=0.2
) -> str:
    """
    调用LLM模型进行一次推理
    :param prompt: 输入的提示文本
    :param model_name: 模型名称
    :param max_tokens: 最大输出token数
    :return: 模型的输出文本
    """
    from openai import OpenAI

    if isinstance(input, str):
        message = [{"role": "user", "content": input}]
    elif isinstance(input, list):
        message = input

    completion = client.chat.completions.create(model=model_name, messages=message, max_tokens=max_tokens, stream=True)
    text = ""
    for chunk in completion:
        if chunk.choices:
            c = chunk.choices[0].delta.content or ""
            text += c
            print(c, end="")
        else:
            print()
            print(chunk.usage)
    return text


def call_once(
    client, input: Optional[Union[str, list]], model_name: str = "qwen3-0626-e4", max_tokens: int = 8192, temperature=0.8
) -> str:
    """
    调用LLM模型进行一次推理
    :param prompt: 输入的提示文本
    :param model_name: 模型名称
    :param max_tokens: 最大输出token数
    :return: 模型的输出文本
    """
    from openai import OpenAI

    if isinstance(input, str):
        message = [{"role": "user", "content": input}]
    elif isinstance(input, list):
        message = input

    response = client.chat.completions.create(model=model_name, messages=message, max_tokens=max_tokens,temperature=temperature)

    return response.choices[0].message.content
