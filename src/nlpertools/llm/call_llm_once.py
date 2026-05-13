from ..io.file import read_yaml
from tqdm import tqdm
import os
from typing import Optional, Union

"""
从你当前的项目里找到.key文件 获取url和key
"""


def get_client():
    """
    需要项目下有.env文件,且load_dotenv()已经被调用
    我按照我的习惯叫DEFAULT了
    """
    from openai import OpenAI

    default_api_key = os.getenv("DEFAULT_API_KEY")
    default_base_url = os.getenv("DEFAULT_BASE_URL")
    print("api_key:", default_api_key)
    print("base_url:", default_base_url)
    client = OpenAI(api_key=default_api_key, base_url=default_base_url)
    return client


def call_once_stream(
    client,
    input: Optional[Union[str, list]],
    model_name: str = "qwen3-0626-e4",
    max_tokens: int = 8192,
    temperature=0.2,
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

    completion = client.chat.completions.create(
        model=model_name,
        messages=message,
        max_tokens=max_tokens,
        stream=True,
        temperature=temperature,
    )
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
    client,
    input: Optional[Union[str, list]],
    model_name: str = "qwen3-0626-e4",
    max_tokens: int = 8192,
    temperature=0.8,
    system_prompt: str = "",
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
        if system_prompt:
            message.insert(0, {"role": "system", "content": system_prompt})
    elif isinstance(input, list):
        message = input

    response = client.chat.completions.create(
        model=model_name,
        messages=message,
        max_tokens=max_tokens,
        temperature=temperature,
    )

    # print(response)

    return response.choices[0].message.content
    response = client.response.create(
        model=model_name,
        messages=message,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return response.output_text
