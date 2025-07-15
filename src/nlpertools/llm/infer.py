import os
from tqdm import tqdm
from openai import OpenAI
import concurrent.futures


INFER_PARAS = {
    "temperature": 0.7,
    "infer_times": 1,
    "max_tokens": 8192,
    "top_p": 0.95,
    "top_k": 40,
    "repetition_penalty": 1.0,
}


def parse_infer_data(infer_data: list):
    if isinstance(infer_data[0], str):
        message = [{"role": "user", "content": i} for i in infer_data]
    elif isinstance(infer_data[0], list):
        message = infer_data
    return message


def common_api_infer_func(model_name, infer_data: list, infer_paras, client: OpenAI):
    """
    infer_data: list of messages/prompt
    """
    messages = parse_infer_data(infer_data)

    def get_response(model_name, messages, infer_paras):
        responses = []
        infer_times = infer_paras.get("infer_times", 1)
        for _ in range(infer_times):
            # 使用OpenAI API进行推理
            response = client.chat.completions.create(model=model_name, messages=messages, **infer_paras)
            text = response.choices[0].message.content
            responses.append({"text": text})
        return responses

    with concurrent.futures.ThreadPoolExecutor(16) as executor:
        futures = [executor.submit(get_response, model_name, message, infer_paras) for message in messages]
        results = [future.result() for future in concurrent.futures.as_completed(futures)]

    return results


def common_vllm_infer_func(model_path, infer_data: list, infer_paras: dict):
    """
    infer_data: list of messages/prompt
    """
    messages = parse_infer_data(infer_data)
    from vllm import LLM, SamplingParams

    temperature = infer_paras.get("temperature", 0.7)
    infer_times = infer_paras.get("infer_times", 1)
    vllm_card_num = len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))

    llm = LLM(model=model_path, tensor_parallel_size=vllm_card_num, trust_remote_code=True, gpu_memory_utilization=0.85)
    sampling_params = SamplingParams(
        temperature=temperature,
        n=infer_times,
        max_tokens=8192,
        # qwen3非思考模式推荐参数
        # **infer_paras.get(template_name, {}),
        # qwen3思考模式推荐参数
    )
    conversation = messages
    outputs = llm.chat(conversation, sampling_params=sampling_params, use_tqdm=True)
    return_texts = []
    for idx, output in tqdm(enumerate(outputs)):
        result = [{"text": i.text} for i in output.outputs]
        return_texts.append(result)
    return return_texts
