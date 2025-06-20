import os
from tqdm import tqdm


def common_vllm_infer_func(model_path, infer_data, infer_paras):
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
    conversation = infer_data
    outputs = llm.chat(conversation, sampling_params=sampling_params, use_tqdm=True)
    return_texts = []
    for idx, output in tqdm(enumerate(outputs)):
        result = [{"text": i.text} for i in output.outputs]
        return_texts.append(result)
    return return_texts
