MODEL_PATH='longcat_omni'

import os
import json
import asyncio

import torch
from transformers import AutoTokenizer
from safetensors import safe_open
from vllm import LLM
from vllm.sampling_params import SamplingParams

# 加载embedding权重
def load_embedding_weights(model_path):
    # 查找safetensors文件
    safetensor_files = [f for f in os.listdir(model_path) if f.endswith(".safetensors")]
    if not safetensor_files:
        raise FileNotFoundError(f"No safetensor files found in {model_path}")

    # 从第一个文件加载embedding权重
    safetensor_file = os.path.join(model_path, safetensor_files[0])
    with safe_open(safetensor_file, framework="pt") as f:
        # 尝试常见的embedding层名称
        possible_names = [
            "model.embed_tokens.weight",
            "embed_tokens.weight",
            "model.embeddings.word_embeddings.weight",
            "transformer.wte.weight"
        ]

        for name in possible_names:
            if name in f.keys():
                return f.get_tensor(name)

        # 如果都没找到，打印可用的keys
        print(f"Available keys: {list(f.keys())}")
        raise KeyError("Could not find embedding weights")

# 生成input_ids和input_embeds的函数
def generate_input_ids_and_embeds(prompts, tokenizer, embedding_layer):
    all_input_ids = []
    all_input_embeds = []

    for prompt in prompts:
        # 生成input_ids
        input_ids = tokenizer.encode(prompt, return_tensors="pt")

        # 生成input_embeds
        with torch.no_grad():
            input_embeds = embedding_layer(input_ids).squeeze(0)

        all_input_ids.append(input_ids.squeeze().tolist())
        all_input_embeds.append(input_embeds)

    return all_input_ids, all_input_embeds

async def main():
    prompts = [
        "Hello, my name is",
        "USER:请将一个关于狗的笑话,尽量长一点,输出语音:\nVOICE ASSISTANT:",
        "USER:写一首诗\nVOICE ASSISTANT:",
        # "The president of the United States is",
        # "The capital of France is",
        # "USER:The capital of France is\nVOICE ASSISTANT:",
        # "The future of AI is",
        # "please introduce yourself."
    ]

    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    # 加载embedding权重并创建embedding层
    embedding_weights = load_embedding_weights(MODEL_PATH)
    embedding_layer = torch.nn.Embedding.from_pretrained(embedding_weights, freeze=True)

    # 生成所有prompts的input_ids和input_embeds
    input_ids_list, input_embeds_list = generate_input_ids_and_embeds(prompts, tokenizer, embedding_layer)

    # 为了兼容原代码，我们使用第一个prompt的结果
    prompt=prompts[0]
    input_ids = input_ids_list[0]
    input_embeds0 = input_embeds_list[0]
    input_embeds1 = input_embeds_list[1]
    print(f"Generated input_ids shape: {len(input_ids)}")
    print(f"Generated input_embeds shape: {input_embeds0.shape}")
    print(f"Input_ids: {input_ids}")
    print(f"Input_embeds: {input_embeds0}")

    # sampling_params = {"temperature": 0.2, "top_p": 0.1, "max_new_tokens": 32, "ignore_eos":True} # 

    llm = LLM(
        model=MODEL_PATH,
        tensor_parallel_size=8,
        enable_expert_parallel=True,
        enforce_eager=True,
        trust_remote_code=True,
        enable_prompt_embeds=True,
        hf_overrides={
            "hf_path" : MODEL_PATH, 
            "num_multi_ids": 4,
            "audio_head_num": 4,
            "audio_vocab_size": 8224,
            "audio_embed_pt": MODEL_PATH + "/audio/audio_embeddings.pt",
            "audio_id_offset": 32,
            "audio_rep_penalty_window": 30,
            "text_rep_penalty_window": 30,
            "audio_output_layer_pt": MODEL_PATH + "/audio/audio_output_layers.pt",
            "has_proj": False,
            "audio_repetition_penalty" : 1.1,
        },
    )

    sampling_params = SamplingParams(top_k=1, max_tokens=2048, ignore_eos=True)

    inputs = []
    for i in range(3):
        inputs.append({"prompt_embeds": input_embeds_list[i]})
    outputs = llm.generate(inputs, sampling_params)
    # Print the outputs.
    print("-" * 50)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        generated_text_tokens = output.outputs[0].token_ids
        generated_audio_tokens = output.outputs[0].audio_token_ids
        print(f"Prompt: {prompt!r}\nGenerated text: {generated_text!r}")
        print(f"Generated text tokens: {generated_text_tokens!r}")
        print(f"Generated audio tokens: {generated_audio_tokens!r}")
        print("-" * 50)
        generated_aux_infos = output.outputs[0].aux_output_infos
        print(f"Generated aux_infos: {generated_aux_infos!r}")
        print("-" * 50)


if __name__ == '__main__':
    asyncio.run(main())