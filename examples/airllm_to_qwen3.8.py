from airllm import AutoModel

MAX_LENGTH = 128
# just pass a hugging face repo id — works with almost any popular model:
# model = AutoModel.from_pretrained("Qwen/Qwen3-32B")

# go bigger with the exact same one line:
# model = AutoModel.from_pretrained("Qwen/Qwen3.8-27B")          # 27B dense VL, 3.33GB
model = AutoModel.from_pretrained("Qwen/Qwen3.8-Flash-Next")    # 125B MoE + 51B PLE, 5.95GB
#model = AutoModel.from_pretrained("Qwen/Qwen3-235B-A22B")     # 235B, runs in ~3GB
#model = AutoModel.from_pretrained("deepseek-ai/DeepSeek-V3")  # 671B, runs in ~12GB

# or use a model's local path...
#model = AutoModel.from_pretrained("/home/ubuntu/.cache/huggingface/hub/models--Qwen--Qwen3-32B/snapshots/...")

input_text = [
        'What is the capital of United States?',
        #'I like',
    ]

input_tokens = model.tokenizer(input_text,
    return_tensors="pt", 
    return_attention_mask=False, 
    truncation=True, 
    max_length=MAX_LENGTH, 
    padding=False)
           
generation_output = model.generate(
    input_tokens['input_ids'].cuda(), 
    max_new_tokens=20,
    use_cache=True,
    return_dict_in_generate=True)

output = model.tokenizer.decode(generation_output.sequences[0])

print(output)
