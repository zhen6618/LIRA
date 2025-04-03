from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig

model_path = "/root/paddlejob/workspace/env_run/zhouzhen05/Data_SSD3/LISA_checkpoint/Qwen-7B-Chat"

# Model names: "Qwen/Qwen-7B", "Qwen/Qwen-14B" 
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
# use bf16
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-7B", device_map="auto", trust_remote_code=True, bf16=True).eval()
# use fp16
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-7B", device_map="auto", trust_remote_code=True, fp16=True).eval()
# use cpu only
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-7B", device_map="cpu", trust_remote_code=True).eval()
# use auto mode, automatically select precision based on the device.
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    trust_remote_code=True,
    bf16=True,
).eval()

# Specify hyperparameters for generation. But if you use transformers>=4.32.0, there is no need to do this.
model.generation_config = GenerationConfig.from_pretrained(model_path, trust_remote_code=True)

# inputs = tokenizer('蒙古国的首都是乌兰巴托(Ulaanbaatar)\n冰岛的首都是雷克雅未克(Reykjavik)\n中国的首都是', return_tensors='pt')
# inputs = tokenizer("There are three chairs with the following 3D coordinates: A: [1, 1, 1], B: [2, 2, 2], C: [4, 4, 4]. Which chair is closer to B? Please think step by step (if distance calculation is needed, use Euclidean distance) and provide the final answer in the format: [Answer].", return_tensors='pt')
inputs = tokenizer('There are three objects in the room now: bed A, chair B, and toilet C. Identify the object that can be used to lie down and rest.', return_tensors='pt')

inputs = inputs.to(model.device)
pred = model.generate(**inputs)
print(tokenizer.decode(pred.cpu()[0], skip_special_tokens=True))
# 蒙古国的首都是乌兰巴托（Ulaanbaatar）\n冰岛的首都是雷克雅未克（Reykjavik）\n埃塞俄比亚的首都是亚的斯亚贝巴（Addis Ababa）...



"ModelScope"
# from modelscope import AutoModelForCausalLM, AutoTokenizer
# from modelscope import GenerationConfig

# # Model names: "qwen/Qwen-7B-Chat", "qwen/Qwen-14B-Chat"
# tokenizer = AutoTokenizer.from_pretrained("qwen/Qwen-7B-Chat", trust_remote_code=True)
# model = AutoModelForCausalLM.from_pretrained("qwen/Qwen-7B-Chat", device_map="auto", trust_remote_code=True, fp16=True).eval()
# model.generation_config = GenerationConfig.from_pretrained("Qwen/Qwen-7B-Chat", trust_remote_code=True) # 可指定不同的生成长度、top_p等相关超参

# response, history = model.chat(tokenizer, "你好", history=None)
# print(response)
# response, history = model.chat(tokenizer, "浙江的省会在哪里？", history=history) 
# print(response)
# response, history = model.chat(tokenizer, "它有什么好玩的景点", history=history)
# print(response)

