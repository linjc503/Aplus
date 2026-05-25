from modelscope import AutoTokenizer, AutoModelForCausalLM
import torch

# 模型ID与核心初始化
model_name = "qwen/Qwen3.5-0.8B"
# 一次性初始化分词器，避免重复调用from_pretrained
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True,
    pad_token_id=AutoTokenizer.from_pretrained(model_name, trust_remote_code=True).eos_token_id
)
# 加载模型
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto",
    trust_remote_code=True
)

# 通用生成回答函数
def generate_response(prompt):
    inputs = tokenizer(prompt, return_tensors="pt", pad_token_id=tokenizer.eos_token_id).to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.3,
            do_sample=True,
            top_p=0.8,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1
        )
    # 统一解码逻辑，去掉输入内容只保留回答
    response = tokenizer.decode(outputs[0], skip_special_tokens=True).replace(prompt, "").strip()
    return response

# 交互式对话
def interactive_chat():
    print("===== 进入交互式对话（输入 exit 退出） =====")
    while True:
        prompt = input("你：")
        if prompt.lower() == "exit":
            print("对话结束！")
            break
        response = generate_response(prompt)
        print(f"AI：{response}\n")

# 主程序入口
if __name__ == "__main__":
    print("模型加载中...")
    interactive_chat()