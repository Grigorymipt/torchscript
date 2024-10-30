import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "gpt2"  # Replace with desired model
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

# Generate text
input_text = "Hello, how can I assist you today?"
inputs = tokenizer(input_text, return_tensors="pt").to(device)
outputs = model.generate(inputs["input_ids"])
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
