from langchain_community.llms import LlamaCpp
from langchain_core.prompts import PromptTemplate

llm = LlamaCpp(
    model_path="Phi-3-mini-4k-instruct-q4.gguf",
    #model_path="Phi-3-mini-4k-instruct-fp16.gguf",
    n_gpu_layers=-1,
    max_tokens=500,
    n_ctx=4096,
    seed=42,
    verbose=False
)

template = """<|user|>
{input_prompt}<|end|>
<|assistant|>"""
prompt = PromptTemplate(
    template=template,
    input_variables=["input_prompt"]
)

input_text = "Explain the difference between supervised and unsupervised learning in simple terms."

chain = prompt | llm

# Run inference via chain
response = chain.invoke({"input_prompt": input_text})

print("=== Model Response ===")
print(response)
