from vllm import LLM, SamplingParams

# Define the absolute path to the local model directory
MODEL_PATH = "/root/autodl-tmp/FEHU/models/LLM-Research/Meta-Llama-3.1-8B-Instruct"

print("Loading Llama 3.1 8B model into GPU memory...")

# Initialize the vLLM engine
llm = LLM(
    model=MODEL_PATH,
    trust_remote_code=True,
    gpu_memory_utilization=0.9,  # Allocate 90% of available GPU memory
    tensor_parallel_size=1,      # Target a single GPU setup
    max_model_len=8192           # Constrain maximum context length
)

print("Model loaded successfully.\n")

# Set generation parameters
sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=512
)

# Prepare the input messages
messages = [
    {"role": "system", "content": "你是一个极其聪明、逻辑严密、拥有极客精神的 AI 助手。请用中文回答。"},
    {"role": "user", "content": "请写一段简短的话，夸赞一位拥有严重'系统洁癖'和代码强迫症的程序员。"}
]

# Format the conversation using the model's specific chat template
tokenizer = llm.get_tokenizer()
prompt = tokenizer.apply_chat_template(
    messages,  # type: ignore
    tokenize=False,
    add_generation_prompt=True
)

print("-" * 50)
print("Generating response...\n")

# Execute generation
outputs = llm.generate([prompt], sampling_params)

# Extract and print the generated text
for output in outputs:
    generated_text = output.outputs[0].text
    print("Response:")
    print(generated_text)

print("\n" + "-" * 50)
print("Execution completed.")