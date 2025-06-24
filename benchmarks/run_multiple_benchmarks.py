import pandas as pd
import os
import time
import subprocess

current_dir = os.path.dirname(os.path.realpath(__file__))
print(current_dir)

# Load the model configurations from the CSV file   
model_configs_path = f'{current_dir}/configs.csv'
model_configs_df = pd.read_csv(model_configs_path)

tokenizer_mapping = {
    # Sambanova models
    # "Meta-Llama-3.2-1B-Instruct": "unsloth/Llama-3.2-1B-Instruct",
    # "Meta-Llama-3.2-3B-Instruct": "unsloth/Llama-3.2-3B-Instruct",
    # "Meta-Llama-3.1-8B-Instruct": "unsloth/Llama-3.1-8B-Instruct",
    # "Meta-Llama-3.1-70B-Instruct": "unsloth/Meta-Llama-3.1-70B-Instruct",
    # "Meta-Llama-3.3-70B-Instruct": "unsloth/Llama-3.3-70B-Instruct",
    # "Meta-Llama-3.1-405B-Instruct": "unsloth/Meta-Llama-3.1-405B-Instruct-bnb-4bit",
    # Meta models
    "meta-llama/Llama-3.2-1B-Instruct": "meta-llama/Llama-3.2-1B-Instruct",
}

# set openai api key: export OPENAI_API_KEY="..."

# Fixed parameters
backend = "vllm"
# backend = "openai-chat"
request_rate = "inf"
time_delay = 0

base_command = [
    "python", "benchmarks/benchmark_serving.py",
    "--backend", backend,
    # "--base-url", "https://api.sambanova.ai/",
    # "--base-url", "https://tnxyqiwofh6p.cloud.snova.ai/",
    "--endpoint", "/v1/completions",
    "--ignore-eos",
    f"--request-rate={request_rate}",
    
    # random parameters
    "--dataset-name", "random",
    # sonnet parameters
    # "--dataset-name", "sonnet",
    # "--dataset-path", "benchmarks/benchmarking_kit_prompt.txt",
    # "--dataset-path", "benchmarks/sonnet.txt",
    
    "--save-result",
    "--save-detailed",
    "--result-dir", "/mnt/space/rodrigom/vllm/benchmarks/results/llama3.2-1b/llama3.2-1b-tp1-bf16-nocaching-random"
]


# Iterate over all combinations
for model, input_len, output_len, num_prompts, max_concurrency in zip(model_configs_df['model_names'], model_configs_df['input_tokens'], model_configs_df['output_tokens'], model_configs_df['num_prompts'], model_configs_df['max_concurrency']):
    print(f"Running config for {model}: input_len={input_len}, output_len={output_len}, num_prompts={num_prompts}")
    command = base_command + [
        "--served_model_name", model,
        "--model", tokenizer_mapping[model],
        
        # random parameters
        "--random-input-len", str(input_len),
        "--random-output-len", str(output_len),
        # sonnet parameters
        # "--sonnet-input-len", str(input_len),
        # "--sonnet-output-len", str(output_len),
        
        "--num-prompts", str(num_prompts),
        "--max-concurrency", str(max_concurrency),
        "--result-filename", f"{backend}_{request_rate}qps_{model.replace('/','-')}_{input_len}_{output_len}_{num_prompts}_{max_concurrency}.json",
    ]
    subprocess.run(command)
    time.sleep(min(time_delay,num_prompts))  # Optional: sleep to avoid overwhelming the server
