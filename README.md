<div align="center">
  <a href="https://github.com/katanemo/Arch-Function/tree/main"><img width="75%" height="auto" src="./assets/icon.jpeg"></a>

**Arch-Function: Advanced Function Calling Models**

</div>


<div align="center">

   [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-blue)](https://huggingface.co/katanemo)
   [![Discord](https://img.shields.io/badge/Discord-%235865F2.svg?&logo=discord&logoColor=white)](https://discord.gg/pGZf2gcwEc)
   [![License](https://img.shields.io/badge/License-Apache-green.svg)]()

</div>


**Arch-Function** represents a comprehensive research and development initiative focused on creating state-of-the-art function calling capabilities in large language models. Our mission is to build AI systems that can seamlessly understand, interpret, and execute complex function calls with unprecedented accuracy and reliability.

This project encompasses multiple model families specifically engineered for function calling tasks, designed to understand complex function signatures, identify required parameters, and produce accurate function call outputs based on natural language prompts. The current release includes three major collections with models available in multiple sizes, with additional breakthrough models planned for future releases that will further advance the state-of-the-art in function calling capabilities.

## 📰 News & Updates

- **June 2025**: 🏆🏆🏆 [Arch-Agent collection](https://huggingface.co/collections/katanemo/arch-function-chat-67e6feb6e33793d82adeded1)released for advanced multi-turn, multi-step workflow automation, achieving Top-3 performance on the [BFCL Leaderboard](https://gorilla.cs.berkeley.edu/leaderboard.html#leaderboard)!
- **February 2025**: 🚀🚀🚀 [Arch-Function-Chat collection](https://huggingface.co/collections/katanemo/arch-function-chat-67e6feb6e33793d82adeded1) launched with conversational function calling capabilities
- **Dec 2024**: 🔥🔥🔥 Complete model suite updated with latest improvements across all sizes for [Arch-Function collection](https://huggingface.co/collections/katanemo/arch-function-66f209a693ea8df14317ad68)
- **Sep 2024**: 🏆🏆🏆 [Arch-Function collection](https://huggingface.co/collections/katanemo/arch-function-66f209a693ea8df14317ad68) officially launched on Hugging Face, achieving Top-7 performance on the [BFCL Leaderboard](https://gorilla.cs.berkeley.edu/leaderboard.html#leaderboard)!

## 🚀 Current Model Collections

### Collection 1: Base Function Calling Models
*Hugging Face Collection: [Arch-Function](https://huggingface.co/collections/katanemo/arch-function-66f209a693ea8df14317ad68)*

| Model Name | Size | Key Features | Downloads |
|------------|------|--------------|-----------|
| **Arch-Function-1.5B** | 1.5B | • Compact size for edge deployment<br>• Efficient function calling<br>• Low resource requirements | [🤗 HuggingFace](https://huggingface.co/katanemo/Arch-Function-1.5B) |
| **Arch-Function-3B** | 3B | • Balanced performance and efficiency<br>• High accuracy function calling<br>• Production-ready | [🤗 HuggingFace](https://huggingface.co/katanemo/Arch-Function-3B) |
| **Arch-Function-7B** | 7B | • Maximum performance<br>• Complex function handling<br>• Enterprise-grade capabilities | [🤗 HuggingFace](https://huggingface.co/katanemo/Arch-Function-7B) |

### Collection 2: Chat-Optimized Models
*Hugging Face Collection: [Arch-Function-Chat](https://huggingface.co/collections/katanemo/arch-function-chat-67e6feb6e33793d82adeded1)*

| Model Name | Size | Key Features | Downloads |
|------------|------|--------------|-----------|
| **Arch-Function-Chat-1.5B** | 1.5B | • Conversational function calling<br>• Interactive agent capabilities<br>• Lightweight deployment | [🤗 HuggingFace](https://huggingface.co/katanemo/Arch-Function-Chat-1.5B) |
| **Arch-Function-Chat-3B** | 3B | • Advanced dialogue management<br>• Context-aware function usage<br>• Multi-turn conversations | [🤗 HuggingFace](https://huggingface.co/katanemo/Arch-Function-Chat-3B) |
| **Arch-Function-Chat-7B** | 7B | • Sophisticated reasoning<br>• Complex multi-step workflows<br>• Premium chat experience | [🤗 HuggingFace](https://huggingface.co/katanemo/Arch-Function-Chat-7B) |


### Collection 3: Agentic Models
*Hugging Face Collection: [Arch-Agent](https://huggingface.co/collections/katanemo/arch-agent-[collection-id])*

| Model Name | Size | Key Features | Downloads |
|------------|------|--------------|-----------|
| **Arch-Agent-1.5B** | 1.5B | • Lightweight autonomous workflows<br>• Edge-optimized performance<br>• Low resource requirements | [🤗 HuggingFace](https://huggingface.co/katanemo/Arch-Agent-1.5B) |
| **Arch-Agent-3B** | 3B | • Balanced autonomous performance<br>• Multi-step task execution<br>• High accuracy workflows | [🤗 HuggingFace](https://huggingface.co/katanemo/Arch-Agent-3B) |
| **Arch-Agent-7B** | 7B | • Advanced autonomous behavior<br>• Complex workflow orchestration<br>• Maximum performance | [🤗 HuggingFace](https://huggingface.co/katanemo/Arch-Agent-7B) |
| **Arch-Agent-32B** | 32B | • Premium autonomous systems<br>• Sophisticated multi-step workflows<br>• Superior capabilities | [🤗 HuggingFace](https://huggingface.co/katanemo/Arch-Agent-32B) |



## 📚 1. Fine-tuning Arch-Function Models

Here we provide a script to fine-tune Arch-Function models with [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory):

### 1.1 Set up environment
- Create the environment following the instructions of [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)
- If you would like to use deepspeed and flash-attn, you can install packages with the following command:
```
pip install deepspeed
pip install flash-attn --no-build-isolation
```

### 1.2 Prepare training data
[LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) supports datasets in `alpaca` and `sharegpt` format. We recommend using the `sharegpt` format for function calling tasks. Below is an example of dataset in:
```json
[
	{
		"conversations": [
			{
				"from": "human",
				"value": "user instruction"
			},
			{
				"from": "function_call",
				"value": "tool arguments"
			},
			{
				"from": "observation",
				"value": "tool result"
			},
			{
				"from": "gpt",
				"value": "model response"
			}
		],
		"system": "system prompt (optional)",
		"tools": "tool description (optional)"
	}
]
```

Next, update `data/dataset_info.json` with the dataset description below:
```json
"dataset_name": {
	"file_name": "data.json",
	"formatting": "sharegpt",
	"columns": {
		"messages": "conversations",
		"system": "system",
		"tools": "tools"
	}
}
```

### 1.3 Training
[LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) provides diverse examples of training for LLMs under `examples`. You can follow these examples and create a training script for your purpose. To kick off training, run the following command:
```bash
CUDA_VISIBLE_DEVICES={YOUR_DEVICE_IDS} llamafactory-cli train {PATH_TO_YOUR_TRAINING_SCRIPT}
```

**Key considerations for fine-tuning:**
- Prepare high-quality function calling examples with proper format
- Use gradient accumulation for larger effective batch sizes
- Monitor validation loss to prevent overfitting
- Consider using LoRA for parameter-efficient fine-tuning

## 📚 2. Inference with Arch-Function Models

To run inference with Arch-Function models for function calling tasks, follow the steps below:

### 2.1 Set up environment
Arch-Function models have been in the Hugging Face [transformers library](https://github.com/huggingface/transformers) and we advise you to install latest version with the following command:
```bash
pip install transformers>=4.51.0
```

### 2.2 Inference

Below is a script demonstrating how to use Arch-Function models for function calling tasks.

#### 2.2.1 Create models and tokenizers
You can specify the desired model name and create models and corresponding tokenizers with the following script:
```python
import json
from typing import Any, Dict, List
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "katanemo/Arch-Function-Chat-1.5B" // Specify the desired model name here

model = AutoModelForCausalLM.from_pretrained(
    model_name, device_map="auto", torch_dtype="auto", trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
```

#### 2.2.2 Format prompts
Our models perform best when using the recommended prompt format, which can be found in the corresponding model cards on Hugging Face. You can run the following script to format prompts:

```python
# Please use the recommended prompt for each model.
TASK_PROMPT = (
    "You are a helpful assistant designed to assist with the user query by making one or more function calls if needed."
    "\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>\n{tools}\n</tools>"
    "\n\nYour task is to decide which functions are needed and collect missing parameters if necessary."
)

FORMAT_PROMPT = (
    "\n\nBased on your analysis, provide your response in one of the following JSON formats:"
    '\n1. If no functions are needed:\n```json\n{"response": "Your response text here"}\n```'
    '\n2. If functions are needed but some required parameters are missing:\n```json\n{"required_functions": ["func_name1", "func_name2", ...], "clarification": "Text asking for missing parameters"}\n```'
    '\n3. If functions are needed and all required parameters are available:\n```json\n{"tool_calls": [{"name": "func_name1", "arguments": {"argument1": "value1", "argument2": "value2"}},... (more tool calls as required)]}\n```'
)

# Define available tools
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "str",
                        "description": "The city and state, e.g. San Francisco, New York",
                    },
                    "unit": {
                        "type": "str",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "The unit of temperature to return",
                    },
                },
                "required": ["location"],
            },
        },
    }
]


# Helper function to create the system prompt for our model
def format_prompt(tools: List[Dict[str, Any]]):
    tools = "\n".join(
        [json.dumps(tool["function"], ensure_ascii=False) for tool in tools]
    )
    return TASK_PROMPT.format(tools=tools) + FORMAT_PROMPT


system_prompt = format_prompt(tools)

messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": "What is the weather in Seattle?"},
]
```

#### 2.2.3 Run inference
Now, you can run the following script to do inference with Arch-Function models.
```python
model_inputs = tokenizer.apply_chat_template(
    messages, add_generation_prompt=True, return_tensors="pt"
).to(model.device)

generated_ids = model.generate(**model_inputs, max_new_tokens=32768)

generated_ids = [
    output_ids[len(input_ids) :]
    for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(response)
```

**Inference optimization tips:**
- Use appropriate temperature settings (0.0 - 0.1 for function calling)
- User proper prompt formatting for best results
- Consider batching for multiple requests
- Use quantized models for faster inference


## 📚 3. Deployment with Popular Hosting Frameworks

Below we show how to deploy Arch-Function models using popular model hosting frameworks.

### 3.1 vLLM Deployment

[vLLM](https://github.com/vllm-project/vllm) provides high-throughput serving with advanced optimizations. Following the steps below to deploy Arch-Function models with vLLM

#### 3.1.1 Set up environment
```bash
# Install vLLM
pip install vllm
```

#### 3.1.2 Start vLLM server
```bash
vllm serve \
    --model katanemo/Arch-Function-3B \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 1
```

#### 3.1.3 Get responses
To get responses from the vLLM server for function calling, first format prompts following [here](https://github.com/katanemo/Arch-Function?tab=readme-ov-file#222-format-prompts). Then, replace `messages` in the script below with the formatted prompts and run the script.
```python
# Client code for vLLM
from openai import OpenAI

# Point to the local server
client = OpenAI(
    api_key="EMPTY",
    base_url="http://localhost:8000/v1",
)

# 
completion = client.chat.completions.create(
    model="katanemo/Arch-Function-3B",
    messages=[
        {"role": "user", "content": "Get the current temperature in San Francisco"}
    ],
    temperature=0.01,
    max_tokens=512
)

print(completion.choices[0].message.content)
```

### 3.2 ollama Deployment

[ollama](https://ollama.ai) provides easy local deployment with automatic model management. Below we provide scripts to show how to use ollama for deployment.


#### 3.2.1 Install ollama (see [ollama](https://ollama.ai) for installation)

#### 3.2.2 Start ollama server
Specify your desired model name below and run the follwoing command to start the ollama server:
```bash
ollama run hf.co/katanemo/{MODEL_NAME}
```

#### 3.2.3 Get responses
Format prompts following [here](https://github.com/katanemo/Arch-Function?tab=readme-ov-file#222-format-prompts), and the replace `formatted_prompt` in the script below and run the script to get responses.

```python
# Client code for Ollama
import requests
import json

def call_ollama(formatted_prompt):
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "arch-function", 
            "prompt": formatted_prompt, 
            "stream": False,
        },
    )
    return json.loads(response.text)["response"]

# # Replace with formatted prompts
formatted_prompt = "Calculate the area of a circle with radius 5"

result = call_ollama("Calculate the area of a circle with radius 5")
print(result)
```


### 3.3 SGLang Deployment

[SGLang](https://github.com/sgl-project/sglang) offers structured generation capabilities with high performance. To use SGLang for deployment, follow the steps below.

#### 3.3.1 Set up experiment
```bash
# Install SGLang
pip install sglang[all]
```

#### 3.3.2 Start SGLang server
```bash
python -m sglang.launch_server \
    --model-path katanemo/Arch-Function-3B \
    --host 0.0.0.0 \
    --port 30000 \
    --tp 1 \
    --trust-remote-code
```

#### 3.3.3 Get responses
As sglang provides OpenAI-compatible APIs, you can follow the same way as vLLM to get responses from the server. First format prompts following [here](https://github.com/katanemo/Arch-Function?tab=readme-ov-file#222-format-prompts). Then, replace `messages` in the script below with the formatted prompts and run the script.
```python
# Client code for vLLM
from openai import OpenAI

# Point to the local server
client = OpenAI(
    api_key="EMPTY",
    base_url="http://localhost:8000/v1",
)

# 
completion = client.chat.completions.create(
    model="katanemo/Arch-Function-3B",
    messages=[
        {"role": "user", "content": "Get the current temperature in San Francisco"}
    ],
    temperature=0.01,
    max_tokens=512
)

print(completion.choices[0].message.content)
```


## 🔬 Research & Development

The Arch-Function project is actively developing next-generation models that will:
- Further advance function calling accuracy beyond current SOTA
- Introduce novel architectures optimized for tool usage
- Expand to multimodal function calling capabilities
- Support more complex reasoning patterns in function selection


## 📄 License

Please refer to the individual model pages on Hugging Face for specific licensing information.


## 🤝 Contributing

We welcome contributions to improve the Arch-Function tutorials and documentation! You can help by:

- Fixing errors or improving existing tutorials
- Adding new deployment examples or use cases
- Suggesting additional framework integrations
- Improving documentation clarity

Feel free to open an issue or submit a pull request with your improvements.


## 📞 Support

For questions and support:
- Open an issue in this repository
- Visit our [Hugging Face Hub](https://huggingface.co/katanemo)
- Check the [Katanemo organization](https://github.com/katanemo) on Github

