from vllm import LLM, SamplingParams
from huggingface_hub import login

login("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")
# Sample prompts.
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=100)

# Create an LLM.
llm = LLM(model="mistralai/Mistral-7B-v0.3", tensor_parallel_size=2, dtype='float16', gpu_memory_utilization=0.97, max_num_seqs=5, max_model_len=1000)
# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
outputs = llm.generate(prompts, sampling_params)
# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"{len(output.outputs[0].token_ids)}")
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
