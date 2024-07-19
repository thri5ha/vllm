import time
import os,json,re
import subprocess
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
from vllm import LLM, SamplingParams
from vllm.utils import FlexibleArgumentParser
import os


SCRIPT_PATH="/home/mcw/vllm/benchmarks/benchmark_prefix_caching.py"
GRAPH_OUTPUT_PATH="/home/mcw/vllm_results/plots_prefix_cache"
OUTPUT_PATH="/home/mcw/vllm_results/json_prefix_caching"
TIMESTAMP = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
OUTPUT_PATH=os.path.join(OUTPUT_PATH,TIMESTAMP)
GRAPH_OUTPUT_PATH=os.path.join(GRAPH_OUTPUT_PATH,TIMESTAMP)
os.makedirs(OUTPUT_PATH, exist_ok=True)
os.makedirs(GRAPH_OUTPUT_PATH, exist_ok=True)

# "facebook/opt-125m"
SMALL_MODELS = ["facebook/opt-2.7b"]
LARGE_MODELS = ["meta-llama/Llama-2-7b-hf","meta-llama/Meta-Llama-3-8B"]

MAX_MODEL_LEN = 2048
BATCH_SIZE=24

models=SMALL_MODELS+LARGE_MODELS

output_lengths=[200,400,600,800,1000,1200,1400]
NUM_GPUS=[1,2]

def run_subprocess_realtime(cmd: list) -> int:
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)

    while True:
        output = process.stdout.readline()
        if output == '' and process.poll() is not None:
            break
        if output:
            print(output.strip())

    return process.returncode

def generate_plot_modelwise(data):

    # Collect unique models and their GPU counts
    models = {entry['model']: entry['num_gpus'] for entry in data}

    # Plotting for each model
    for model, num_gpus in models.items():
        model_data_true = sorted([entry for entry in data if entry['model'] == model and entry['prefix_cache']], key=lambda x: x['output_len'])
        model_data_false = sorted([entry for entry in data if entry['model'] == model and not entry['prefix_cache']], key=lambda x: x['output_len'])
        
        plt.figure(figsize=(10, 6))
        
        # Plotting lines with markers and labels
        plt.plot([entry['output_len'] for entry in model_data_true],
                [entry['elapsed_time'] for entry in model_data_true],
                label='Prefix Cache True',
                marker='o',
                linestyle='-')
        
        plt.plot([entry['output_len'] for entry in model_data_false],
                [entry['elapsed_time'] for entry in model_data_false],
                label='Prefix Cache False',
                marker='o',
                linestyle='-')
        
        # Annotating each point with x and y values
        for entry in model_data_true:
            plt.annotate(f"({entry['output_len']}, {entry['elapsed_time']:.2f})",
                        (entry['output_len'], entry['elapsed_time']),
                        textcoords="offset points",
                        xytext=(0, 5),
                        ha='center')
        
        for entry in model_data_false:
            plt.annotate(f"({entry['output_len']}, {entry['elapsed_time']:.2f})",
                        (entry['output_len'], entry['elapsed_time']),
                        textcoords="offset points",
                        xytext=(0, 5),
                        ha='center')
        
        plt.xlabel('Output Length')
        plt.ylabel('Latency (seconds)')
        plt.title(f'Latency vs Output Length for {model} (GPUs used: {num_gpus})')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        sanitized_model_name = model.replace("/", "_")
        plt.savefig(os.path.join(GRAPH_OUTPUT_PATH, f'{sanitized_model_name}_latency_vs_output_len.png'))
        plt.show()

def generate_single_plot(data, output_length):
    # Filter data by the given output length
    filtered_data = [entry for entry in data if entry['output_len'] == output_length]
    
    # Collect unique models
    models = set(entry['model'] for entry in filtered_data)
    
    # Initialize lists to hold latencies for prefix cache true and false
    latencies_true = []
    latencies_false = []
    model_names = []

    for model in models:
        model_data_true = next((entry for entry in filtered_data if entry['model'] == model and entry['prefix_cache']), None)
        model_data_false = next((entry for entry in filtered_data if entry['model'] == model and not entry['prefix_cache']), None)
        
        if model_data_true and model_data_false:
            latencies_true.append(model_data_true['elapsed_time'])
            latencies_false.append(model_data_false['elapsed_time'])
            model_names.append(model)
    
    # Plotting the bar graph
    x = range(len(model_names))  # the label locations
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bar_width = 0.2
    
    bars1 = ax.bar(x, latencies_true, bar_width, label='Prefix Cache True')
    bars2 = ax.bar([p + bar_width for p in x], latencies_false, bar_width, label='Prefix Cache False')
    
    ax.set_xlabel('Model')
    ax.set_ylabel('Latency (seconds)')
    ax.set_title(f'Latency by Model for Output Length {output_length}')
    ax.set_xticks([p + bar_width / 2 for p in x])
    ax.set_xticklabels(model_names)
    ax.legend()
    
    # Adding the text labels on the bars
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(os.path.join(GRAPH_OUTPUT_PATH, f'single_plot.png'))
    # plt.show()



def generate_graphs(output_folder_path):
    try:
        combined_data = []

        # List all files in the folder
        for filename in os.listdir(output_folder_path):
            if filename.endswith('.json'):
                file_path = os.path.join(output_folder_path, filename)
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    combined_data.append(data)
        generate_plot_modelwise(combined_data)
        output_length=300 # define output length for single plot
        generate_single_plot(combined_data,output_length)

    except Exception as e:
            print(f"failed with exception : {e}")
    

def main():
    for model in models:
        if model in LARGE_MODELS:
            tp=2
        else:
            tp=1
        for bool_prefix_cache in [True,False]:
            for output_len in output_lengths:
                max_output_tokens = output_len
                model_string = model.replace('/', '_').replace('.', '_').replace('-', '_')
                experiment_name = f"{model_string}_gpu{tp}_o{max_output_tokens}_pc{bool_prefix_cache}"
                output_json = os.path.join(OUTPUT_PATH, f"{experiment_name}.json")
                
                try:
                    command = ["python3", SCRIPT_PATH,  
                                f"--model", model,
                                f"--max-num-seqs", f"{BATCH_SIZE}", 
                                f"--tensor-parallel-size", f"{tp}",
                                f"--output-json", output_json,
                                f"--output-len", f"{max_output_tokens}",
                                f"--max-model-len", f"{MAX_MODEL_LEN}"] 
                    if bool_prefix_cache:
                        command+=["--enable-prefix-caching"] 
                        
                    print(f"Executing command {' '.join(command)}")
                    run_subprocess_realtime(command)


                except Exception as e:
                        print(f"FAILED [[{experiment_name}]] : {e}")
                    
                print(f"[{experiment_name}] Done.")
        
    generate_graphs(OUTPUT_PATH)
                       
if __name__ == "__main__":
    main()