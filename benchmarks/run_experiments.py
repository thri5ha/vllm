import os
import time
import json
import subprocess
from datetime import datetime
import json,re
import matplotlib.pyplot as plt
import pandas as pd

SMALL_MODELS = ["facebook/opt-125m", "facebook/opt-1.3b", "facebook/opt-2.7b"]
LARGE_MODELS = ["meta-llama/Llama-2-7b-hf", "meta-llama/Meta-Llama-3-8B", "meta-llama/Meta-Llama-3-8B-Instruct", 
                "mistralai/Mistral-7B-v0.1", "mistralai/Mistral-7B-Instruct-v0.3"]

MODELS = SMALL_MODELS + LARGE_MODELS

NUM_PROMPTS = 100
NUM_GPUS = [1, 2]
BATCH_SIZES = [4, 8, 16, 24, 32, 36, 40, 64]
MAX_OUTPUT_LEN = 100
MAX_MODEL_LEN = 1300

gpu_type = "Nvidia GeForce RTX 2080"

GRAPH_OUTPUT_PATH_1="/home/mcw/vllm_results/benchmark_plots1"
GRAPH_OUTPUT_PATH_2="/home/mcw/vllm_results/benchmark_plots2"
CSV_OUTPUT_PATH="/home/mcw/vllm_results/benchmark_all_csv"
OUTPUT_PATH = "/home/mcw/common/results_07_14"
# DATASET_PATH = "/home/mcw/common/learning/testing_data/prompts.json"
DATASET_PATH = "/home/mcw/thrisha/data/ShareGPT_V3_unfiltered_cleaned_split.json"

SCRIPT_PATH = "/home/mcw/vllm/benchmarks/benchmark_throughput.py"

TIMESTAMP = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
OUTPUT_PATH = os.path.join(OUTPUT_PATH, TIMESTAMP)
GRAPH_OUTPUT_PATH = os.path.join(GRAPH_OUTPUT_PATH_1, TIMESTAMP)
GRAPH_OUTPUT_PATH = os.path.join(GRAPH_OUTPUT_PATH_2, TIMESTAMP)
CSV_OUTPUT_PATH= os.path.join(CSV_OUTPUT_PATH, TIMESTAMP)
os.makedirs(OUTPUT_PATH, exist_ok=True)
os.makedirs(GRAPH_OUTPUT_PATH_1, exist_ok=True)
os.makedirs(GRAPH_OUTPUT_PATH_2, exist_ok=True)
os.makedirs(CSV_OUTPUT_PATH, exist_ok=True)

def run_subprocess_realtime(cmd: list) -> int:
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)

    while True:
        output = process.stdout.readline()
        if output == '' and process.poll() is not None:
            break
        if output:
            print(output.strip())

    return process.returncode
# Function to plot latency and throughput vs batch size for each model


def plot_model_data(model_name, df, graph_output_path):
    model_df = df[df['model_base_name'] == model_name]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    for num_gpus in [1, 2]:
        gpu_df = model_df[model_df['num_gpus'] == num_gpus]
        gpu_df_sorted = gpu_df.sort_values(by='batch_size')
        ax1.plot(gpu_df_sorted['batch_size'], gpu_df_sorted['elapsed_time'], marker='o', label=f'{model_name} - {num_gpus} GPU(s) - {gpu_type}')
        ax2.plot(gpu_df_sorted['batch_size'], gpu_df_sorted['tokens_per_second'], marker='o', label=f'{model_name} - {num_gpus} GPU(s) - {gpu_type}')

        # Annotate data points with exact numbers for Latency
        for i in range(len(gpu_df_sorted)):
            ax1.annotate(f'({gpu_df_sorted.iloc[i]["batch_size"]},{gpu_df_sorted.iloc[i]["elapsed_time"]:.2f})', 
                         (gpu_df_sorted.iloc[i]['batch_size'], gpu_df_sorted.iloc[i]['elapsed_time']), textcoords="offset points", xytext=(0,10), ha='center')
            ax2.annotate(f'({gpu_df_sorted.iloc[i]["batch_size"]},{gpu_df_sorted.iloc[i]["tokens_per_second"]:.2f})', 
                         (gpu_df_sorted.iloc[i]['batch_size'], gpu_df_sorted.iloc[i]['tokens_per_second']), textcoords="offset points", xytext=(0,10), ha='center')
    
    ax1.set_xlabel('Batch Size')
    ax1.set_ylabel('Latency (sec)')
    ax1.set_title(f'Latency vs Batch Size for {model_name}')
    ax1.legend()
    ax1.grid(True)

    ax2.set_xlabel('Batch Size')
    ax2.set_ylabel('Throughput (Tokens per Second)')
    ax2.set_title(f'Throughput vs Batch Size for {model_name}')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(graph_output_path, f'{model_name}_latency_throughput.png'))
    plt.close()

# Function to plot Latency vs Batch Size for all models for a given number of GPUs
def plot_latency_vs_batch_size(df, num_gpus, graph_output_path):
    plt.figure(figsize=(12, 8))
    for model_name in df['model_base_name'].unique():
        df_model = df[(df['model_base_name'] == model_name) & (df['num_gpus'] == num_gpus)].sort_values('batch_size')
        batch_sizes = df_model['batch_size'].tolist()
        latency = df_model['elapsed_time'].tolist()
        plt.plot(batch_sizes, latency, 'o-', label=model_name)
        for i, txt in enumerate(latency):
            plt.annotate(f'({batch_sizes[i]},{txt:.2f})', (batch_sizes[i], latency[i]), textcoords="offset points", xytext=(0, 10), ha='center')
    plt.xlabel('Batch Size')
    plt.ylabel('Latency (sec)')
    plt.title(f'Latency vs Batch Size for {num_gpus} GPU(s)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(graph_output_path, f'latency_vs_batch_size_{num_gpus}_gpus.png'))
    plt.close()

# Function to plot Throughput vs Batch Size for all models for a given number of GPUs
def plot_throughput_vs_batch_size(df, num_gpus, graph_output_path):
    plt.figure(figsize=(12, 8))
    for model_name in df['model_base_name'].unique():
        df_model = df[(df['model_base_name'] == model_name) & (df['num_gpus'] == num_gpus)].sort_values('batch_size')
        batch_sizes = df_model['batch_size'].tolist()
        throughput = df_model['tokens_per_second'].tolist()
        plt.plot(batch_sizes, throughput, 'o-', label=model_name)
        for i, txt in enumerate(throughput):
            plt.annotate(f'({batch_sizes[i]},{txt:.2f})', (batch_sizes[i], throughput[i]), textcoords="offset points", xytext=(0, 10), ha='center')
    plt.xlabel('Batch Size')
    plt.ylabel('Throughput (Tokens per Second)')
    plt.title(f'Throughput vs Batch Size for {num_gpus} GPU(s)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(graph_output_path, f'throughput_vs_batch_size_{num_gpus}_gpus.png'))
    plt.close()
        
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
        # Convert JSON data to DataFrame
        df = pd.DataFrame(combined_data)

        df['batch_size'] = df['model_name'].apply(lambda x: int(re.search(r'_bs(\d+)_', x).group(1)))
        df['model_base_name'] = df['model_name'].apply(lambda x: re.search(r'^(.*)_gpu', x).group(1))

        df.to_csv('output.tsv', sep='\t', index=False)
        # Generate and save plots for each unique model
        for model in df['model_base_name'].unique():
            plot_model_data(model, df,GRAPH_OUTPUT_PATH_1)
        
        for i in NUM_GPUS:
        # Plot Latency vs Batch Size for 1 GPU and 2 GPUs
            plot_latency_vs_batch_size(df, i, GRAPH_OUTPUT_PATH_2)
            plot_throughput_vs_batch_size(df, i, GRAPH_OUTPUT_PATH_2)

    except Exception as e:
            print(f"failed with exception : {e}")
    

def main():
    for model in MODELS:
        for tp in NUM_GPUS:
            if tp == 1 and model in LARGE_MODELS:
                print("Model cannot fit into a single GPU..")
                continue
            
            for batch_size in BATCH_SIZES:
                max_output_tokens = MAX_OUTPUT_LEN
                model_string = model.replace('/', '_').replace('.', '_').replace('-', '_')
                experiment_name = f"{model_string}_gpu{tp}_bs{batch_size}_o{max_output_tokens}"
                output_json = os.path.join(OUTPUT_PATH, f"{experiment_name}.json")
                
                try:
                    command = ["python3", SCRIPT_PATH, 
                              f"--dataset", DATASET_PATH, 
                              f"--num-prompts", f"{NUM_PROMPTS}",
                              f"--model", model,
                              f"--max-num-seqs", f"{batch_size}", 
                              f"--tensor-parallel-size", f"{tp}",
                              f"--output-json", output_json,
                              f"--output-len", f"{max_output_tokens}",
                              f"--max-model-len", f"{MAX_MODEL_LEN}"] 
                    
                    print(f"Executing command {' '.join(command)}")
                    run_subprocess_realtime(command)
                
                    if os.path.exists(output_json):
                        with open(output_json, 'r') as f:
                            results = json.load(f)
                        
                        results['model_name'] = experiment_name
                        results['num_gpus'] = tp
                        results['max_output_len'] = max_output_tokens
                        
                        with open(output_json, 'w') as f:
                            json.dump(results, f, indent=4)

                except Exception as e:
                    print(f"FAILED [[{experiment_name}]] : {e}")
                
                print(f"[{experiment_name}] Done.")
                
    print("Done with inferencing...Creating Graphs..")

    # generate_graphs(OUTPUT_PATH)

                        
if __name__ == "__main__":
    main()