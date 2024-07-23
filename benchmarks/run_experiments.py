import os
import time
import json
import subprocess
from datetime import datetime
import json,re
import matplotlib.pyplot as plt
import pandas as pd

# SMALL_MODELS = ["facebook/opt-125m", "facebook/opt-1.3b", "facebook/opt-2.7b"]
# LARGE_MODELS = ["meta-llama/Llama-2-7b-hf","meta-llama/Meta-Llama-3-8B-Instruct", "mistralai/Mistral-7B-v0.1", "mistralai/Mistral-7B-Instruct-v0.3"]

SMALL_MODELS = ["facebook/opt-125m","facebook/opt-2.7b"]
LARGE_MODELS = ["meta-llama/Llama-2-7b-hf","meta-llama/Meta-Llama-3-8B"]
MODELS = SMALL_MODELS  + LARGE_MODELS

NUM_PROMPTS = 200
NUM_GPUS = [2, 1]
BATCH_SIZES = [2,8,16,32,48,64]
MAX_OUTPUT_LEN = 200
MAX_MODEL_LEN = 2048

gpu_type = "Nvidia GeForce RTX 2080"

GRAPH_OUTPUT_PATH_1="/home/mcw/vllm_results/benchmark_plots1"
GRAPH_OUTPUT_PATH_2="/home/mcw/vllm_results/benchmark_plots2"
CSV_OUTPUT_PATH="/home/mcw/vllm_results/benchmark_throughput_csv"
OUTPUT_PATH = "/home/mcw/vllm_results/json_benchmark_throughput"

# DATASET_PATH = "/home/mcw/common/learning/testing_data/prompts.json"
DATASET_PATH = "/home/mcw/thrisha/data/ShareGPT_V3_unfiltered_cleaned_split.json"

SCRIPT_PATH = "/home/mcw/vllm/benchmarks/benchmark_throughput.py"

TIMESTAMP = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
OUTPUT_PATH = os.path.join(OUTPUT_PATH, TIMESTAMP)
GRAPH_OUTPUT_PATH_1 = os.path.join(GRAPH_OUTPUT_PATH_1, TIMESTAMP)
GRAPH_OUTPUT_PATH_2 = os.path.join(GRAPH_OUTPUT_PATH_2, TIMESTAMP)
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

def plot_model_data(df, graph_output_path,parallelism,bool_prefix_cache):

    df = df[(df['parallelism'] == parallelism) & (df['enable_chunked_prefill'] == bool_prefix_cache)]

    for model_name in df['model'].unique():
        model_df = df[df['model'] == model_name]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

        for num_gpus in [1, 2]:
            gpu_df = model_df[model_df['num_gpus'] == num_gpus]
            if gpu_df.empty:
                continue
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
        ax1.set_title(f'Latency vs Batch Size for {model_name} parallelism: {parallelism}')
        ax1.legend()
        ax1.grid(True)

        ax2.set_xlabel('Batch Size')
        ax2.set_ylabel('Throughput (Tokens per Second)')
        ax2.set_title(f'Throughput vs Batch Size for {model_name} parallelism: {parallelism}')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        sanitized_model_name = model_name.replace("/", "_")
        plt.savefig(os.path.join(graph_output_path, f'{sanitized_model_name}_latency_throughput_{parallelism}_prefix_cache{bool_prefix_cache}.png'))
        plt.close()


# Function to plot Latency vs Batch Size for all models for a given number of GPUs
def plot_latency_vs_batch_size(df,graph_output_path,parallelism,bool_prefix_cache):
    df = df[(df['parallelism'] == parallelism) & (df['enable_chunked_prefill'] == bool_prefix_cache)]
    for num_gpus in [1,2]:
        plt.figure(figsize=(12, 8))
        for model_name in df['model'].unique():
            df_model = df[(df['model'] == model_name) & (df['num_gpus'] == num_gpus)].sort_values('batch_size')
            if df_model.empty:
                continue
            batch_sizes = df_model['batch_size'].tolist()
            latency = df_model['elapsed_time'].tolist()
            plt.plot(batch_sizes, latency, 'o-', label=model_name)
            for i, txt in enumerate(latency):
                plt.annotate(f'({batch_sizes[i]},{txt:.2f})', (batch_sizes[i], latency[i]), textcoords="offset points", xytext=(0, 10), ha='center')
        plt.xlabel('Batch Size')
        plt.ylabel('Latency (sec)')
        plt.title(f'Latency vs Batch Size for {num_gpus} GPU(s)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(graph_output_path, f'latency_vs_batch_size_{num_gpus}_gpus.png'))
        plt.close()

# Function to plot Throughput vs Batch Size for all models for a given number of GPUs
def plot_throughput_vs_batch_size(df,graph_output_path,parallelism,bool_prefix_cache):
    df = df[(df['parallelism'] == parallelism) & (df['enable_chunked_prefill'] == bool_prefix_cache)]
    for num_gpus in [1,2]:
        plt.figure(figsize=(12, 8))
        for model_name in df['model'].unique():
            df_model = df[(df['model'] == model_name) & (df['num_gpus'] == num_gpus)].sort_values('batch_size')
            if df_model.empty:
                continue
            batch_sizes = df_model['batch_size'].tolist()
            throughput = df_model['tokens_per_second'].tolist()
            plt.plot(batch_sizes, throughput, 'o-', label=model_name)
            for i, txt in enumerate(throughput):
                plt.annotate(f'({batch_sizes[i]},{txt:.2f})', (batch_sizes[i], throughput[i]), textcoords="offset points", xytext=(0, 10), ha='center')
        plt.xlabel('Batch Size')
        plt.ylabel('Throughput (Tokens per Second)')
        plt.title(f'Throughput vs Batch Size for {num_gpus} GPU(s)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(graph_output_path, f'throughput_vs_batch_size_{num_gpus}_gpus_parallelism{parallelism}_prefix_cache{bool_prefix_cache}.png'))
        plt.close()


def plot_latency_vs_batch_size_prefill(df,graph_output_path):
    # Filter the data based on the number of GPUs
    for num_gpus in [1, 2]:

        df_filtered = df[df['num_gpus'] == num_gpus]
        
        # Get the unique models
        unique_models = df_filtered['model'].unique()
        
        for model in unique_models:
            # Filter data for the current model
            df_model = df_filtered[df_filtered['model'] == model]
            
            # Separate data for chunked prefill true and false, then sort by batch_size
            df_chunked_true = df_model[df_model['enable_chunked_prefill'] == True].sort_values(by='batch_size')
            df_chunked_false = df_model[df_model['enable_chunked_prefill'] == False].sort_values(by='batch_size')
            
            plt.figure(figsize=(10, 6))
            
            # Plot data with chunked prefill true
            plt.plot(df_chunked_true['batch_size'], df_chunked_true['elapsed_time'], marker='o', linestyle='-', label='Chunked Prefill True')
            
            # Annotate points for chunked prefill true
            for i, row in df_chunked_true.iterrows():
                plt.annotate(f'({row["batch_size"]:.2f}, {row["elapsed_time"]:.2f})', 
                             (row['batch_size'], row['elapsed_time']),
                             textcoords="offset points", xytext=(5,5), ha='center')
            
            # Plot data with chunked prefill false
            plt.plot(df_chunked_false['batch_size'], df_chunked_false['elapsed_time'], marker='o', linestyle='-', label='Chunked Prefill False')
            
            # Annotate points for chunked prefill false
            for i, row in df_chunked_false.iterrows():
                plt.annotate(f'({row["batch_size"]:.2f}, {row["elapsed_time"]:.2f})', 
                             (row['batch_size'], row['elapsed_time']),
                             textcoords="offset points", xytext=(5, 5), ha='center')
            
            plt.title(f'Latency vs Batch Size for Model {model} with {num_gpus} GPU(s)')
            plt.xlabel('Batch Size')
            plt.ylabel('Elapsed Time (Latency)')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(graph_output_path, f'Latency vs Batch Size for Model {model} with {num_gpus} GPU(s)'))
            plt.close()
            plt.show()
            
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
        df = df.sort_values(by=['model', 'num_gpus', 'batch_size'])

        output_path = os.path.join(CSV_OUTPUT_PATH, 'output.csv')
        
        df.to_csv(output_path, sep='\t', index=False)
        # Generate and save plots for each unique model

        # throughput vs batch size for tensor parallelism and pipeline parallelism for each model
        plot_model_data(df,GRAPH_OUTPUT_PATH_1,parallelism="tensor",bool_prefix_cache=True)
        # plot_model_data(df,GRAPH_OUTPUT_PATH,parallelism="pipeline",bool_prefix_cache=False)

        # Plot Latency vs Batch Size for 1 GPU and 2 GPUs
        plot_latency_vs_batch_size(df, GRAPH_OUTPUT_PATH_2,parallelism="tensor",bool_prefix_cache=False)
        plot_throughput_vs_batch_size(df, GRAPH_OUTPUT_PATH_2,parallelism="tensor",bool_prefix_cache=False)

        # plot comparing chunked prefill true vs false
        plot_latency_vs_batch_size_prefill(df,GRAPH_OUTPUT_PATH_2)



    except Exception as e:
            print(f"failed with exception : {e}")
    
# Tensor parallel size & Chunked prefill
def experiment(tensor_parallel = False, pipeline_parallel = False):
    
    if any([n_gpu > 1 for n_gpu in NUM_GPUS]) and not (tensor_parallel or pipeline_parallel):
        raise Exception("Either tensor parallel or pipeline parallel must be true when NUM GPUS > 1")
    if tensor_parallel and pipeline_parallel:
        raise Exception("Either choose tensor parallel or pipeline parallel.")
    
    for cp in [True,False]:
        for n_gpus in NUM_GPUS:
            for model in MODELS:
                if n_gpus == 1 and model in LARGE_MODELS:
                    print("Model cannot fit into a single GPU..")
                    continue
                if cp==True and n_gpus==2 and model in SMALL_MODELS:
                    print("No need to calculate for chunked prefill=True when tp=2 for small models")
                    continue
                
                for batch_size in BATCH_SIZES:
                    max_output_tokens = MAX_OUTPUT_LEN
                    model_string = model.replace('/', '_').replace('.', '_').replace('-', '_')
                    experiment_name = f"{model_string}_gpu{n_gpus}_bs{batch_size}_o{max_output_tokens}_{'tp' if tensor_parallel else 'pp'}_cp{cp}"
                    output_json = os.path.join(OUTPUT_PATH, f"{experiment_name}.json")
                    
                    try:
                        command = ["python3", SCRIPT_PATH, 
                                f"--dataset", DATASET_PATH, 
                                f"--num-prompts", f"{NUM_PROMPTS}",
                                f"--model", model,
                                f"--max-num-seqs", f"{batch_size}", 
                                f"--output-json", output_json,
                                f"--output-len", f"{max_output_tokens}",
                                f"--max-model-len", f"{MAX_MODEL_LEN}"]
                    
                        if tensor_parallel:
                            command+=[f"--tensor-parallel-size", f"{n_gpus}"]
                        elif pipeline_parallel:
                            command+=[f"--pipeline-parallel-size", f"{n_gpus}"]
                        
                        if cp:
                            command+=[f"--enable-chunked-prefill"]
                        
                        print(f"Executing command {' '.join(command)}")
                        run_subprocess_realtime(command)
                    
                        if os.path.exists(output_json):
                            with open(output_json, 'r') as f:
                                results = json.load(f)
                            
                            results['model']=model
                            results['model_name'] = experiment_name
                            results['num_gpus'] = n_gpus
                            results['max_output_len'] = max_output_tokens
                            results['batch_size']=batch_size
                            results['parallelism']='tensor' if tensor_parallel else 'pipeline'
                            results['enable_chunked_prefill']=cp
                            
                            with open(output_json, 'w') as f:
                                json.dump(results, f, indent=4)

                    except Exception as e:
                        print(f"FAILED [[{experiment_name}]] : {e}")
                    
                    print(f"[{experiment_name}] Done.")
                    
    print("Done with inferencing....")


def main():

    experiment(tensor_parallel=True)
    # experiment(tensor_parallel=False,pipeline_parallel=True) # PIPELINE parallel is only supported in online, when serving the model.
                    
if __name__ == "__main__":
    main()