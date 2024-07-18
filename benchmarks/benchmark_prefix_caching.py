import time
import os,json,re
import subprocess
from datetime import datetime
from vllm import LLM, SamplingParams
from vllm.utils import FlexibleArgumentParser
import os

PROMPT = "You are a helpful assistant in recognizes the content of tables in markdown format. Here is a table as fellows. You need to answer my question about the table.\n# Table\n|Opening|Opening|Sl. No.|Film|Cast|Director|Music Director|Notes|\n|----|----|----|----|----|----|----|----|\n|J A N|9|1|Agni Pushpam|Jayabharathi, Kamalahasan|Jeassy|M. K. Arjunan||\n|J A N|16|2|Priyamvada|Mohan Sharma, Lakshmi, KPAC Lalitha|K. S. Sethumadhavan|V. Dakshinamoorthy||\n|J A N|23|3|Yakshagaanam|Madhu, Sheela|Sheela|M. S. Viswanathan||\n|J A N|30|4|Paalkkadal|Sheela, Sharada|T. K. Prasad|A. T. Ummer||\n|F E B|5|5|Amma|Madhu, Srividya|M. Krishnan Nair|M. K. Arjunan||\n|F E B|13|6|Appooppan|Thikkurissi Sukumaran Nair, Kamal Haasan|P. Bhaskaran|M. S. Baburaj||\n|F E B|20|7|Srishti|Chowalloor Krishnankutty, Ravi Alummoodu|K. T. Muhammad|M. S. Baburaj||\n|F E B|20|8|Vanadevatha|Prem Nazir, Madhubala|Yusufali Kechery|G. Devarajan||\n|F E B|27|9|Samasya|Madhu, Kamalahaasan|K. Thankappan|Shyam||\n|F E B|27|10|Yudhabhoomi|K. P. Ummer, Vidhubala|Crossbelt Mani|R. K. Shekhar||\n|M A R|5|11|Seemantha Puthran|Prem Nazir, Jayabharathi|A. B. Raj|M. K. Arjunan||\n|M A R|12|12|Swapnadanam|Rani Chandra, Dr. Mohandas|K. G. George|Bhaskar Chandavarkar||\n|M A R|19|13|Thulavarsham|Prem Nazir, sreedevi, Sudheer|N. Sankaran Nair|V. Dakshinamoorthy||\n|M A R|20|14|Aruthu|Kaviyoor Ponnamma, Kamalahasan|Ravi|G. Devarajan||\n|M A R|26|15|Swimming Pool|Kamal Haasan, M. G. Soman|J. Sasikumar|M. K. Arjunan||\n\n# Question\nWhat' s the content in the (1,1) cells\n"  # noqa: E501


def test_prefix(llm=None, sampling_params=None, prompts=None):

    NUM_ITERS = 1 # will change it later at time of benchmarking
    WARM_UP = 0
    # print("Warm up")
    for i in range(WARM_UP):
        llm.generate(prompts, sampling_params, use_tqdm=True)
    print("Warm up done.")

    elapsed_time = 0
    for i in range(NUM_ITERS):
        start = time.perf_counter()
        llm.generate(prompts, sampling_params, use_tqdm=True)
        end = time.perf_counter()
        elapsed_time += (end - start)
    
    mean_time = elapsed_time / NUM_ITERS
    print(f"Results calculated for model: {llm}")
    return mean_time



def main(args):
    if args.prefix_caching.lower()=="true":
        prefix_cache_enable=True
    else:
        prefix_cache_enable=False

    llm = LLM(model=args.model,
              tokenizer_mode='auto',
              trust_remote_code=True,
              enforce_eager=True,
              use_v2_block_manager=args.use_v2_block_manager,
              tensor_parallel_size=args.tensor_parallel_size,
              enable_prefix_caching=prefix_cache_enable,
              max_num_seqs = args.max_num_seqs,
              dtype=args.dtype,
              gpu_memory_utilization=args.gpu_memory_utilization,
              )

    num_prompts = 300
    prompts = [PROMPT] * num_prompts
    sampling_params = SamplingParams(temperature=0, max_tokens=args.output_len)

    elapsed_time=test_prefix(
        llm=llm,
        prompts=prompts,
        sampling_params=sampling_params,
    )
    

    if args.output_json:
        results = {
            "model":args.model,
            "output_len":args.output_len,
            "elapsed_time": elapsed_time,
            "prefix_cache":prefix_cache_enable,
            "num_gpus":args.tensor_parallel_size,
            "batch_size":args.max_num_seqs
        }
        with open(args.output_json, "w+") as f:
            json.dump(results, f, indent=4)

    

if __name__ == "__main__":
    parser = FlexibleArgumentParser(
        description='Benchmark the performance with or without automatic '
        'prefix caching.')
    parser.add_argument('--model',
                        type=str,
                        default='baichuan-inc/Baichuan2-13B-Chat')
    parser.add_argument('--tensor-parallel-size', '-tp', type=int, default=1)
    parser.add_argument('--output-len', type=int, default=10)
    # parser.add_argument('--enable-prefix-caching',
    #                     action='store_true',
    #                     help='enable prefix caching')
    parser.add_argument('--prefix-caching',
                        type=str,
                        choices=['True','False'],
                        help='enable prefix caching')
    parser.add_argument('--use-v2-block-manager',
                        action='store_true',
                        help='Use BlockSpaceMangerV2')
    parser.add_argument('--max-num-seqs',type=int, help="batch size")
    parser.add_argument(
        '--output-json',
        type=str,
        default=None,
        help='Path to save the throughput results in JSON format.')
    parser.add_argument(
        '--max-model-len',
        type=int,
        default=None,
        help='Maximum length of a sequence (including prompt and output). '
        'If None, will be derived from the model.')
    parser.add_argument('--dtype',
        type=str,
        default='float16',
        choices=['auto', 'half', 'float16', 'bfloat16', 'float', 'float32'],
        help='data type for model weights and activations. '
        'The "auto" option will use FP16 precision '
        'for FP32 and FP16 models, and BF16 precision '
        'for BF16 models.')
    parser.add_argument('--gpu-memory-utilization',
                        type=float,
                        default=0.9,
                        help='the fraction of GPU memory to be used for '
                        'the model executor, which can range from 0 to 1.'
                        'If unspecified, will use the default value of 0.9.')
    
    args = parser.parse_args()
    main(args)
