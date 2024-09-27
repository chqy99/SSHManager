import os
import re
import pandas as pd

client_res_pattern = [
  "Successful requests",
  "Benchmark duration (s)",
  "Total input tokens",
  "Total generated tokens",
  "Mean input tokens",
  "Median input tokens",
  "Max input tokens",
  "Mean generated tokens",
  "Median generated tokens",
  "Max generated tokens",
  "Request throughput (req/s)",
  "Input token throughput (tok/s)",
  "Output token throughput (tok/s)",
  "Mean Latency (ms)",
  "Median Latency (ms)",
  "P99 Latency (ms)",
  "Mean TTFT (ms)",
  "Median TTFT (ms)",
  "P99 TTFT (ms)",
  "Mean TPOT (ms)",
  "Median TPOT (ms)",
  "P99 TPOT (ms)"
]

server_run_columns = ["waiting", "running", "swapped",
              "wait_to_run_reqs", "run_to_wait_reqs", "wait_to_run_tokens",
              "batch_utils", "block_utils", "preempt_ratio"]
server_run_rows = ["Max", "Mean", "Min"]
server_run_pattern = re.compile(r'scheduler.py:117\]\s(?:Max|Mean|Min)\s+([\d.,]+)\s+([\d.,]+)\s+([\d.,]+)\s+([\d.,]+)\s+([\d.,]+)\s+([\d.,]+)\s+([\d.,]+)\s+([\d.,]+)\s+([\d.,]+)')

server_res_columns = ["ttft/s", "time_in_queue/s", "context_latency/s",
              "decoder_latency/s", "per_token_latency/s", "decoder_tokens"]
server_res_rows = ["Sum", "Max", "Mean", "Min", "P99"]
server_res_pattern = re.compile(r'scheduler.py:118\]\s(?:Sum|Max|Mean|Min|P99)\s+([\d.,]+)\s+([\d.,]+)\s+([\d.,]+)\s+([\d.,]+)\s+([\d.,]+)\s+([\d.,]+)')

def extract_log(logpath):
    with open(logpath, 'r', encoding='utf-8') as file:
        data = file.read()

    client_data, server_data = data.split("data split")

    client_data_lines = client_data.split('\n')

    client_res = {}
    for line in client_data_lines:
        for key in client_res_pattern:
            if key in line:
                value = line.split(":")[-1]
                client_res[key] = float(value.replace(' ', ''))

    matches = server_run_pattern.findall(server_data)
    server_run_df = pd.DataFrame(matches, columns=server_run_columns,
                                 index=server_run_rows)
    server_run_df = server_run_df.applymap(
        lambda x: float(x.replace(',', '')) if isinstance(x, str) else x)

    matches = server_res_pattern.findall(server_data)
    server_res_df = pd.DataFrame(matches, columns=server_res_columns,
                                 index=server_res_rows)
    server_res_df = server_res_df.applymap(
        lambda x: float(x.replace(',', '')) if isinstance(x, str) else x)

    return client_res, server_run_df, server_res_df

def handle_r_str(str):
    output = []
    lines = str.split('\n')
    for line in lines:
        split_r_lines = line.split('\r')
        ll = len(split_r_lines)
        if ll >= 2 and split_r_lines[ll - 2] != "":
            output.append(split_r_lines[ll - 2] + "\n")
    return output

def write_to_file(strlines, output_path):
    with open(output_path, 'a', encoding='utf-8') as file:
        file.writelines(strlines)

def extract_logs(folder_path, output_path):
    outfile = open(output_path, 'w')
    title = "model_name,dataset_name,enable_chunked,num_prompts,mns,mnbt,rr," \
            "max_batch_utils,mean_block_utils,max_block_utils,preempt_ratio," \
            "request_throughput,output_token_throughput,mean_ttft,p99_ttft," \
            "mean_tpot,p99_tpot,p99_time_in_queue,p99_time_context,p99_time_decoder\n"
    outfile.write(title)

    for root, dirs, files in os.walk(folder_path):
        for name in files:
            if 'log.txt' != name:
                continue
            file_path = os.path.join(root, name)
            tmp = file_path.split('output')[1].split('/')
            model_name = tmp[1]
            dataset_name = tmp[2]
            enable_chunked = tmp[3]
            num_prompts = tmp[4]
            mns, mnbt, rr = tmp[5].split('_')
            if enable_chunked == "disable_chunked":
                mnbt = None
            client_res, server_run_df, server_res_df = extract_log(file_path)
            if int(num_prompts) != int(client_res["Successful requests"]):
                print(f"{file_path} is wrong")
                continue

            max_batch_utils = server_run_df.loc["Max", "batch_utils"]
            mean_block_utils = server_run_df.loc["Mean", "block_utils"]
            max_block_utils = server_run_df.loc["Max", "block_utils"]
            preempt_ratio = server_run_df.loc["Max", "preempt_ratio"]
            request_throughput = client_res["Request throughput (req/s)"]
            output_token_throughput = client_res["Output token throughput (tok/s)"]
            mean_ttft = client_res["Mean TTFT (ms)"]
            p99_ttft = client_res["P99 TTFT (ms)"]
            mean_tpot = client_res["Mean TPOT (ms)"]
            p99_tpot = client_res["P99 TPOT (ms)"]

            p99_time_in_queue = 1000 * server_res_df.loc["P99", "time_in_queue/s"]
            p99_time_context = 1000 * server_res_df.loc["P99", "context_latency/s"]
            p99_time_decoder = 1000 * server_res_df.loc["P99", "per_token_latency/s"]

            write_str = f"{model_name},{dataset_name},{enable_chunked},{num_prompts},{mns},{mnbt},{rr}," \
                f"{max_batch_utils},{mean_block_utils},{max_block_utils},{preempt_ratio}," \
                f"{request_throughput},{output_token_throughput},{mean_ttft},{p99_ttft}," \
                f"{mean_tpot},{p99_tpot},{p99_time_in_queue:.0f},{p99_time_context:.0f},{p99_time_decoder:.0f}\n"
            outfile.write(write_str)

if __name__ == "__main__":
    extract_logs('output', 'output/table.csv')
