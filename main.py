import os
# 获取当前文件夹的路径
current_folder = os.path.dirname(os.path.abspath(__file__))
# 设置工作目录为当前文件夹
os.chdir(current_folder)

import yaml
from ssh_tools import SSHManager
from log_process import extract_log, handle_r_str, write_to_file
from optimizer import OptiPlan

with open('config.yaml', 'r', encoding='utf-8') as file:
    config = yaml.safe_load(file)

ip = config["ssh_setting"]["ip"]
port = config["ssh_setting"]["port"]
username = config["ssh_setting"]["username"]
password = config["ssh_setting"]["password"]
prompt = config["ssh_setting"]["prompt"]
server_config = config["server_config"]
client_config = config["client_config"]
app_ip = server_config["app_ip"]
app_port = server_config["app_port"]
num_prompts = config["params"]["num-prompts"]
disable_chunked_config = config["params"]["disable_chunked_prefill"]
enable_chunked_config = config["params"]["enable_chunked_prefill"]
models_folder = config["models_folder"]
datasets_folder = config["datasets_folder"]

class vllm_experiment:
    def __init__(self, model_name, dataset_name, num_prompts, output_folder='output'):
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.num_prompts = num_prompts
        self.server_ssh = None
        self.client_ssh = None
        self.folder_path = output_folder + '/' + model_name + '/' + dataset_name + '/' + \
                            str(num_prompts) + '/'

    # 前处理，只需处理一次
    def set_env(self):
        self.server_ssh = SSHManager(ip, username, password, "[server]:", port, prompt)
        self.client_ssh = SSHManager(ip, username, password, "[client]:", port, prompt)

        for cmd in server_config["pre_cmds"]:
            self.server_ssh.execute_command_wait_finish(cmd, max_duration=3)

        for cmd in client_config["pre_cmds"]:
            self.client_ssh.execute_command_wait_finish(cmd, max_duration=3)

    # mns means max_num_seqs
    # mnbt means max_num_batched_tokens
    # rr means request_rate
    def item_test(self, chunked_prefill, mns, mnbt, rr):
        chunked_str = "enable_chunked" if chunked_prefill else "disable_chunked"
        son_folder = chunked_str + "/" + str(mns) + "_" + str(mnbt) + "_" + str(rr)
        print("\n======ItemTest:", son_folder, "======\n")

        item_folder = self.folder_path + son_folder
        log_path = item_folder + "/log.txt"

        # 如果实验已经存在
        if os.path.exists(log_path):
            print(f"The experiment has been completed. Result in {log_path}")
            return True, log_path
        else:
            os.makedirs(item_folder, exist_ok=True)

        # 每次重启环境，稳定性更好
        self.set_env()
        # server launch
        model_config = config["models"][self.model_name]
        model_path = models_folder + model_config["repath"]
        server_cmd = "python -m vllm.entrypoints.openai.api_server --trust-remote-code --model " + \
            model_path + " -tp " + str(model_config["tp"]) + " --host " + app_ip + \
            " --port " + str(app_port) + " --max_num_seqs=" + str(mns)
        if chunked_prefill:
            server_cmd += " --enable_chunked_prefill --max_num_batched_tokens=" + str(mnbt)

        self.server_ssh.execute_command(server_cmd)
        status, _ = self.server_ssh.read_until_prompt("Uvicorn running on", show_log=True)
        if not status:
            self.server_ssh.execute_command(chr(3))
            return False, ""
        print("\nserver launched\n")

        dataset_config = config["datasets"][self.dataset_name]
        dataset_path = datasets_folder + dataset_config["repath"]
        client_cmd = "python benchmark_serving2.py --backend vllm --trust-remote-code --model " + \
                    model_path + " --dataset-name " + self.dataset_name + \
                    " --dataset-path " + dataset_path + " --num-prompts=" + \
                    str(self.num_prompts) + " --request-rate=" + str(rr) + \
                    " --host " + app_ip + " --port " + str(app_port)

        self.client_ssh.execute_command(client_cmd)
        # 必须马上读取，否则服务器端有过多的日志，使得缓冲区堵塞，会导致程序卡住
        thread_num = self.server_ssh.start_recv_thread()
        status, log_data = self.client_ssh.read_until_prompt(prompt, show_log=True)
        self.server_ssh.stop_thread(thread_num)

        if not status:
            self.server_ssh.execute_command(chr(3))
            return False, ""

        write_to_file(handle_r_str(log_data), log_path)
        write_to_file("\ndata split\n", log_path)

        # 额外信息抓取
        client_cmd = "python3 " + client_config["utils_path"] + " --host " + \
                    app_ip + " --port " + str(app_port) + " --action save"
        log_data = self.client_ssh.execute_command_wait_finish(client_cmd)

        # 只存储 vLLM scheduler profiling save... 之后的字符
        status, log_data = self.server_ssh.read_until_prompt("vLLM scheduler profiling save...") \
                                  .split("vLLM scheduler profiling save...")[1]
        if not status:
            self.server_ssh.execute_command(chr(3))
            return False, ""
        status, log_data += self.server_ssh.read_until_prompt("/v1/completions HTTP/", max_duration=3)
        if not status:
            self.server_ssh.execute_command(chr(3))
            return False, ""
        print("[running statistics]:", log_data)
        write_to_file(log_data, log_path)
        print("\n======ItemTest finish======\n")

        self.post_handle(item_folder)
        return True, log_path

    def post_handle(self, item_folder):
        for cmd in server_config["post_cmds"]:
            self.server_ssh.execute_command_wait_finish(cmd, max_duration=3)

        for cmd in client_config["post_cmds"]:
            self.client_ssh.execute_command_wait_finish(cmd, max_duration=3)

        # 远端文件 scp 到本地
        import subprocess
        scp_command = [
            'scp',
            # username + "@" + ip + ":/home/chenqiyang/tmp/vllm_scheduler*",
            f"-P {port} {ip}:/workspace/volume/chenqiyang/vllm_test/vllm_scheduler*",
            item_folder
        ]
        subprocess.run(scp_command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # 关闭连接
        self.server_ssh.close()
        self.client_ssh.close()

    def opti_experiment(self, chunked_prefill):
        chunked_str = "enable_chunked" if chunked_prefill else "disable_chunked"

        if chunked_prefill:
            params_config = enable_chunked_config
        else:
            params_config = disable_chunked_config

        mns_cfg = params_config["max_num_seqs"]
        mnbt_cfg = params_config["max_num_batched_tokens"]
        rr_cfg = params_config["request-rate"]

        mns = mns_cfg["default"]
        mnbt = mnbt_cfg["default"]
        rr = rr_cfg["default"]
        if mns > mnbt:
            mns = mnbt
        status, result_txt = self.item_test(chunked_prefill, mns, mnbt, rr)
        if not status:
            return False

        input_params = [mns, mnbt, rr]
        cli_res, ser_run, ser_res = extract_log(result_txt)

        vllm_opti = OptiPlan(chunked_prefill, params_config, config["limitation"])
        flag, mns, mnbt, rr = vllm_opti.append_experiment(input_params, cli_res,
                                                          ser_run, ser_res)

        while flag:
            status, result_txt = self.item_test(chunked_prefill, num_prompts, mns, mnbt, rr)
            if not status:
                return False

            input_params = [mns, mnbt, rr]
            cli_res, ser_run, ser_res = extract_log(result_txt)

            flag, mns, mnbt, rr = vllm_opti.append_experiment(input_params, cli_res,
                                                              ser_run, ser_res)

        print(f"\n======{chunked_str} Opti Finish======")
        print("best params:", vllm_opti.input_params_list[vllm_opti.best_idx])

        return True


for model, _ in config["models"].items():
    for dataset, _ in config["datasets"].items():
        from datetime import datetime
        current_time_str = datetime.now().strftime("%Y-%m-%d-%H:%M")
        ve = vllm_experiment(model, dataset, num_prompts, "2-A100/")
        # enable_chunked 调优
        ve.opti_experiment(True)
        # disable_chunked 调优
        ve.opti_experiment(False)