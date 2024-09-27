class OptiPlan:
    def __init__(self, chunked_prefill, step_cfg, limits):
        self.chunked_prefill = chunked_prefill
        self.step_cfg = step_cfg
        self.limits = limits

        self.input_params_list = []
        self.client_res_list = []
        self.eva_list = []
        self.server_run_df_list = []
        self.server_res_df_list = []
        self.opti_dir_list = []

        self.best_throughput = -1
        self.best_idx = -1
        self.opti_loss_times = 0

    def exist(self, input_params):
        for item in self.input_params_list:
            if item == input_params:
                return True
        return False

    def get_next_by_step(self, param_str, input, step):
        cfg = self.step_cfg[param_str]

        if abs(step) < cfg["min_step_num"]:
            return False, 0
        else:
            step = int(step / cfg["min_step_num"]) * cfg["min_step_num"]

        if input + step <= cfg["bound"][0]:
            if input == cfg["bound"][0]:
                return False, 0
            else:
                return True, cfg["bound"][0]

        if input + step >= cfg["bound"][1]:
            if input == cfg["bound"][1]:
                return False, 0
            else:
                return True, cfg["bound"][1]

        res = input + step
        if abs(res - int(res)) < 0.01:
            res = int(res)

        return True, res

    def evaluate_experiment(self, client_res):
        if client_res["P99 TTFT (ms)"] > self.limits["ttft_p99_limit"]:
            print("TTFT overtime")
            return -1
        if client_res["P99 TPOT (ms)"] > self.limits["tpot_p99_limit"]:
            print("TPOT overtime")
            return -1
        return client_res["Request throughput (req/s)"]

    # max_num_seqs 变化, idx 为实验序号
    def find_dir0(self, idx):
        inputs_in_dir = []
        for i in range(len(self.input_params_list)):
            if self.input_params_list[i][1] == self.input_params_list[idx][1] and \
               self.input_params_list[i][2] == self.input_params_list[idx][2]:
                inputs_in_dir.append([i, self.input_params_list[i][0]])

        value = self.input_params_list[idx][0]
        opti_dir = self.opti_dir_list[idx][0]
        param_str = "max_num_seqs"
        step = opti_dir * self.step_cfg[param_str]["step_num"]

        if len(inputs_in_dir) == 1:
            return self.get_next_by_step(param_str, value, step)
        # 按值大小排序
        inputs_in_dir = sorted(inputs_in_dir, key=lambda x: opti_dir * x[1])

        tmp = False
        last_eva = 0
        mulpy = 1
        for item in inputs_in_dir:
            if tmp:
                cur_step = item[1] - value
                # 如果调优后，结果变差
                if self.eva_list[idx] > self.eva_list[item[0]]:
                    return self.get_next_by_step(param_str, value, cur_step / 2)
                # 如果调优方向上，出现有效值，但本次是无效值
                if self.eva_list[idx] == -1 and self.eva_list[item[0]] > 0:
                    return self.get_next_by_step(param_str, value, cur_step / 2)
            if item[0] == idx:
                tmp = True

            # 连续的有效优化，加速优化
            if self.eva_list[item[0]] > last_eva and last_eva > 0:
                mulpy += 1
            # 连续的无效优化，加速优化
            elif self.eva_list[item[0]] == -1 and last_eva == -1:
                mulpy += 1
            else:
                mulpy = 1
            last_eva = self.eva_list[item[0]]

        return self.get_next_by_step(param_str, value, mulpy * step)

    # max_num_batched_tokens 变化, idx 为实验序号
    def find_dir1(self, idx):
        inputs_in_dir = []
        for i in range(len(self.input_params_list)):
            if self.input_params_list[i][0] == self.input_params_list[idx][0] and \
               self.input_params_list[i][2] == self.input_params_list[idx][2]:
                inputs_in_dir.append([i, self.input_params_list[i][1]])

        value = self.input_params_list[idx][1]
        opti_dir = self.opti_dir_list[idx][1]
        param_str = "max_num_batched_tokens"
        step = opti_dir * self.step_cfg[param_str]["step_num"]

        return self.get_next_by_step(param_str, value, step)
        # if len(inputs_in_dir) == 1:
        #     return self.get_next_by_step(param_str, value, step)
        # inputs_in_dir = sorted(inputs_in_dir, key=lambda x: opti_dir * x[1])

        # tmp = False
        # last_eva = self.eva_list[inputs_in_dir[0][0]]
        # mulpy = 1
        # for item in inputs_in_dir:
        #     if tmp:
        #         cur_step = item[1] - value
        #         # 如果调优后，结果变差
        #         if self.eva_list[idx] > self.eva_list[item[0]]:
        #             return self.get_next_by_step(param_str, value, cur_step / 2)
        #         # 如果调优方向上，出现有效值，但本次是无效值
        #         if self.eva_list[idx] == -1 and self.eva_list[item[0]] > 0:
        #             return self.get_next_by_step(param_str, value, cur_step / 2)
        #     if item[0] == idx:
        #         tmp = True

        #     if self.eva_list[item[0]] > last_eva and last_eva > 0:
        #         mulpy += 1
        #     elif self.eva_list[item[0]] == -1 and last_eva == -1:
        #         mulpy += 1
        #     else:
        #         mulpy = 1
        #     last_eva = self.eva_list[item[0]]

        # return self.get_next_by_step(param_str, value, mulpy * step)

    # request-rate 变化, idx 为实验序号
    def find_dir2(self, idx):
        inputs_in_dir = []
        for i in range(len(self.input_params_list)):
            if self.input_params_list[i][0] == self.input_params_list[idx][0] and \
               self.input_params_list[i][1] == self.input_params_list[idx][1]:
                inputs_in_dir.append([i, self.input_params_list[i][2]])

        value = self.input_params_list[idx][2]
        opti_dir = self.opti_dir_list[idx][2]
        param_str = "request-rate"
        step = opti_dir * self.step_cfg[param_str]["step_num"]

        if len(inputs_in_dir) == 1:
            return self.get_next_by_step(param_str, value, step)
        inputs_in_dir = sorted(inputs_in_dir, key=lambda x: opti_dir * x[1])

        tmp = False
        last_eva = 0
        mulpy = 1
        for item in inputs_in_dir:
            if tmp:
                cur_step = item[1] - value
                # 如果调优后，结果变差
                if self.eva_list[idx] > self.eva_list[item[0]]:
                    return self.get_next_by_step(param_str, value, cur_step / 2)
                # 如果调优方向上，出现有效值，但本次是无效值
                if self.eva_list[idx] == -1 and self.eva_list[item[0]] > 0:
                    return self.get_next_by_step(param_str, value, cur_step / 2)
            if item[0] == idx:
                tmp = True

            if self.eva_list[item[0]] > last_eva and last_eva > 0:
                mulpy += 1
            elif self.eva_list[item[0]] == -1 and last_eva == -1:
                mulpy += 1
            else:
                mulpy = 1
            last_eva = self.eva_list[item[0]]

        return self.get_next_by_step(param_str, value, mulpy * step)

    def Choose_next_by_dir(self, idx):
        flag = True
        mns = self.input_params_list[idx][0]
        mnbt = self.input_params_list[idx][1]
        rr = self.input_params_list[idx][2]

        if self.opti_dir_list[idx][0] != 0:
            flag, mns = self.find_dir0(idx)
        if self.opti_dir_list[idx][1] != 0:
            flag, mnbt = self.find_dir1(idx)
        if self.opti_dir_list[idx][2] != 0:
            flag, rr = self.find_dir2(idx)
        # 如果存在该实验，返回错误
        if self.exist([mns, mnbt, rr]):
            return False, 0, 0, 0
        # mnbt 不小于 mns
        if mns > mnbt:
                mns = mnbt
        return flag, mns, mnbt, rr

    def append_experiment(self, input_params, client_res, server_run_df, server_res_df):
        # 实验参数
        self.input_params_list.append(input_params)
        # client端结果，字典数据，单位为 ms
        self.client_res_list.append(client_res)
        eva = self.evaluate_experiment(client_res)
        self.eva_list.append(eva)
        # server端抓取信息，pandas 的 dataframe 数据，单位为 s
        self.server_run_df_list.append(server_run_df)
        self.server_res_df_list.append(server_res_df)
        cur_idx = len(self.input_params_list) - 1

        if eva > self.best_throughput:
            self.best_throughput = eva
            self.best_idx = cur_idx
            self.opti_loss_times = 0
        elif self.best_throughput > 0:
            self.opti_loss_times += 1
        if self.opti_loss_times >= self.limits["opti_loss_limit"]:
            return False, 0, 0, 0

        # 如果调优后结果变差
        if len(self.eva_list) > 1 and \
            self.eva_list[-2] > self.eva_list[-1]:
            # 沿用上一次调优方向，重新调优
            opti = self.opti_dir_list[-1]
            self.opti_dir_list.append(opti)
            return self.Choose_next_by_dir(cur_idx - 1)
        else:
            if self.chunked_prefill:
                opti = self.get_opti_dir2(input_params, client_res,
                                           server_run_df, server_res_df)
            else:
                opti = self.get_opti_dir1(input_params, client_res,
                                           server_run_df, server_res_df)
        self.opti_dir_list.append(opti)
        return self.Choose_next_by_dir(cur_idx)

    # disable_chunked_prefill 评估实验结果，指示调优方向
    def get_opti_dir1(self, input_params, client_res, server_run_df, server_res_df):
        Max_batch_utils = server_run_df.loc["Max", "batch_utils"] * 100
        Max_block_utils = server_run_df.loc["Max", "block_utils"] * 100
        Max_running = server_run_df.loc["Max", "running"]

        T_time_in_queue = 1000 * server_res_df.loc["P99", "time_in_queue/s"]
        T_context_infer = 1000 * server_res_df.loc["P99", "context_latency/s"]
        T_decoder_infer = 1000 * server_res_df.loc["P99", "per_token_latency/s"]

        opti = [0,0,0]

        if client_res["P99 TTFT (ms)"] > self.limits["ttft_p99_limit"] or \
            client_res["P99 TPOT (ms)"] > self.limits["tpot_p99_limit"]:
            if T_time_in_queue > self.limits["ttft_p99_limit"]:
                opti[2] = -1
            # elif T_context_infer > self.limits["ttft_p99_limit"]:
            #     opti[1] = -1
            elif T_decoder_infer > self.limits["tpot_p99_limit"]:
                opti[0] = -1
            elif T_time_in_queue + T_context_infer + T_decoder_infer > self.limits["ttft_p99_limit"]:
                opti[2] = -0.5
            elif client_res["P99 TTFT (ms)"] > self.limits["ttft_p99_limit"]:
                # TTFT 超时，默认调节 request-rate
                opti[2] = -0.5
            else:
                # TPOT 超时
                if client_res["P99 TPOT (ms)"] < 1.2 * self.limits["tpot_p99_limit"] \
                    and input_params[2] > 1:
                    # tpot 略微超出限制，减小 request-rate
                    opti[2] = -0.5
                else:
                    # 默认调节 max_num_seqs
                    opti[0] = -0.5
        else:
            if client_res["P99 TPOT (ms)"] <= self.limits["tpop_lower_limit"] and \
                Max_running == input_params[0]:
                # TPOT 偏低时，调大 max_num_seqs
                opti[0] = 1
            elif Max_batch_utils <= self.limits["batch_lower_limit"] or \
                Max_block_utils <= self.limits["block_lower_limit"]:
                # batch/block 利用率偏低时，调大 request-rate
                opti[2] = 1

        return opti

    # enable_chunked_prefill
    def get_opti_dir2(self, input_params, client_res, server_run_df, server_res_df):
        Max_batch_utils = server_run_df.loc["Max", "batch_utils"] * 100
        Max_block_utils = server_run_df.loc["Max", "block_utils"] * 100
        Max_running = server_run_df.loc["Max", "running"]
        # Mean_running = server_run_df.loc["Mean", "running"]

        T_time_in_queue = 1000 * server_res_df.loc["P99", "time_in_queue/s"]
        T_context_infer = 1000 * server_res_df.loc["P99", "context_latency/s"]
        T_decoder_infer = 1000 * server_res_df.loc["P99", "per_token_latency/s"]

        opti = [0,0,0]

        if client_res["P99 TTFT (ms)"] > self.limits["ttft_p99_limit"] or \
            client_res["P99 TPOT (ms)"] > self.limits["tpot_p99_limit"]:
            if T_time_in_queue > self.limits["ttft_p99_limit"]:
                opti[2] = -1
            elif T_context_infer > self.limits["ttft_p99_limit"]:
                opti[1] = -1
            elif T_decoder_infer > self.limits["tpot_p99_limit"]:
                # 略微小于 tpot
                if T_decoder_infer < 1.2 * self.limits["tpot_p99_limit"]:
                    opti[1] = -0.5
                else:
                    opti[0] = -1
            elif T_time_in_queue + T_context_infer + T_decoder_infer > self.limits["ttft_p99_limit"]:
                opti[2] = -0.5
            elif client_res["P99 TTFT (ms)"] > self.limits["ttft_p99_limit"]:
                # TTFT 超时，默认调节 request-rate
                opti[2] = -0.5
            else:
                # TPOT 超时
                if client_res["P99 TPOT (ms)"] < 1.2 * self.limits["tpot_p99_limit"] \
                    and input_params[2] > 1:
                    # tpot 略微超出限制，减小 request-rate
                    opti[2] = -0.5
                elif Max_running > 0.9 * input_params[0] and input_params[0] > 8:
                    opti[0] = -0.5
                else:
                    opti[1] = -0.5
        else:
            if Max_batch_utils <= self.limits["batch_lower_limit"] or \
                Max_block_utils <= self.limits["block_lower_limit"]:
                # batch/block 利用率偏低时，调大 request-rate
                opti[2] = 1
            elif client_res["P99 TPOT (ms)"] <= self.limits["tpop_lower_limit"]:
                # TPOT 偏低时
                if Max_running == input_params[0]:
                    # 序列数达到上限，调大 max_num_seqs
                    opti[0] = 1
                else:
                    # 调大 max_num_batched_tokens，减小 ttft，提升 tpot
                    opti[1] = 1
            else:
                P_D_ratio = client_res["Total input tokens"] / client_res["Total generated tokens"]
                C_B_ratio = (input_params[1] - Max_running + 1) / (Max_running - 1)
                print("P_D_ratio:", P_D_ratio)
                print("C_B_ratio:", C_B_ratio)
                if P_D_ratio  > 1.25 * C_B_ratio:
                    opti[1] = 0.5
                elif P_D_ratio < 0.8 * C_B_ratio:
                    if Max_running == input_params[0]:
                        opti[0] = 0.5
                    else:
                        opti[1] = -0.5
        return opti