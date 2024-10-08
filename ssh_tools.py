import paramiko
import time
import threading

class SSHManager:
    def __init__(self, ip, username, password, ssh_name="", port=22, final_prompt="$ "):
        super().__init__()
        self.ip = ip
        self.port = port
        self.username = username
        self.password = password
        self.final_prompt = final_prompt
        self.ssh_name = ssh_name
        self.ssh = None
        self.sftp = None
        self.channel = None
        self.threads = {} # 保存线程实例
        self.connect()

    def connect(self):
        self.ssh = paramiko.SSHClient()
        self.ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        self.ssh.connect(self.ip, self.port, self.username, self.password)
        self.sftp = self.ssh.open_sftp()
        self.channel = self.ssh.invoke_shell()
        self.channel.settimeout(3)
        print(self.ssh_name, "ssh connect")
        # 连接命令的缓冲区立即读取
        self.read_until_prompt(self.final_prompt)

    # 阻塞式读取
    # prompt，读取到指定字符串时停止
    # max_duration，该命令最长执行时间(s)
    # once_max_wait，单次读取缓冲区最长等待时间(s)
    # show_log，是否打印读取的信息
    # buffer_size，单次读取缓冲区大小
    # interval，单次读取失败后的间隔时间(s)
    def read_until_prompt(self,
                          prompt,
                          max_duration=3600,
                          once_max_wait = 360,
                          show_log=False,
                          buffer_size=1024,
                          interval=1):
        output = ''
        buffer = b''  # 使用字节类型的缓冲区
        # last_decode = True

        if once_max_wait > max_duration:
            once_max_wait = max_duration

        start_time = time.time()
        last_recv_time = start_time
        while time.time() - start_time < max_duration:
            # 等待有效读取出现
            while not self.channel.recv_ready():
                time.sleep(interval)  # 避免过高的 CPU 使用率
                # 超时检测
                if time.time() - last_recv_time >= once_max_wait:
                    print(self.ssh_name, "Reached once_max_wait")
                    return output
                # 出现错误流时，抛出异常
                if self.channel.recv_stderr_ready():
                    print(self.ssh_name, "an error occurred")
                    print(self.channel.recv_stderr(buffer_size))
                    raise RuntimeError(self.ssh_name)

            buffer += self.channel.recv(buffer_size)
            last_recv_time = time.time()

            # 缓冲区转换成 utf-8 编码
            try:
                recv = buffer.decode('utf-8')
                if show_log:
                    print(recv, end="")
                buffer = b''
                # if not last_decode:
                    # print(self.ssh_name, "redecode success")
                    # last_decode = True
            except UnicodeDecodeError as e:
                # print(self.ssh_name, f"decode error: {e}, try read more buffer to decode")
                # last_decode = False
                continue

            output += recv
            # 查询字符可能分割在两个 recv 中
            if prompt and prompt in output:
                break

        return output

    def execute_command(self,
                        command,
                        max_duration=3600,
                        once_max_wait = 360,
                        show_log=False,
                        buffer_size=1024,
                        interval=1):
        print(self.ssh_name, command)
        self.channel.send(command + "\r")
        return self.read_until_prompt(self.final_prompt, max_duration, once_max_wait,
                                      show_log, buffer_size, interval)

    # 可以搭配线程使用，防止缓冲区堵塞
    def execute_command_async(self, command):
        print(self.ssh_name, command)
        self.channel.send(command + "\r")

    # server 端一直运行，需要靠线程实现非阻塞式地读取缓冲区，否则缓冲区堵塞会造成错误
    def start_recv_thread(self, max_duration=3600, buffer_size=4096, interval=0.1):
        """ 开启新线程以持续接收数据 """
        thread_name = f"RecvThread-{len(self.threads) + 1}"
        thread = threading.Thread(target=self.recv_thread,
                                args=(thread_name,max_duration,buffer_size,interval))
        self.threads[thread_name] = {'thread': thread, 'running': True}
        thread.start()
        return thread_name

    def recv_thread(self, thread_name, max_duration=3600, buffer_size=4096, interval=0.1):
        """ 线程运行的函数，不断接收数据，防止缓冲区占满 """
        print(f"Thread {thread_name} started")
        start_time = time.time()
        while self.threads[thread_name]['running']:
            if self.channel.recv_ready():
                self.channel.recv(buffer_size)
            time.sleep(interval)
            if time.time() - start_time >= max_duration:
                print(f"Thread {thread_name} Reached maximum duration.")
                return
        print(f"Thread {thread_name} stopped")

    def stop_thread(self, thread_name):
        """ 停止指定的线程 """
        if thread_name in self.threads:
            self.threads[thread_name]['running'] = False
            self.threads[thread_name]['thread'].join()  # 等待线程结束
            del self.threads[thread_name]
            print(f"Thread {thread_name} has been stopped")
        else:
            print(f"Thread {thread_name} not found")

    def download_file(self, remote_path, local_path):
        try:
            print(f"下载文件: {remote_path} 到 {local_path}")
            self.sftp.get(remote_path, local_path)
            print(f"成功下载: {remote_path}")
        except Exception as e:
            import sys
            print(f"下载文件失败: {remote_path}. 错误: {e}", file=sys.stderr)

    def download_directory(self, remote_dir, local_dir):
        import os
        import stat
        os.makedirs(local_dir, exist_ok=True)
        for entry in self.sftp.listdir_attr(remote_dir):
            remote_path = os.path.join(remote_dir, entry.filename)
            local_path = os.path.join(local_dir, entry.filename)
            if stat.S_ISDIR(entry.st_mode):
                self.download_directory(remote_path, local_path)
            else:
                self.download_file(remote_path, local_path)

    def close(self):
        if self.ssh:
            self.sftp.close()
            self.ssh.close()
            print(self.ssh_name, "ssh close")
            self.ssh = None
            # 停止所有线程
            for thread_name in list(self.threads.keys()):
                self.stop_thread(thread_name)

    def __del__(self):
        self.close()
