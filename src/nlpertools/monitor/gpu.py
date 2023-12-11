import os
import time


def gpu_memory():
    while 1:
        current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        usage = os.popen("nvidia-smi --query-gpu=memory.used --format=csv").read()
        usage_string = ",".join(usage.split("\n"))
        # 将内存使用情况写入文件
        with open("mem_usage_gpu.txt", "a") as f:
            f.write(f"{current_time} - Usage: {usage_string}\n")
        time.sleep(1)


if __name__ == "__main__":
    print("start gpu memory monitor!")
    gpu_memory()
