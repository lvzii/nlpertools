import time

import psutil


def memory_monitor():
    while True:
        # 获取当前时间
        current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

        # 获取内存使用情况
        mem = psutil.virtual_memory()
        total = round(mem.total / 1024 / 1024 / 1024, 2)
        available = round(mem.available / 1024 / 1024 / 1024, 2)
        # 将内存使用情况写入文件
        with open("mem_usage_cpu.txt", "a") as f:
            f.write(f"{current_time} - Total: {total}, Available: {available}\n")

        time.sleep(10)


if __name__ == "__main__":
    print("start cpu memory monitor!")
    memory_monitor()
