# 数学函数
def draw_log():
    import matplotlib.pyplot as plt
    import numpy as np

    # 生成一些数据
    x = np.linspace(0.1, 10, 100)
    # 默认log指的时log_e，还可以支持log10
    y = np.log(x)

    # 创建一个新的图形和轴
    fig, ax = plt.subplots()

    # 绘制log图像
    ax.plot(x, y)

    # 设置图像标题和轴标签
    ax.set_title("Logarithmic Function")
    ax.set_xlabel("x")
    ax.set_ylabel("log(x)")

    # 显示图像
    plt.show()
