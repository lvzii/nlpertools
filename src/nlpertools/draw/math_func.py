# 数学函数
def draw_log():
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.ticker import MultipleLocator, FormatStrFormatter

    # 生成一些数据
    x = np.linspace(0.1, 10, 100)
    # 默认log指的时loge
    y = np.log(x)

    # 创建一个新的图形和轴
    fig, ax = plt.subplots()

    # 绘制log图像
    ax.plot(x, y)

    # 设置图像标题和轴标签
    ax.set_title("Logarithmic Function")
    ax.set_xlabel("x")
    ax.set_ylabel("log(x)")
    # 设置横坐标的刻度间隔为1
    ax.xaxis.set_major_locator(MultipleLocator(1))

    # 设置横坐标的刻度格式
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    # 添加x=1的虚线
    ax.axvline(x=1, linestyle="--", color="gray")
    # 添加y=1的虚线
    ax.axhline(y=0, linestyle="--", color="gray")

    # 显示图像
    plt.show()
