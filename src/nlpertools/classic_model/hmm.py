import numpy as np


class HMM:
    def __init__(self, pi, A, B):
        self.pi = pi
        self.A = A
        self.B = B

    def get_data_with_distribute(self, dist):
        """
        input example : [0.2,0.3,0.5]
        output example: 1
        根据给定的概率分布随机返回数据
        """
        r = np.random.rand()
        for i, p in enumerate(dist):
            if r < p:
                return i
            r -= p

    def generate(self, T):
        """
        T:指定生成的数量
        """
        z = self.get_data_with_distribute(self.pi)
        x = self.get_data_with_distribute(self.B[z])
        result = [x]
        for _ in range(T - 1):
            z = self.get_data_with_distribute(self.A[z])
            x = self.get_data_with_distribute(self.B[z])
            result.append(x)
        return result


# 状态转移矩阵
A = np.array([[0, 1, 0, 0], [0.4, 0, 0.6, 0], [0, 0.4, 0, 0.6], [0, 0, 0.5, 0.5]])
# 观测概率矩阵/发射矩阵
B = np.array([[0.5, 0.5], [0.3, 0.7], [0.6, 0.4], [0.8, 0.2]])
# 初始概率分布
pi = np.array([0.25, 0.25, 0.25, 0.25])

hmm = HMM(pi, A, B)
print(hmm.generate(10))  # 生成10个数据
