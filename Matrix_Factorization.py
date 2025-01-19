import numpy as np
import gc

"""
构建矩阵分解函数（MF），目的通过“用户-物品评分矩阵”已知的评分数据来预测缺失的评分，采用SGD（随机梯度下降）来优化矩阵因子和偏差项
"""

class MF():
    def __init__(self,X,k,alpha,beta,iterations):
        """
        self.X 输入的用户——评分矩阵，缺失值为nan
        self.k 用户特征矩阵U和物品特征矩阵V的隐性因子个数
        self.alpha 学习率，控制每次梯度更新步长
        self.beta 正则化参数，防止过拟合，控制矩阵U V大小
        self.iterations 随机梯度下降迭代次数
        self.not_nan_index 记录有效元素（非空）索引
        """
        self.X = X
        self.num_samples,self.num_features = X.shape#分别代表用户和电影数量
        self.k = k
        self.alpha = alpha
        self.beta = beta
        self.iterations = iterations
        self.not_nan_index = np.isnan(self.X) == False

    def train(self):

        #初始化分解矩阵U、V
        self.U = np.random.normal(scale=1./self.k,size=(self.num_samples,self.k))
        self.V = np.random.normal(scale=1./self.k,size=(self.num_features,self.k))

        #初始化偏置项
        self.b = np.mean(self.X[self.not_nan_index])
        #全局变量b设置为“用户-评分矩阵”非空值的均值，目的是给模型一开始一个基准评分，让模型捕捉整体趋势
        self.b_u = np.zeros(self.num_samples)
        self.b_v = np.zeros(self.num_features)

        #创建训练样本
        self.samples = [(i,j,self.X[i,j])
                        for i in range(self.num_samples)
                        for j in range(self.num_features)
                        if not np.isnan(self.X[i,j])]

        #执行随机梯度下降
        training_process = []
        for i in range(self.iterations):
            np.random.shuffle(self.samples)
            self.sgd()#执行梯度更新
            se = self.square_error()#预测评分与实际评分的平方误差
            training_process.append((i,se))
            if (i+1) % 10 == 0:#10次更新一次损失数据
                print('当前迭代次数：{}，预测评分与实际评分的平方误差值={:.4f}'.format(i+1,se))

    def sgd(self):
        """
        执行随机梯度下降，并进行更新
        """
        for i,j,x in self.samples:
            prediction = self.get_x(i,j)#计算（i,j）位置的预测评分
            e = (x - prediction)#当前预测评分与真实评分的误差
            #print(f"样本误差 ({i},{j}): {e}")

            #更新偏置项:通过误差e来更新
            self.b_u[i] += self.alpha * (2 * e - self.beta * self.b_u[i])
            self.b_v[j] += self.alpha * (2 * e - self.beta * self.b_v[j])

            #更新分解矩阵（隐含因子矩阵）U、V,每次都会根据误差e和用户、物品的特征更新相应的行
            self.U[i,:] += self.alpha * (2*e*self.V[j,:] - self.beta*self.U[i,:])
            self.V[j,:] += self.alpha * (2*e*self.U[i,:] - self.beta*self.V[j,:])

            # 处理 NaN 或无穷大的数值
            self.U[i, :] = np.nan_to_num(self.U[i, :])
            self.V[j, :] = np.nan_to_num(self.V[j, :])

            # 检查更新后 U 和 V 的值
            #print(f"U[{i}, :]: {self.U[i, :]}")
            #print(f"V[{j}, :]: {self.V[j, :]}")

        #手动垃圾回收以释放内存
        gc.collect()# 清理不再需要的内存

    def get_x(self,i,j):
        """
        计算（i,j）位置的预测评分
        """
        prediction = self.U[i,:].dot(self.V[j,:].T) + self.b +self.b_u[i] + self.b_v[j]
        return prediction

    def square_error(self):
        """该方法计算矩阵分解模型的平方误差。平方误差表示模型的预测评分与实际评分之间的差异"""
        predicted = self.full_matrix()#获取完整的评分矩阵
        error = 0
        for i in range(self.num_samples):
            for j in range(self.num_features):
                if self.not_nan_index[i,j]:
                    #只对非空的有效评分进行误差平方和计算，当self.not_nan_index[i,j]为TRUE才会下一步
                    error += pow(self.X[i,j] - predicted[i,j],2)
        return error

    def full_matrix(self):
        """
        使用当前的U、V和偏置项计算完整的评分预测矩阵
        b_u是一维数组，转换为（n,1)，在加法过程中会广播成为(n,m)给每一个评分都加上用户偏置项
        b_v是一维数组，转换为（1,m)，在加法过程中会广播成为(n,m)给每一个评分都加上物品偏置项
         """
        return self.U.dot(self.V.T) + self.b + self.b_u[:,np.newaxis] + self.b_v[np.newaxis,:]

    def replace_nan(self,x_hat):
        """将预测矩阵 X_hat 中对应的评分替换掉原始矩阵 X 中的 NaN 值，返回替换后的矩阵x_new"""
        x_new = np.copy(self.X)
        for i in range(self.num_samples):
            for j in range(self.num_features):
                if np.isnan(x_new[i,j]):
                    x_new[i,j] = x_hat[i,j]
        return x_new

if __name__ == '__main__':
    X = np.array([
        [5, 3, 0, 1],
        [4, 0, 0, 1],
        [1, 1, 0, 5],
        [1, 0, 0, 4],
        [0, 1, 5, 4],
    ], dtype=np.float32)
    X[X == 0] = np.nan
    print('替换nan后的矩阵：',X)

    #测试函数
    mf = MF(X,k=2,alpha=0.1,beta=0.1,iterations=100)
    mf.train()
    x_hat = mf.full_matrix()
    x_comp = mf.replace_nan(x_hat)#获得将预测值替换nan后矩阵

    print('评分预测矩阵',x_hat)
    print('在原用户-评分矩阵上替换nan值的完整矩阵',x_comp)













