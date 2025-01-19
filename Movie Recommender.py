import os
import pandas as pd
import numpy as np
from Matrix_Factorization import MF
import pickle

#一、用户评分数据集加载，转换为用户-电影评分矩阵
ra_data_path = 'ml-latest-small/ml-latest-small/ratings.csv'
mf_cache_dir = 'mf_cache'#缓存文件夹
mf_cache_path = os.path.join(mf_cache_dir,'mf_RatingMatrix_cache')
def load_data(data_path,cache_path):
    print('开始分批加载数据集...')
    if not os.path.exists(mf_cache_dir):
        os.makedirs(mf_cache_dir)

    #读取用户-电影评分矩阵
    if os.path.exists(mf_cache_path):
        print('加载缓冲中...')
        rating_matrix = pd.read_pickle(mf_cache_path)
        print('从缓存加载数据集完毕')

    else:
        # 没有缓存就分批处理数据并保存为pickle格式
        dtypes = {'userId':np.int32,'movieId':np.int32,'rating':np.float32}
        print('加载新数据中')

        # 加载前三列数据：用户ID、电影ID、评分
        ratings = pd.read_csv(data_path,dtype=dtypes,usecols=range(3))

        # 转换为用户-电影评分矩阵
        rating_matrix = pd.pivot_table(data=ratings,index=['userId'],columns=['movieId'],values='rating')

        #缓存数据
        rating_matrix.to_pickle(cache_path)
        print('数据加载完毕')
    return rating_matrix
rating_matrix = load_data(ra_data_path,mf_cache_path)
rating_matrix_np = rating_matrix.to_numpy()
print('矩阵类型是否是np',type(rating_matrix_np))
print(rating_matrix_np.shape)
print(rating_matrix_np[:10,:])

#二、电影名称数据集加载
movies_data_path = 'ml-latest-small/ml-latest-small/movies.csv'
dtypes_movies = {'movieId':np.int32,'title':np.str_,'genres':np.str_}
movies = pd.read_csv(movies_data_path,dtype=dtypes_movies)
movies_np = movies.to_numpy()#转换成numpy数组运算内存占用小
print('电影名称文件大小和预览')
print(movies_np.shape)
print(movies_np[:10,:])


#三、调用MF矩阵分解函数
"""
1、k值选择：通过实现选择最合适的k值，K一般在10-100之间，k值要比用户、电影数量最小值要小，要不然不能体现特征优化
"""
#1、计算评查误差和，来判定k值
def compute_rmse(rating_matrix,mf):
    #获取预测的矩阵
    predicted_matrix = mf.full_matrix()
    #计算平方误差和
    square_error_sum = mf.square_error()
    #计算非nan的评分数量,~是反运算符号，也就是取非空值为TRUE的数量
    non_nan_count = np.sum(~np.isnan(rating_matrix))
    #计算平方根误差和
    rmse = np.sqrt(square_error_sum/non_nan_count)
    return rmse

#2、迭代挑选最优的K值
#k_values = [60,70,80,90,100,110]
#best_k = None
#best_rmse = float('inf')#取正无穷大

#for k in k_values:
#    mf = MF(rating_matrix_np,k=k,alpha=0.01,beta=0.1,iterations=100)
#    training_process = mf.train()
#    rmse = compute_rmse(rating_matrix_np,mf)
 #   print('k值={},RMSE={}'.format(k,rmse))

#    if rmse <best_rmse:#如果迭代的K误差更低，就判定最最优
#        best_rmse = rmse
#        best_k = k
#print('最优的K值为：{}，RMSE={}'.format(best_k,best_rmse))
"""
最优的K值为：110，RMSE=0.2890092949391867
"""

#2、缓存结果
mfOutcome_cache_path = os.path.join(mf_cache_dir,'mf_outcome')
if not os.path.exists(mf_cache_dir):
    os.makedirs(mf_cache_dir)

# 读取用户-电影评分矩阵
if os.path.exists(mfOutcome_cache_path):
    print('加载缓冲中...')
    try:
        with open(mfOutcome_cache_path, 'rb') as f:
            x_comp = pickle.load(f)
        print('从缓存加载数据集完毕')
    except EOFError:
        print('缓存文件损坏，重新训练并保存')
        os.remove(mfOutcome_cache_path)
        # 强制重新训练
        mf = MF(rating_matrix_np, k=110, alpha=0.01, beta=0.1, iterations=100)
        mf.train()
        x_hat = mf.full_matrix()
        x_comp = mf.replace_nan(x_hat)
        # 保存模型结果
        with open(mfOutcome_cache_path, 'wb') as f:
            pickle.dump(x_comp, f)
        print('模型已重新训练并保存到缓存')

else:
    print('加载新数据中')
    mf = MF(rating_matrix_np, k=110, alpha=0.01, beta=0.1, iterations=100)
    mf.train()
    x_hat = mf.full_matrix()
    x_comp = mf.replace_nan(x_hat)
    print('原始用户—评分矩阵', rating_matrix_np[:40, :])
    print('评分预测矩阵', x_hat[:40, :])
    print('在原用户-评分矩阵上替换nan值的完整矩阵', x_comp[:40, :])
    #保存模型结果
    with open(mfOutcome_cache_path, 'wb') as f:
        pickle.dump(x_comp, f)
    print('模型已保存到缓存')

#四、预测用户电影
if x_comp is not None:
    #获取用户id
    user_id = input('您要向哪位用户推荐电影？请输入用户编号： ')
    user_id = int(user_id) - 1

    #获取该用户评分列表，将该用户评分降序排序
    sortResult = x_comp[int(user_id),:].argsort()[::-1]

    #推荐评分最高10部电影
    idx = 0
    print('为该用户推荐的评分最高的10部电影是：'.center(80,'='))

    #开始推荐
    for i in sortResult:
        print('评分：{:.2f},电影名称：{}'.format(x_comp[int(user_id),i],movies_np[i][1]))
        idx += 1
        if idx == 10:
            break
else:
    print('由于缓冲损坏，程序无法继续，请检查缓存文件')