import torch
import numpy as np
from sklearn.neighbors import (
    NearestNeighbors,
    LocalOutlierFactor,
)
from sklearn.metrics import DistanceMetric


class EmbeddingDetector():
    def __init__(self, train_embs):
        '''
        Args:
            train_embs ({mid: array}): all the embeddings of the training set
        '''
        self.mean_emb_per_ = {mid: {} for mid in train_embs.keys()}
        self.maha_dist = {mid: {} for mid in train_embs.keys()}
        self.clf = {mid: {} for mid in train_embs.keys()}
        self.lof = {mid: {} for mid in train_embs.keys()}
        self.mean_emb_per_cos = {mid: {} for mid in train_embs.keys()}

        all_embs = None
        for mid in train_embs.keys():
            if all_embs is None:
                all_embs = train_embs[mid]
            else:
                all_embs = np.vstack([all_embs, train_embs[mid]])
        #将所有训练集的嵌入向量合并为一个大的数组 all_embs。

        # maha
        #计算所有嵌入的平均值和协方差矩阵。初始化马氏距离计算器，用协方差矩阵作为参数。
        self.mean_emb_per_ = np.mean(all_embs, axis=0)
        cov = np.cov(all_embs, rowvar=False)
        if np.isnan(cov).sum() > 0:
            raise ValueError("there is nan in the cov of train_embs")
        self.maha_dist = DistanceMetric.get_metric('mahalanobis', V=cov)


        # knn
        #使用余弦距离初始化 k-NN，这里选择的邻居数量为2。使用所有嵌入数据训练 k-NN 模型。
        self.clf = NearestNeighbors(n_neighbors=2, metric='cosine')  # metric='mahalanobis'
        self.clf.fit(all_embs)



        # lof
        #使用余弦距离初始化 LOF，设置邻居数量为4，污染率非常低，以便于捕捉异常值。使用所有嵌入数据训练 LOF 模型。
        self.lof = LocalOutlierFactor(n_neighbors=4,
                                      contamination=1e-6,
                                      metric='cosine',
                                      novelty=True)
        self.lof.fit(all_embs)


        # cos
        #将平均嵌入向量转换为 PyTorch 张量，以便用于计算余弦相似度。
        self.mean_emb_per_cos = torch.from_numpy(self.mean_emb_per_)

    def delnan(self, mat): #检查数组中是否有 NaN 值，并将它们替换为浮点数的最大值。这可以防止计算错误并保持结果的稳定性。
        if np.isnan(mat).sum() > 0:
            mat[np.isnan(mat)] = np.finfo('float32').max
        return mat

    '''
    maha_score: 计算测试嵌入与平均嵌入之间的马氏距离。
    knn_score: 计算测试嵌入与最近邻居的距离之和。
    lof_score: 使用 LOF 模型计算测试嵌入的异常分数（负值）。
    cos_score: 计算测试嵌入与平均嵌入之间的余弦距离。
    '''
    def maha_score(self, test_embs):
        score = self.maha_dist.pairwise([self.mean_emb_per_], test_embs)[0]
        score = self.delnan(score)
        return score

    def knn_score(self, test_embs):
        score = self.clf.kneighbors(test_embs)[0].sum(-1)
        score = self.delnan(score)
        return score

    def lof_score(self, test_embs):
        score = - self.lof.score_samples(test_embs)[0].sum(-1)
        score = self.delnan(score)
        return score

    def cos_score(self, test_embs):
        test_embs = torch.from_numpy(test_embs)
        refer = self.mean_emb_per_cos.repeat(test_embs.shape[0], 1)
        score = 1 - torch.cosine_similarity(test_embs, refer).numpy()
        score = self.delnan(score)
        return score
