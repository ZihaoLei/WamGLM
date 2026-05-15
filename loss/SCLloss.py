import torch
import torch.nn as nn
import torch.nn.functional as F


class SCLoss(nn.Module):
    def __init__(self, temperature=0.1):
        """
        初始化监督对比损失。

        参数:
        - temperature: 温度参数，控制相似性的缩放程度。
        """
        super(SCLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        """
        计算监督对比损失。

        参数:
        - features: 输入特征张量，形状为 (batch_size, feature_dim)。
        - labels: 标签张量，形状为 (batch_size,)。

        返回:
        - loss: 监督对比损失值。
        """
        # 归一化特征向量
        # features = F.normalize(features, dim=1)
        # print(features.shape)

        # 计算相似度矩阵 (余弦相似度)
        similarity_matrix = torch.matmul(features, features.T) / self.temperature
        # similarity_matrix = features.mm(features.T) / self.temperature

        # 创建一个掩码，标记相同类别的样本对
        labels = labels.contiguous().view(-1, 1)
        labels = torch.cat([labels.repeat(2, 1)], dim=0)
        mask = torch.eq(labels, labels.T).float()  # 形状为 (batch_size, batch_size)
        self_mask = 1 - torch.eye(mask.size(0), device=mask.device)

        # 去掉对角线上的自身比较
        mask = mask * self_mask

        # 保持计算稳定
        similarity_max, _ = torch.max(similarity_matrix, dim=1, keepdim=True)
        similarity_matrix = similarity_matrix - similarity_max.detach()

        # 计算分母：所有正样本和负样本的指数和
        exp_similarities = torch.exp(similarity_matrix) * self_mask
        log_prob = similarity_matrix - torch.log(exp_similarities.sum(dim=1, keepdim=True))

        # 只保留正样本对的 log_prob
        positive_log_prob = (mask * log_prob).sum(dim=1) / mask.sum(dim=1)

        # 计算最终的损失
        loss = -positive_log_prob.mean()

        return loss