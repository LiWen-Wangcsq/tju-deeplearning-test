import monai
import torch.nn.functional as F

from einops import rearrange
from monai.metrics import HausdorffDistanceMetric, compute_hausdorff_distance

import torch
import torch.nn as nn
import numpy as np


def hd95(output, target, batch_size=1000):
    batched_distances = []
    for b in range(output.shape[0]):
        pred_pts = torch.nonzero(output[b], as_tuple=False)
        true_pts = torch.nonzero(target[b], as_tuple=False)

        if pred_pts.size(0) == 0 or true_pts.size(0) == 0:
            continue  # Skip this sample if no points are present

        # 分批处理距离计算
        distances = []
        for i in range(0, pred_pts.size(0), batch_size):
            end_i = min(i + batch_size, pred_pts.size(0))
            for j in range(0, true_pts.size(0), batch_size):
                end_j = min(j + batch_size, true_pts.size(0))
                dist_matrix = torch.cdist(pred_pts[i:end_i].float(), true_pts[j:end_j].float())
                distances.append(dist_matrix)

        # 合并所有批次的结果
        if distances:
            all_distances = torch.cat([d.min(dim=1)[0] for d in distances] + [d.min(dim=0)[0] for d in distances])
            hd95 = torch.quantile(all_distances, 0.95).item()
            batched_distances.append(hd95)

    if batched_distances:
        return sum(batched_distances) / len(batched_distances)
    else:
        return float('inf')  # Return inf if no valid distances were calculated


def hd95_multi_class(output, target, class_indices):
    """
    Compute the HD95 for multiple classes separately.
    output: Binary segmentations per class (b, num_classes, d, h, w)
    target: Binary segmentations per class (b, d, h, w)
    class_indices: List of class indices for which to calculate HD95
    """
    hd95_results = {}
    for class_idx in class_indices:
        output_class = (output == class_idx).float()
        target_class = (target == class_idx).float()
        hd95_results[class_idx] = hd95(output_class, target_class)
    return hd95_results





class HD95Loss(nn.Module):
    """
    HD95 Loss: A loss module that calculates the 95th percentile Hausdorff Distance as a loss.
    """

    def __init__(self):
        super(HD95Loss, self).__init__()

    def forward(self, input, target):
        input_bin = (input > 0.5).float()  # Assuming input is probability map
        target_bin = (target > 0.5).float()  # Assuming target is binary mask
        hd95_distance = hd95(input_bin, target_bin)
        return torch.tensor(hd95_distance).mean()  # Returning the average HD95 across the batch




def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0.):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule


def Dice(output, target, eps=1e-3):
    inter = torch.sum(output * target,dim=(1,2,-1)) + eps
    union = torch.sum(output,dim=(1,2,-1)) + torch.sum(target,dim=(1,2,-1)) + eps * 2
    x = 2 * inter / union
    dice = torch.mean(x)
    return dice

def cal_dice(output, target):
    '''
    output: (b, num_class, d, h, w)  target: (b, d, h, w)
    dice1(ET):label4
    dice2(TC):label1 + label4
    dice3(WT): label1 + label2 + label4
    注,这里的label4已经被替换为3
    '''
    output = torch.argmax(output,dim=1)
    dice1 = Dice((output == 3).float(), (target == 3).float())
    dice2 = Dice(((output == 1) | (output == 3)).float(), ((target == 1) | (target == 3)).float())
    dice3 = Dice((output != 0).float(), (target != 0).float())

    return dice1, dice2, dice3




# def cal_hd95(output, target):
#     '''
#     output: (b, num_class, d, h, w)  target: (b, d, h, w)
#     hd95_1(ET): label 4 (already replaced by 3)
#     hd95_2(TC): label 1 + label 4 (already replaced by 3)
#     hd95_3(WT): label 1 + label 2 + label 4 (already replaced by 3)
#     '''
#
#     # 将输出概率转换为类别标签
#     output = torch.argmax(output, dim=1)
#
#     # 计算ET的HD95
#     et_output = (output == 3).float()
#     et_target = (target == 3).float()
#     hd95_1 = compute_hausdorff_distance(et_output, et_target, percentile=95)
#
#     # 计算TC的HD95
#     tc_output = ((output == 1) | (output == 3)).float()
#     tc_target = ((target == 1) | (target == 3)).float()
#     hd95_2 = compute_hausdorff_distance(tc_output, tc_target, percentile=95)
#
#     # 计算WT的HD95
#     wt_output = (output != 0).float()
#     wt_target = (target != 0).float()
#     hd95_3 = compute_hausdorff_distance(wt_output, wt_target, percentile=95)
#
#     # 将结果转化为Python标量并返回
#     return hd95_1.mean().item(), hd95_2.mean().item(), hd95_3.mean().item()




def cal__hd95(output, target):
    '''
    output: (b, num_class, d, h, w)  target: (b, d, h, w)
    hd95_1(ET): label 4
    hd95_2(TC): label 1 + label 4
    hd95_3(WT): label 1 + label 2 + label 4
    注,这里的label4已经被替换为3
    '''
    output = torch.argmax(output, dim=1)
    hd_metric = HausdorffDistanceMetric(percentile=95, include_background=False,
            reduction=monai.utils.MetricReduction.MEAN_BATCH,
            get_not_nans=False,)

    def compute__hd(output, target):
        for i in range(target.shape[1]):  # 假设第一个维度是类别维
            if np.any(target[i]) and np.any(target[i]):
               hd = hd_metric(output, target)
            return torch.tensor([torch.mean(hd).item()])  # 返回一个包含平均HD95的张量


    # 计算 ET 的 HD95
    et_output = (output == 3)
    et_target = (target == 3)
    hd95_1 = compute__hd(et_output, et_target)

    # 计算 TC 的 HD95
    tc_output = (output == 1) | (output == 3)
    tc_target = (target == 1) | (target == 3)
    hd95_2 = compute__hd(tc_output, tc_target)

    # 计算 WT 的 HD95
    wt_output = (output != 0)
    wt_target = (target != 0)
    hd95_3 = compute__hd(wt_output, wt_target)

    # 计算并返回每个batch的平均值
    return torch.mean(hd95_1), torch.mean(hd95_2), torch.mean(hd95_3)


def compute_hd(output, target):
    hd_metric = HausdorffDistanceMetric(percentile=95, include_background=True,
                                        reduction=monai.utils.MetricReduction.NONE,
                                        get_not_nans=False)
    hd = hd_metric(output.unsqueeze(0), target.unsqueeze(0))
    return hd.clone().detach()  # 使用 clone().detach() 替代 torch.tensor()
# def compute_hd(output, target):
#     # 确保数据类型为torch.bool，因为torch.any()需要bool类型
#     hd_metric = HausdorffDistanceMetric(percentile=95, include_background=False,
#                                         reduction=monai.utils.MetricReduction.NONE,
#                                         get_not_nans=False)
#     target_bool = target.bool()
#     output_bool = output.bool()
#
#     if torch.any(target_bool) and torch.any(output_bool):
#         hd = hd_metric(output_bool, target_bool)
#         return torch.tensor([torch.mean(hd).item()])  # 返回一个包含平均HD95的张量
#     else:
#         return torch.tensor([float('nan')])  # 如果没有有效的数据来计算HD95，返回NaN


def cal_hd95(output, target):
    '''
    output: (b, num_class, d, h, w)  target: (b, d, h, w)
    hd95_1(ET): label 4
    hd95_2(TC): label 1 + label 4
    hd95_3(WT): label 1 + label 2 + label 4
    注,这里的label4已经被替换为3
    '''
    output = torch.argmax(output, dim=1)  # assuming output is (b, num_class, d, h, w)

    # 计算 ET 的 HD95
    et_output = (output == 3).float()  # Assuming label 3 is ET
    et_target = (target == 3).float()
    hd95_1 = compute_hd(et_output, et_target)

    # 计算 TC 的 HD95
    tc_output = ((output == 1) | (output == 3)).float()  # Assuming labels 1 and 3 are TC
    tc_target = ((target == 1) | (target == 3)).float()
    hd95_2 = compute_hd(tc_output, tc_target)

    # 计算 WT 的 HD95
    wt_output = (output != 0).float()  # Assuming label 0 is background, others are WT
    wt_target = (target != 0).float()
    hd95_3 = compute_hd(wt_output, wt_target)

    # 计算并返回每个batch的平均值
    return torch.mean(hd95_1), torch.mean(hd95_2), torch.mean(hd95_3)
    #return hd95_1,hd95_2,hd95_3


def cal_dice_and_hd95(output, target):
    '''
    Calculate both Dice and HD95 for ET, TC, and WT.
    output: (b, num_class, d, h, w)  target: (b, d, h, w)
    ET (label4, now 3), TC (label1 + label4), WT (label1 + label2 + label4)
    Note: label4 has been replaced by 3.
    '''
    output = torch.argmax(output, dim=1)
    # Calculate Dice scores
    dice1 = Dice((output == 3).float(), (target == 3).float())
    dice2 = Dice(((output == 1) | (output == 3)).float(), ((target == 1) | (target == 3)).float())
    dice3 = Dice((output != 0).float(), (target != 0).float())

    # Calculate HD95 metrics
    hd95_et = hd95((output == 3).float(), (target == 3).float())
    hd95_tc = hd95(((output == 1) | (output == 3)).float(), ((target == 1) | (target == 3)).float())
    hd95_wt = hd95((output != 0).float(), (target != 0).float())

    # Combine results
    dice_scores = (dice1, dice2, dice3)
    hd95_scores = {'HD95_ET': hd95_et, 'HD95_TC': hd95_tc, 'HD95_WT': hd95_wt}

    return dice1,dice2,dice3, hd95_scores


class Loss(nn.Module):
    def __init__(self, n_classes, weight=None, alpha=0.5):
        "dice_loss_plus_cetr_weighted"
        super(Loss, self).__init__()
        self.n_classes = n_classes
        self.weight = weight.cuda()
        # self.weight = weight
        self.alpha = alpha

    def forward(self, input, target):
        # print(torch.unique(target))
        smooth = 0.01

        input1 = F.softmax(input, dim=1)
        target1 = F.one_hot(target,self.n_classes)
        input1 = rearrange(input1,'b n h w s -> b n (h w s)')
        target1 = rearrange(target1,'b h w s n -> b n (h w s)')

        input1 = input1[:, 1:, :]
        target1 = target1[:, 1:, :].float()

        # 以batch为单位计算loss和dice_loss，据说训练更稳定，那我试试
        inter = torch.sum(input1 * target1)
        union = torch.sum(input1) + torch.sum(target1) + smooth
        dice = 2.0 * inter / union

        loss = F.cross_entropy(input,target, weight=self.weight)

        total_loss = (1 - self.alpha) * loss + (1 - dice) * self.alpha

        return total_loss


if __name__ == '__main__':
    torch.manual_seed(3)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    losser = Loss(n_classes=4, weight=torch.tensor([0.2, 0.3, 0.25, 0.25])).to(device)
    x = torch.randn((2, 4, 16, 16, 16)).to(device)
    y = torch.randint(0, 4, (2, 16, 16, 16)).to(device)
    print(losser(x, y))
    print(cal_dice_and_hd95(x, y))
