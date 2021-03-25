import numpy as np
import torch
from torch import nn
from torchvision.models.resnet import resnet18
from torch.utils.data import DataLoader

from .dataset import NUM_BBOX, CLASSES, VOC2012


class Loss_yolov1(nn.Module):
    def __init__(self):
        super(Loss_yolov1, self).__init__()

    def forward(self, pred, labels):
        """
        :param pred: (batchsize,30,7,7)的网络输出数据
        :param labels: (batchsize,30,7,7)的样本标签数据
        :return: 当前批次样本的平均损失
        """
        num_gridx, num_gridy = labels.size()[-2:]  # 划分网格数量
        num_b = 2  # 每个网格的bbox数量
        num_cls = 20  # 类别数量
        noobj_confi_loss = 0.  # 不含目标的网格损失(只有置信度损失)
        coor_loss = 0.  # 含有目标的bbox的坐标损失
        obj_confi_loss = 0.  # 含有目标的bbox的置信度损失
        class_loss = 0.  # 含有目标的网格的类别损失
        n_batch = labels.size()[0]  # batchsize的大小

        # 可以考虑用矩阵运算进行优化，提高速度，为了准确起见，这里还是用循环
        for i in range(n_batch):  # batchsize循环
            for n in range(7):  # x方向网格循环
                for m in range(7):  # y方向网格循环
                    if labels[i, 4, m, n] == 1:  # 如果包含物体
                        # 将数据(px,py,w,h)转换为(x1,y1,x2,y2)
                        # 先将px,py转换为cx,cy，即相对网格的位置转换为标准化后实际的bbox中心位置cx,xy
                        # 然后再利用(cx-w/2,cy-h/2,cx+w/2,cy+h/2)转换为xyxy形式，用于计算iou
                        bbox1_pred_xyxy = ((pred[i, 0, m, n] + m) / num_gridx - pred[i, 2, m, n] / 2,
                                           (pred[i, 1, m, n] + n) / num_gridy - pred[i, 3, m, n] / 2,
                                           (pred[i, 0, m, n] + m) / num_gridx + pred[i, 2, m, n] / 2,
                                           (pred[i, 1, m, n] + n) / num_gridy + pred[i, 3, m, n] / 2)
                        bbox2_pred_xyxy = ((pred[i, 5, m, n] + m) / num_gridx - pred[i, 7, m, n] / 2,
                                           (pred[i, 6, m, n] + n) / num_gridy - pred[i, 8, m, n] / 2,
                                           (pred[i, 5, m, n] + m) / num_gridx + pred[i, 7, m, n] / 2,
                                           (pred[i, 6, m, n] + n) / num_gridy + pred[i, 8, m, n] / 2)
                        bbox_gt_xyxy = ((labels[i, 0, m, n] + m) / num_gridx - labels[i, 2, m, n] / 2,
                                        (labels[i, 1, m, n] + n) / num_gridy - labels[i, 3, m, n] / 2,
                                        (labels[i, 0, m, n] + m) / num_gridx + labels[i, 2, m, n] / 2,
                                        (labels[i, 1, m, n] + n) / num_gridy + labels[i, 3, m, n] / 2)
                        iou1 = calculate_iou(bbox1_pred_xyxy, bbox_gt_xyxy)
                        iou2 = calculate_iou(bbox2_pred_xyxy, bbox_gt_xyxy)
                        # 选择iou大的bbox作为负责物体
                        if iou1 >= iou2:
                            coor_loss = coor_loss + 5 * (torch.sum((pred[i, 0:2, m, n] - labels[i, 0:2, m, n]) ** 2) \
                                                         + torch.sum(
                                        (pred[i, 2:4, m, n].sqrt() - labels[i, 2:4, m, n].sqrt()) ** 2))
                            obj_confi_loss = obj_confi_loss + (pred[i, 4, m, n] - iou1) ** 2
                            # iou比较小的bbox不负责预测物体，因此confidence loss算在noobj中，注意，对于标签的置信度应该是iou2
                            noobj_confi_loss = noobj_confi_loss + 0.5 * ((pred[i, 9, m, n] - iou2) ** 2)
                        else:
                            coor_loss = coor_loss + 5 * (torch.sum((pred[i, 5:7, m, n] - labels[i, 5:7, m, n]) ** 2) \
                                                         + torch.sum(
                                        (pred[i, 7:9, m, n].sqrt() - labels[i, 7:9, m, n].sqrt()) ** 2))
                            obj_confi_loss = obj_confi_loss + (pred[i, 9, m, n] - iou2) ** 2
                            # iou比较小的bbox不负责预测物体，因此confidence loss算在noobj中,注意，对于标签的置信度应该是iou1
                            noobj_confi_loss = noobj_confi_loss + 0.5 * ((pred[i, 4, m, n] - iou1) ** 2)
                        class_loss = class_loss + torch.sum((pred[i, 10:, m, n] - labels[i, 10:, m, n]) ** 2)
                    else:  # 如果不包含物体
                        noobj_confi_loss = noobj_confi_loss + 0.5 * torch.sum(pred[i, [4, 9], m, n] ** 2)

        loss = coor_loss + obj_confi_loss + noobj_confi_loss + class_loss
        # 此处可以写代码验证一下loss的大致计算是否正确，这个要验证起来比较麻烦，比较简洁的办法是，将输入的pred置为全1矩阵，再进行误差检查，会直观很多。
        return loss / n_batch


def calculate_iou(bbox1, bbox2):
    """计算bbox1=(x1,y1,x2,y2)和bbox2=(x3,y3,x4,y4)两个bbox的iou"""
    intersect_bbox = [0., 0., 0., 0.]  # bbox1和bbox2的交集
    if bbox1[2] < bbox2[0] or bbox1[0] > bbox2[2] or bbox1[3] < bbox2[1] or bbox1[1] > bbox2[3]:
        pass
    else:
        intersect_bbox[0] = max(bbox1[0], bbox2[0])
        intersect_bbox[1] = max(bbox1[1], bbox2[1])
        intersect_bbox[2] = min(bbox1[2], bbox2[2])
        intersect_bbox[3] = min(bbox1[3], bbox2[3])

    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])  # bbox1面积
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])  # bbox2面积
    area_intersect = (intersect_bbox[2] - intersect_bbox[0]) * (intersect_bbox[3] - intersect_bbox[1])  # 交集面积
    # print(bbox1,bbox2)
    # print(intersect_bbox)
    # input()

    if area_intersect > 0:
        return area_intersect / (area1 + area2 - area_intersect)  # 计算iou
    else:
        return 0


class YOLO_v1(nn.module):
    def __init__(self):
        super(YOLO_v1, self).__init__()
        resnet = resnet18(pretrained=True)  # 调用torchvision里的resnet34预训练模型
        resnet_out_channel = resnet.fc.in_features  # 记录resnet全连接层之前的网络输出通道数，方便连入后续卷积网络中
        self.resnet = nn.Sequential(*list(resnet.children())[:-2])  # 去除resnet的最后两层
        # 以下是YOLOv1的最后四个卷积层
        self.Conv_layers = nn.Sequential(
            nn.Conv2d(resnet_out_channel, 1024, 3, padding=1),
            nn.BatchNorm2d(1024),  # 为了加快训练，这里增加了BN层，原论文里YOLOv1是没有的
            nn.LeakyReLU(),
            nn.Conv2d(1024, 1024, 3, stride=2, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(),
            nn.Conv2d(1024, 1024, 3, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(),
            nn.Conv2d(1024, 1024, 3, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(),
        )
        # 以下是YOLOv1的最后2个全连接层
        self.Conn_layers = nn.Sequential(
            nn.Linear(7 * 7 * 1024, 4096),
            nn.LeakyReLU(),
            nn.Linear(4096, 7 * 7 * 30),
            nn.Sigmoid()  # 增加sigmoid函数是为了将输出全部映射到(0,1)之间，因为如果出现负数或太大的数，后续计算loss会很麻烦
        )

    def forward(self, input_):
        input_ = self.resnet(input_)
        input_ = self.Conv_layers(input_)
        input_ = input_.view(input_.size()[0], -1)
        input_ = self.Conn_layers(input_)
        return input_.reshape(-1, (5 * NUM_BBOX + len(CLASSES)), 7, 7)  # 记住最后要reshape一下输出数据


def demo():
    epoch = 50
    batchsize = 5
    lr = 0.01

    trainData = VOC2012()
    trainDataLoader = DataLoader(VOC2012(is_train=True), batch_size=batchsize, shuffle=True)

    model = YOLO_v1().cuda()
    # model.children()里是按模块(Sequential)提取的子模块，而不是具体到每个层，具体可以参见pytorch帮助文档
    # 冻结resnet34特征提取层，特征提取层不参与参数更新
    for layer in model.children():
        layer.requires_grad = False
        break
    criterion = Loss_yolov1()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)

    for e in range(epoch):
        model.train()
        yl = torch.Tensor([0]).cuda()
        for i, (inputs, labels) in enumerate(trainDataLoader):
            inputs = inputs.cuda()
            labels = labels.float().cuda()
            pred = model(inputs)
            loss = criterion(pred, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print("Epoch %d/%d| Step %d/%d| Loss: %.2f" % (e, epoch, i, len(trainData) // batchsize, loss))
            yl = yl + loss
        if (e + 1) % 10 == 0:
            torch.save(model, "./models_pkl/YOLOv1_epoch" + str(e + 1) + ".pkl")
            # compute_val_map(model)
