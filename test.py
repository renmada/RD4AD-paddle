import paddle
from dataset import get_data_transforms
from paddle.vision.datasets import ImageFolder
import numpy as np
from paddle.io import DataLoader
from resnet import BN_layer, AttnBottleneck, WideResnet50
from de_resnet import de_resnet18, de_resnet50, de_wide_resnet50_2
from dataset import MVTecDataset
from paddle.nn import functional as F
from sklearn.metrics import roc_auc_score
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import auc
from skimage import measure
import pandas as pd
from numpy import ndarray
from statistics import mean
from scipy.ndimage import gaussian_filter
from sklearn import manifold
from matplotlib.ticker import NullFormatter
from scipy.spatial.distance import pdist
import matplotlib
import pickle


def cal_anomaly_map(fs_list, ft_list, out_size=224, amap_mode='mul'):
    if amap_mode == 'mul':
        anomaly_map = np.ones([out_size, out_size])
    else:
        anomaly_map = np.zeros([out_size, out_size])
    if isinstance(out_size, int):
        out_size = [out_size, out_size]
    a_map_list = []
    for i in range(len(ft_list)):
        fs = fs_list[i]
        ft = ft_list[i]
        # fs_norm = F.normalize(fs, p=2)
        # ft_norm = F.normalize(ft, p=2)
        a_map = 1 - F.cosine_similarity(fs, ft)
        a_map = paddle.unsqueeze(a_map, 1)
        a_map = F.interpolate(a_map, size=out_size, mode='bilinear', align_corners=True)
        a_map = a_map[0, 0, :, :].detach().numpy()
        a_map_list.append(a_map)
        if amap_mode == 'mul':
            anomaly_map *= a_map
        else:
            anomaly_map += a_map
    return anomaly_map, a_map_list


def show_cam_on_image(img, anomaly_map):
    # if anomaly_map.shape != img.shape:
    #    anomaly_map = cv2.applyColorMap(np.uint8(anomaly_map), cv2.COLORMAP_JET)
    cam = np.float32(anomaly_map) / 255 + np.float32(img) / 255
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)


def min_max_norm(image):
    a_min, a_max = image.min(), image.max()
    return (image - a_min) / (a_max - a_min)


def cvt2heatmap(gray):
    heatmap = cv2.applyColorMap(np.uint8(gray), cv2.COLORMAP_JET)
    return heatmap


def evaluation(encoder, bn, decoder, dataloader, _class_=None):
    bn.eval()
    decoder.eval()
    gt_list_px = []
    pr_list_px = []
    gt_list_sp = []
    pr_list_sp = []
    aupro_list = []
    with paddle.no_grad():
        for img, gt, label, _ in dataloader:
            inputs = encoder(img)
            outputs = decoder(bn(inputs))
            anomaly_map, _ = cal_anomaly_map(inputs, outputs, img.shape[-1], amap_mode='a')
            anomaly_map = gaussian_filter(anomaly_map, sigma=4)
            gt[gt > 0.5] = 1
            gt[gt <= 0.5] = 0
            if label.cpu().numpy()[0] != 0:
                aupro_list.append(compute_pro(gt.squeeze(0).cpu().numpy().astype(int),
                                              anomaly_map[np.newaxis, :, :]))
            gt_list_px.extend(gt.cpu().numpy().astype(int).ravel())
            pr_list_px.extend(anomaly_map.ravel())
            gt_list_sp.append(np.max(gt.cpu().numpy().astype(int)))
            pr_list_sp.append(np.max(anomaly_map))

        auroc_px = round(roc_auc_score(gt_list_px, pr_list_px), 3)
        auroc_sp = round(roc_auc_score(gt_list_sp, pr_list_sp), 3)
    bn.train()
    decoder.train()
    return auroc_px, auroc_sp, round(np.mean(aupro_list), 3)


def test(_class_):
    device = 'cuda' if paddle.cuda.is_available() else 'cpu'
    print(device)
    print(_class_)

    data_transform, gt_transform = get_data_transforms(256, 256)
    test_path = '../mvtec/' + _class_
    ckp_path = './checkpoints/' + 'rm_1105_wres50_ff_mm_' + _class_ + '.pth'
    test_data = MVTecDataset(root=test_path, transform=data_transform, gt_transform=gt_transform, phase="test")
    test_dataloader = paddle.io.DataLoader(test_data, batch_size=1, shuffle=False)
    encoder = WideResnet50()
    bn = BN_layer(AttnBottleneck, 3)
    encoder.eval()
    decoder = de_wide_resnet50_2(pretrained=False)
    decoder = decoder
    ckp = paddle.load(ckp_path)
    for k, v in list(ckp['bn'].items()):
        if 'memory' in k:
            ckp['bn'].pop(k)
    decoder.set_state_dict(ckp['decoder'])
    bn.set_state_dict(ckp['bn'])
    auroc_px, auroc_sp, aupro_px = evaluation(encoder, bn, decoder, test_dataloader, device, _class_)
    print(_class_, ':', auroc_px, ',', auroc_sp, ',', aupro_px)
    return auroc_px


def compute_pro(masks: ndarray, amaps: ndarray, num_th: int = 200) -> None:
    """Compute the area under the curve of per-region overlaping (PRO) and 0 to 0.3 FPR
    Args:
        category (str): Category of product
        masks (ndarray): All binary masks in test. masks.shape -> (num_test_data, h, w)
        amaps (ndarray): All anomaly maps in test. amaps.shape -> (num_test_data, h, w)
        num_th (int, optional): Number of thresholds
    """

    assert isinstance(amaps, ndarray), "type(amaps) must be ndarray"
    assert isinstance(masks, ndarray), "type(masks) must be ndarray"
    assert amaps.ndim == 3, "amaps.ndim must be 3 (num_test_data, h, w)"
    assert masks.ndim == 3, "masks.ndim must be 3 (num_test_data, h, w)"
    assert amaps.shape == masks.shape, "amaps.shape and masks.shape must be same"
    assert set(masks.flatten()) == {0, 1}, "set(masks.flatten()) must be {0, 1}"
    assert isinstance(num_th, int), "type(num_th) must be int"

    df = pd.DataFrame([], columns=["pro", "fpr", "threshold"])
    binary_amaps = np.zeros_like(amaps, dtype=np.bool)

    min_th = amaps.min()
    max_th = amaps.max()
    delta = (max_th - min_th) / num_th

    for th in np.arange(min_th, max_th, delta):
        binary_amaps[amaps <= th] = 0
        binary_amaps[amaps > th] = 1

        pros = []
        for binary_amap, mask in zip(binary_amaps, masks):
            for region in measure.regionprops(measure.label(mask)):
                axes0_ids = region.coords[:, 0]
                axes1_ids = region.coords[:, 1]
                tp_pixels = binary_amap[axes0_ids, axes1_ids].sum()
                pros.append(tp_pixels / region.area)

        inverse_masks = 1 - masks
        fp_pixels = np.logical_and(inverse_masks, binary_amaps).sum()
        fpr = fp_pixels / inverse_masks.sum()

        df = df.append({"pro": mean(pros), "fpr": fpr, "threshold": th}, ignore_index=True)

    # Normalize FPR from 0 ~ 1 to 0 ~ 0.3
    df = df[df["fpr"] < 0.3]
    df["fpr"] = df["fpr"] / df["fpr"].max()

    pro_auc = auc(df["fpr"], df["pro"])
    return pro_auc


def detection(encoder, bn, decoder, dataloader, device, _class_):
    # _, t_bn = resnet50(pretrained=True)
    bn.load_state_dict(bn.state_dict())
    bn.eval()
    # t_bn
    # t_bn.load_state_dict(bn.state_dict())
    decoder.eval()
    gt_list_sp = []
    prmax_list_sp = []
    prmean_list_sp = []
    with paddle.no_grad():
        for img, label in dataloader:

            img = img
            if img.shape[1] == 1:
                img = img.repeat(1, 3, 1, 1)
            label = label
            inputs = encoder(img)
            outputs = decoder(bn(inputs))
            anomaly_map, _ = cal_anomaly_map(inputs, outputs, img.shape[-1], 'acc')
            anomaly_map = gaussian_filter(anomaly_map, sigma=4)

            gt_list_sp.extend(label.cpu().data.numpy())
            prmax_list_sp.append(np.max(anomaly_map))
            prmean_list_sp.append(np.sum(anomaly_map))  # np.sum(anomaly_map.ravel().argsort()[-1:][::-1]))

        gt_list_sp = np.array(gt_list_sp)
        indx1 = gt_list_sp == _class_
        indx2 = gt_list_sp != _class_
        gt_list_sp[indx1] = 0
        gt_list_sp[indx2] = 1

        auroc_sp_max = round(roc_auc_score(gt_list_sp, prmax_list_sp), 4)
        auroc_sp_mean = round(roc_auc_score(gt_list_sp, prmean_list_sp), 4)
    return auroc_sp_max, auroc_sp_mean
