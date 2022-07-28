import paddle
from dataset import get_data_transforms
from paddle.vision.datasets import ImageFolder
import numpy as np
import random
import os
from paddle.io import DataLoader
from resnet import BN_layer, AttnBottleneck, WideResnet50
from de_resnet import de_resnet18, de_resnet34, de_wide_resnet50_2, de_resnet50
from dataset import MVTecDataset
import argparse
from test import evaluation, show_cam_on_image
from paddle.nn import functional as F
import time
from test import cal_anomaly_map, gaussian_filter, min_max_norm, cvt2heatmap, cv2


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if not p.stop_gradient)


def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    paddle.seed(seed)


def loss_fucntion(a, b):
    # mse_loss = paddle.nn.MSELoss()
    cos_loss = paddle.nn.CosineSimilarity()
    loss = 0
    for item in range(len(a)):
        # print(a[item].shape)
        # print(b[item].shape)
        # loss += 0.1*mse_loss(a[item], b[item])
        loss += paddle.mean(1 - cos_loss(a[item].reshape([a[item].shape[0], -1]),
                                         b[item].reshape([b[item].shape[0], -1])))
    return loss


def loss_concat(a, b):
    mse_loss = paddle.nn.MSELoss()
    cos_loss = paddle.nn.CosineSimilarity()
    loss = 0
    a_map = []
    b_map = []
    size = a[0].shape[-1]
    for item in range(len(a)):
        # loss += mse_loss(a[item], b[item])
        a_map.append(F.interpolate(a[item], size=size, mode='bilinear', align_corners=True))
        b_map.append(F.interpolate(b[item], size=size, mode='bilinear', align_corners=True))
    a_map = paddle.concat(a_map, 1)
    b_map = paddle.concat(b_map, 1)
    loss += paddle.mean(1 - cos_loss(a_map, b_map))
    return loss


def train(_class_, print_steps=5):
    epochs = args.epochs
    learning_rate = args.lr
    batch_size = args.batch_size
    image_size = args.image_size

    data_transform, gt_transform = get_data_transforms(image_size, image_size)
    train_path = os.path.join(args.data_dir, _class_ + '/train')
    test_path = os.path.join(args.data_dir, _class_)
    ckp_path = os.path.join(args.output_dir, 'wres50_' + _class_ + '.pdparams')
    train_data = ImageFolder(root=train_path, transform=data_transform)  # 训练集不需要label,直接用ImageFolder加载
    test_data = MVTecDataset(root=test_path, transform=data_transform, gt_transform=gt_transform, phase="test")
    train_dataloader = paddle.io.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=8)
    test_dataloader = paddle.io.DataLoader(test_data, batch_size=1, shuffle=False)

    encoder = WideResnet50()
    bn = BN_layer(AttnBottleneck, 3)
    encoder.eval()  # 预训练的WideResnet50，训练时不需要梯度更新
    decoder = de_wide_resnet50_2(pretrained=False)

    optimizer = paddle.optimizer.Adam(parameters=list(decoder.parameters()) + list(bn.parameters()),
                                      learning_rate=learning_rate, beta1=0.5)
    global_step = 0
    train_reader_cost = 0.0
    train_run_cost = 0.0
    total_samples = 0
    reader_start = time.time()
    last_step = epochs * len(train_dataloader)
    best = 0
    for epoch in range(epochs):
        bn.train()
        decoder.train()
        loss_list = []
        for img in train_dataloader:
            img = img[0]
            train_reader_cost += time.time() - reader_start
            train_start = time.time()
            global_step += 1
            inputs = encoder(img)
            outputs = decoder(bn(inputs))  # bn(inputs))
            loss = loss_fucntion(inputs, outputs)
            train_run_cost += time.time() - train_start
            total_samples += len(img)
            optimizer.clear_grad()
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())
            if global_step % print_steps == 0:
                print(
                    "global step %d / %d, loss: %f, avg_reader_cost: %.5f sec, avg_batch_cost: %.5f sec, avg_samples: %.5f, ips: %.5f img/sec"
                    % (global_step, last_step, loss.item(), train_reader_cost /
                       print_steps, (train_reader_cost + train_run_cost)
                       / print_steps, total_samples / print_steps,
                       total_samples / (train_reader_cost + train_run_cost)))
                train_reader_cost = 0.0
                train_run_cost = 0.0
                total_samples = 0
                reader_start = time.time()
        if (epoch + 1) % 10 == 0:
            auroc_px, auroc_sp, aupro_px = evaluation(encoder, bn, decoder, test_dataloader)
            print(
                'Class {}, Pixel Auroc:{:.3f}, Sample Auroc{:.3f}, Pixel Aupro{:.3}'.format(_class_, auroc_px, auroc_sp,
                                                                                            aupro_px))
            score = auroc_px + auroc_sp + aupro_px
            if score > best:
                best = score
                paddle.save({'bn': bn.state_dict(),
                             'decoder': decoder.state_dict()}, ckp_path)
        reader_start = time.time()


def eval_model(_class_):
    image_size = 256
    data_transform, gt_transform = get_data_transforms(image_size, image_size)
    test_path = os.path.join(args.data_dir, _class_)
    test_data = MVTecDataset(root=test_path, transform=data_transform, gt_transform=gt_transform, phase="test")
    test_dataloader = paddle.io.DataLoader(test_data, batch_size=1, shuffle=False)  # eval时batch size为1

    encoder = WideResnet50()
    bn = BN_layer(AttnBottleneck, 3)
    encoder.eval()
    decoder = de_wide_resnet50_2(pretrained=False)
    bn.eval()
    decoder.eval()

    ckpt_path = os.path.join('./checkpoints', 'wres50_{}.pdparams'.format(_class_))
    states = paddle.load(ckpt_path)
    bn.set_state_dict(states['bn'])
    decoder.set_state_dict(states['decoder'])

    auroc_px, auroc_sp, aupro_px = evaluation(encoder, bn, decoder, test_dataloader)

    return auroc_px, auroc_sp, aupro_px


def infer(_class_):
    image_size = 256
    data_transform, gt_transform = get_data_transforms(image_size, image_size)
    test_path = os.path.join(args.data_dir, _class_)
    test_data = MVTecDataset(root=test_path, transform=data_transform, gt_transform=gt_transform, phase="test")

    encoder = WideResnet50()
    bn = BN_layer(AttnBottleneck, 3)
    encoder.eval()
    decoder = de_wide_resnet50_2(pretrained=False)
    bn.eval()
    decoder.eval()

    ckpt_path = os.path.join('./checkpoints', 'wres50_{}.pdparams'.format(_class_))
    states = paddle.load(ckpt_path)
    bn.set_state_dict(states['bn'])
    decoder.set_state_dict(states['decoder'])

    img, gt, label, _ = test_data[0]
    img = img.unsqueeze(0)
    inputs = encoder(img)
    outputs = decoder(bn(inputs))

    anomaly_map, amap_list = cal_anomaly_map(inputs, outputs, img.shape[-1], amap_mode='a')
    ano_map = min_max_norm(anomaly_map)
    ano_map = cvt2heatmap(ano_map * 255)
    img = cv2.cvtColor(img.transpose([0, 2, 3, 1]).cpu().numpy()[0] * 255, cv2.COLOR_BGR2RGB)
    img = np.uint8(min_max_norm(img) * 255)
    cv2.imwrite(os.path.join(args.output_dir, 'org.png'), img)
    cv2.imwrite(os.path.join(args.output_dir, 'ad.png'), ano_map)
    print('Outputs saved in {}'.format(args.output_dir))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='mvtec', type=str)
    parser.add_argument('--mode', default='train', type=str)
    parser.add_argument('--cls', default='', type=str)
    parser.add_argument('--output_dir', default='checkpoints', type=str)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--lr', default=0.005, type=float)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--image_size', default=256, type=int)
    parser.add_argument('--print_steps', default=50, type=int)
    parser.add_argument('--device', default='gpu', type=str)
    args = parser.parse_args()

    setup_seed(123)
    item_list = ['carpet', 'bottle', 'hazelnut', 'leather', 'cable', 'capsule', 'grid', 'pill',
                 'transistor', 'metal_nut', 'screw', 'toothbrush', 'zipper', 'tile', 'wood']
    if args.mode == 'train':
        for i in item_list:
            if os.path.exists(os.path.join(args.data_dir, i)):
                train(i, args.print_steps)

    elif args.mode == 'eval':
        if args.cls:
            auroc_px, auroc_sp, aupro_px = eval_model(args.tgt_class)
            print(
                'Class {}, Pixel Auroc:{:.3f}, Sample Auroc{:.3f}, Pixel Aupro{:.3}'.format(args.tgt_class, auroc_px,
                                                                                            auroc_sp, aupro_px))
        else:
            auroc_pxs, auroc_sps, aupro_pxs = [], [], []
            for i in item_list:
                auroc_px, auroc_sp, aupro_px = eval_model(i)
                auroc_pxs.append(auroc_px)
                auroc_sps.append(auroc_sp)
                aupro_pxs.append(aupro_px)
                print(
                    'Class {}, Pixel Auroc:{:.3f}, Sample Auroc{:.3f}, Pixel Aupro{:.3}'.format(i, auroc_px, auroc_sp,
                                                                                                aupro_px))
            auroc_px = np.mean(auroc_pxs)
            auroc_sp = np.mean(auroc_sps)
            aupro_px = np.mean(aupro_pxs)
            print('Pixel Auroc:{:.3f}, Sample Auroc{:.3f}, Pixel Aupro{:.3}'.format(auroc_px, auroc_sp, aupro_px))

    elif args.mode == 'infer':
        assert args.cls
        infer(args.cls)
