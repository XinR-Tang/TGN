import argparse
import math
import random
import os

import numpy as np
import torch
from torch import nn, autograd, optim
from torch.nn import functional as F
from torch.utils import data
import torch.distributed as dist
from torchvision import transforms, utils
from tqdm import tqdm
import clip
from test_consistency import get_clip_loss

try:
    import wandb

except ImportError:
    wandb = None

from dataset import MultiResolutionDataset, C1SatelliteDataset, DiorDataset
from distributed import (
    get_rank,
    synchronize,
    reduce_loss_dict,
    reduce_sum,
    get_world_size,
)
from op import conv2d_gradfix
from non_leaking import augment, AdaptiveAugment
from resnet50 import ResNet, Bottleneck

import warnings

warnings.filterwarnings("ignore")


class NoiseFeatureFusion(nn.Module):
    def __init__(self, feature_dim, noise_dim):
        super(NoiseFeatureFusion, self).__init__()
        self.fc = nn.Linear(feature_dim, noise_dim)

    def forward(self, gan_feature, noise):
        # print("noise",noise.shape)
        batch_size = noise.size(0)  # 512
        # print('batch_size', batch_size)

        # 调整 gan_feature 的 batch size
        gan_feature = gan_feature[:batch_size]  # gan_feature1 torch.Size([16, 1024, 16, 16])
        # print("gan_feature1",gan_feature.shape)

        # 展平 gan_feature
        gan_feature = gan_feature.view(batch_size, -1)  # gan_feature2 torch.Size([512, 8192])
        # print("gan_feature2",gan_feature.shape)

        # 通过全连接层降维
        gan_feature = self.fc(gan_feature)

        # 融合 noise 和处理后的 gan_feature
        fused_noise = 0.8 * noise + 0.2 * gan_feature
        return fused_noise


def data_sampler(dataset, shuffle, distributed):
    if distributed:
        return data.distributed.DistributedSampler(dataset, shuffle=shuffle)

    if shuffle:
        return data.RandomSampler(dataset)

    else:
        return data.SequentialSampler(dataset)


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)


def sample_data(loader):
    while True:
        for batch in loader:
            yield batch


def d_logistic_loss(real_pred, fake_pred):
    real_loss = F.softplus(-real_pred)
    fake_loss = F.softplus(fake_pred)

    return real_loss.mean() + fake_loss.mean()


def d_r1_loss(real_pred, real_img):
    with conv2d_gradfix.no_weight_gradients():
        grad_real, = autograd.grad(
            outputs=real_pred.sum(), inputs=real_img, create_graph=True
        )
    grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()

    return grad_penalty


def g_nonsaturating_loss(fake_pred):
    loss = F.softplus(-fake_pred).mean()

    return loss


def g_path_regularize(fake_img, latents, mean_path_length, decay=0.01):
    noise = torch.randn_like(fake_img) / math.sqrt(
        fake_img.shape[2] * fake_img.shape[3]
    )
    grad, = autograd.grad(
        outputs=(fake_img * noise).sum(), inputs=latents, create_graph=True
    )
    path_lengths = torch.sqrt(grad.pow(2).sum(2).mean(1))

    path_mean = mean_path_length + decay * (path_lengths.mean() - mean_path_length)

    path_penalty = (path_lengths - path_mean).pow(2).mean()

    return path_penalty, path_mean.detach(), path_lengths


def make_noise(batch, latent_dim, n_noise, device, gan_feature, fusion_module):
    if n_noise == 1:
        noise = torch.randn(batch, latent_dim, device=device)
    else:
        noise = torch.randn(n_noise, batch, latent_dim, device=device).unbind(0)
        # for noise in noise:
        #     print(noise.shape)

    # 使用融合模块
    if fusion_module is not None:
        noise = [fusion_module(gan_feature, n) for n in noise]

    return noise


def mixing_noise(batch, latent_dim, prob, device, gan_feature, fusion_module):
    if prob > 0:
        return make_noise(batch, latent_dim, 2, device, gan_feature, fusion_module)

    else:
        return [make_noise(batch, latent_dim, 1, device, gan_feature, fusion_module)]


def set_grad_none(model, targets):
    for n, p in model.named_parameters():
        if n in targets:
            p.grad = None


def train(args, loader, generator, discriminator, g_optim, d_optim, g_ema, device, Rmodel, fusion_module):
    descriptions = {
        "0": "baseball field",
        "1": "ship",
        "2": "storage tank",
        "3": "tennis court",
        "4": "vehicle",
        "5": "windmill",
        "6": "airport",
        "7": "basketball court",
        "8": "bridge",
        "9": "dam",
        "10": "expressway-toll-station",
        "11": "overpass",
    }
    des = list(descriptions.values())
    text_descriptions = [f"A photo of a " + x for x in des]
    text = clip.tokenize(text_descriptions).to(device)


    loader = sample_data(loader)
    loss_fn = nn.CrossEntropyLoss()

    pbar = range(args.iter)

    if get_rank() == 0:  # 是否是主进程
        """
        initial=args.start_iter 设置进度条的初始值。这意味着进度条将从 args.start_iter 开始，而不是从0开始。
        dynamic_ncols=True 使进度条能够动态地调整其长度以适应控制台的宽度。
        smoothing=0.01 设定进度条更新的平滑度，这个值越小，进度条更新越平滑，变化越细腻。
        """
        pbar = tqdm(pbar, initial=args.start_iter, dynamic_ncols=True, smoothing=0.01)

    mean_path_length = 0

    d_loss_val = 0
    r1_loss = torch.tensor(0.0, device=device)
    g_loss_val = 0
    path_loss = torch.tensor(0.0, device=device)
    path_lengths = torch.tensor(0.0, device=device)
    mean_path_length_avg = 0
    loss_dict = {}

    if args.distributed:
        g_module = generator.module
        d_module = discriminator.module

    else:
        g_module = generator
        d_module = discriminator

    accum = 0.5 ** (32 / (10 * 1000))
    ada_aug_p = args.augment_p if args.augment_p > 0 else 0.0
    r_t_stat = 0

    if args.augment and args.augment_p == 0:
        ada_augment = AdaptiveAugment(args.ada_target, args.ada_length, 8, device)

    sample_z = torch.randn(args.n_sample, args.latent, device=device)

    for idx in pbar:  # 迭代次数
        i = idx + args.start_iter

        if i > args.iter:
            print("Done!")

            break

        real_img, label = next(loader)
        real_img = real_img.to(device)
        label = label.to(device)
        # print(torch.min(real_img[0,:,:]), torch.max(real_img[0,:,:]))
        # print(torch.min(real_img[1,:,:]), torch.max(real_img[1,:,:]))
        # print(torch.min(real_img[2,:,:]), torch.max(real_img[2,:,:]))
        # print(torch.std(real_img[0,:,:]), torch.std(real_img[1,:,:]), torch.std(real_img[0,:,:]))
        # print(torch.mean(real_img[0,:,:]), torch.mean(real_img[1,:,:]), torch.mean(real_img[1,:,:]))

        # print("in next(loader)")
        # print(torch.min(real_img), torch.max(real_img))

        requires_grad(generator, False)
        requires_grad(discriminator, True)

        _, gan_feature = Rmodel(real_img)  # torch.Size([16, 1024, 16, 16])

        noise = mixing_noise(args.batch, args.latent, args.mixing, device, gan_feature, fusion_module)

        fake_img, _ = generator(noise, label)

        # print("fake img:")
        # print(torch.min(fake_img), torch.max(fake_img))

        if args.augment:
            real_img_aug, _ = augment(real_img, ada_aug_p)
            fake_img, _ = augment(fake_img, ada_aug_p)

        else:
            real_img_aug = real_img

        fake_pred = discriminator(fake_img, label)
        real_pred = discriminator(real_img_aug, label)
        d_loss = d_logistic_loss(real_pred, fake_pred)

        loss_dict["d"] = d_loss
        loss_dict["real_score"] = real_pred.mean()
        loss_dict["fake_score"] = fake_pred.mean()

        discriminator.zero_grad()
        d_loss.backward(retain_graph=True)
        d_optim.step()

        if args.augment and args.augment_p == 0:
            ada_aug_p = ada_augment.tune(real_pred)
            r_t_stat = ada_augment.r_t_stat

        d_regularize = i % args.d_reg_every == 0

        if d_regularize:
            real_img.requires_grad = True

            if args.augment:
                real_img_aug, _ = augment(real_img, ada_aug_p)

            else:
                real_img_aug = real_img

            real_pred = discriminator(real_img_aug, label)
            r1_loss = d_r1_loss(real_pred, real_img)

            discriminator.zero_grad()
            (args.r1 / 2 * r1_loss * args.d_reg_every + 0 * real_pred[0]).backward(retain_graph=True)

            d_optim.step()

        loss_dict["r1"] = r1_loss

        requires_grad(generator, True)
        requires_grad(discriminator, False)

        noise = mixing_noise(args.batch, args.latent, args.mixing, device, gan_feature, fusion_module)
        #         print(noise.shape)

        fake_img, _ = generator(noise, label)

        #         #print("fake img:")
        #         #print(torch.min(fake_img), torch.max(fake_img))

        if args.augment:
            fake_img, _ = augment(fake_img, ada_aug_p)

        predict, _ = Rmodel(fake_img)  # torch.Size([16, 1024, 16, 16])

        loss_cl = loss_fn(predict, label)
        clip_loss = get_clip_loss(fake_img, text, label, device)

        # loss_dict["loss_cl"] = loss_cl

        fake_pred = discriminator(fake_img, label)
        g_loss = g_nonsaturating_loss(fake_pred)

        loss_dict["g"] = g_loss + loss_cl + clip_loss

        generator.zero_grad()
        g_loss.backward(retain_graph=True)
        g_optim.step()

        g_regularize = i % args.g_reg_every == 0

        if g_regularize:
            path_batch_size = max(1, args.batch // args.path_batch_shrink)
            noise = mixing_noise(path_batch_size, args.latent, args.mixing, device, gan_feature, fusion_module)
            fake_img, latents = generator(noise, label, return_latents=True)

            path_loss, mean_path_length, path_lengths = g_path_regularize(
                fake_img, latents, mean_path_length
            )

            generator.zero_grad()
            weighted_path_loss = args.path_regularize * args.g_reg_every * path_loss

            if args.path_batch_shrink:
                weighted_path_loss += 0 * fake_img[0, 0, 0, 0]

            weighted_path_loss.backward()

            g_optim.step()

            mean_path_length_avg = (
                    reduce_sum(mean_path_length).item() / get_world_size()
            )

        loss_dict["path"] = path_loss
        loss_dict["path_length"] = path_lengths.mean()

        accumulate(g_ema, g_module, accum)

        loss_reduced = reduce_loss_dict(loss_dict)

        # c_loss_val = loss_reduced["loss_cl"].mean().item()
        d_loss_val = loss_reduced["d"].mean().item()
        g_loss_val = loss_reduced["g"].mean().item()
        r1_val = loss_reduced["r1"].mean().item()
        path_loss_val = loss_reduced["path"].mean().item()
        real_score_val = loss_reduced["real_score"].mean().item()
        fake_score_val = loss_reduced["fake_score"].mean().item()
        path_length_val = loss_reduced["path_length"].mean().item()

        if get_rank() == 0:
            pbar.set_description(
                (
                    f"d: {d_loss_val:.4f}; g: {g_loss_val:.4f}; r1: {r1_val:.4f}; "
                    f"path: {path_loss_val:.4f}; mean path: {mean_path_length_avg:.4f}; "
                    f"augment: {ada_aug_p:.4f}"
                )
            )

            if wandb and args.wandb:
                wandb.log(
                    {
                        "Generator": g_loss_val,
                        "Discriminator": d_loss_val,
                        "Augment": ada_aug_p,
                        "Rt": r_t_stat,
                        "R1": r1_val,
                        "Path Length Regularization": path_loss_val,
                        "Mean Path Length": mean_path_length,
                        "Real Score": real_score_val,
                        "Fake Score": fake_score_val,
                        "Path Length": path_length_val,
                    }
                )

            if i % 100 == 0:
                with torch.no_grad():
                    g_ema.eval()
                    sample, _ = g_ema(sample_z, label)
                    print("visualization:")
                    print(torch.min(sample), torch.max(sample))
                    utils.save_image(
                        sample,
                        f"sample/{str(i).zfill(6)}.png",
                        nrow=int(args.n_sample ** 0.5),
                        normalize=True,
                        range=(-1, 1),
                    )

            if i % 5000 == 0:
                torch.save(
                    {
                        "g": g_module.state_dict(),
                        "d": d_module.state_dict(),
                        "g_ema": g_ema.state_dict(),
                        "g_optim": g_optim.state_dict(),
                        "d_optim": d_optim.state_dict(),
                        "args": args,
                        "ada_aug_p": ada_aug_p,
                    },
                    f"checkpoint/{str(i).zfill(6)}.pt",
                )


if __name__ == "__main__":
    device = "cuda"

    parser = argparse.ArgumentParser(description="CDTG trainer")

    # parser.add_argument("path", type=str, help="path to the lmdb dataset")
    parser.add_argument('--arch', type=str, default='stylegan2', help='model architectures')
    parser.add_argument(
        "--iter", type=int, default=200000, help="total training iterations"
    )
    parser.add_argument(
        "--batch", type=int, default=8, help="batch sizes for each gpus"
    )
    parser.add_argument(
        "--n_sample",
        type=int,
        default=64,
        help="number of the samples generated during training",
    )
    parser.add_argument(
        "--size", type=int, default=256, help="image sizes for the model"
    )
    parser.add_argument(
        "--r1", type=float, default=10, help="weight of the r1 regularization"
    )
    parser.add_argument(
        "--path_regularize",
        type=float,
        default=2,
        help="weight of the path length regularization",
    )
    parser.add_argument(
        "--path_batch_shrink",
        type=int,
        default=2,
        help="路径长度正则化的批大小减少因子(减少内存消耗)",
    )
    parser.add_argument(
        "--d_reg_every",
        type=int,
        default=16,
        help="interval of the applying r1 regularization(应用r1正则化的区间)",
    )
    parser.add_argument(
        "--g_reg_every",
        type=int,
        default=4,
        help="interval of the applying path length regularization(应用路径长度正则化的间隔)",
    )
    parser.add_argument(
        "--mixing", type=float, default=0.9, help="probability of latent code mixing"
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="path to the checkpoints to resume training(去检查点恢复训练的路)",
    )
    parser.add_argument("--lr", type=float, default=0.002, help="learning rate")
    parser.add_argument(
        "--channel_multiplier",
        type=int,
        default=2,
        help="channel multiplier factor for the model. config-f = 2, else = 1",
    )
    parser.add_argument(
        "--wandb", action="store_true", help="use weights and biases logging(使用权重和偏差记录)"
    )
    parser.add_argument(
        "--local_rank", type=int, default=0, help="local rank for distributed training(分布式训练的局部秩)"
    )
    parser.add_argument(
        "--augment", action="store_true", help="apply non leaking augmentation"
    )
    parser.add_argument(
        "--augment_p",
        type=float,
        default=0,
        help="probability of applying augmentation. 0 = use adaptive augmentation",
    )
    parser.add_argument(
        "--ada_target",
        type=float,
        default=0.6,
        help="target augmentation probability for adaptive augmentation(自适应增强的目标增强概率)",
    )
    parser.add_argument(
        "--ada_length",
        type=int,
        default=500 * 1000,
        help="target duraing to reach augmentation probability for adaptive augmentation",
    )
    parser.add_argument(
        "--ada_every",
        type=int,
        default=256,
        help="probability update interval of the adaptive augmentation(自适应增强的概率更新间隔)"
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=12,
        help="总的尾部类别数"
    )

    args = parser.parse_args()

    n_gpu = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = n_gpu > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()

    args.latent = 512
    args.n_mlp = 8

    args.start_iter = 0

    from tgn import Generator, Discriminator

    print(
        f'args.size:{args.size} args.latent:{args.latent} args.n_mlp:{args.n_mlp} args.channel_multiplier:{args.channel_multiplier}',
        args.size, args.latent, args.n_mlp, args.channel_multiplier)

    Rmodel = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=args.num_classes).to(device)

    Rmodel.load_state_dict(
        torch.load('/root/autodl-tmp/TGN/KMDN/save_model_dill/RModel/RModel_epoch_1.pth', map_location=torch.device('cuda:0')))

    # 生成器
    generator = Generator(
        args.size, args.latent, args.num_classes, args.n_mlp, channel_multiplier=args.channel_multiplier
    ).to(device)

    # 判别器
    discriminator = Discriminator(
        args.size, args.num_classes, args.latent, channel_multiplier=args.channel_multiplier
    ).to(device)

    g_ema = Generator(
        args.size, args.latent, args.num_classes, args.n_mlp, channel_multiplier=args.channel_multiplier
    ).to(device)
    g_ema.eval()

    # 模型融合，decay=0等于没融合
    accumulate(g_ema, generator, 0)

    # 学习率因子
    g_reg_ratio = args.g_reg_every / (args.g_reg_every + 1)
    d_reg_ratio = args.d_reg_every / (args.d_reg_every + 1)

    g_optim = optim.Adam(
        generator.parameters(),
        lr=args.lr * g_reg_ratio,
        betas=(0 ** g_reg_ratio, 0.99 ** g_reg_ratio),
    )
    d_optim = optim.Adam(
        discriminator.parameters(),
        lr=args.lr * d_reg_ratio,
        betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio),
    )

    if args.ckpt is not None:
        print("load model:", args.ckpt)

        ckpt = torch.load(args.ckpt, map_location=lambda storage, loc: storage)

        try:
            ckpt_name = os.path.basename(args.ckpt)
            args.start_iter = int(os.path.splitext(ckpt_name)[0])
        except ValueError:
            pass

        # 准备模型的状态字典
        generator_dict = generator.state_dict()
        discriminator_dict = discriminator.state_dict()
        g_ema_dict = g_ema.state_dict()

        # 过滤掉不匹配的键
        pretrained_gen_dict = {k: v for k, v in ckpt["g"].items() if
                               k in generator_dict and generator_dict[k].shape == v.shape}
        pretrained_disc_dict = {k: v for k, v in ckpt["d"].items() if
                                k in discriminator_dict and discriminator_dict[k].shape == v.shape}
        pretrained_g_ema_dict = {k: v for k, v in ckpt["g_ema"].items() if
                                 k in g_ema_dict and g_ema_dict[k].shape == v.shape}

        # 更新模型状态字典
        generator_dict.update(pretrained_gen_dict)
        discriminator_dict.update(pretrained_disc_dict)
        g_ema_dict.update(pretrained_g_ema_dict)

        # 加载更新后的状态字典
        generator.load_state_dict(generator_dict)
        discriminator.load_state_dict(discriminator_dict)
        g_ema.load_state_dict(g_ema_dict)

    if args.distributed:
        generator = nn.parallel.DistributedDataParallel(
            generator,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
        )

        discriminator = nn.parallel.DistributedDataParallel(
            discriminator,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
        )

    transform = transforms.Compose(
        [
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor()  # ,
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
        ]
    )

    dataset = C1SatelliteDataset(
        rootdir=r"/root/autodl-tmp/TGN/CDTG/datasets/dior",
        train=True,
        download=False,
        transform=transforms.Compose([
            transforms.Resize((256, 256)),  # 添加这一行来调整图像大小
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
        ])
    )

    loader = data.DataLoader(dataset,
                             batch_size=args.batch,
                             num_workers=4,
                             sampler=data_sampler(dataset, shuffle=True, distributed=args.distributed),
                             drop_last=True)

    fusion_module = NoiseFeatureFusion(262144, 512).to(device)
    train(args, loader, generator, discriminator, g_optim, d_optim, g_ema, device, Rmodel, fusion_module)
