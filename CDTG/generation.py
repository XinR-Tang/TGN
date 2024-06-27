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
from PIL import Image

try:
    import wandb

except ImportError:
    wandb = None

from dataset import DiorDataset
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


class NoiseFeatureFusion(nn.Module):
    def __init__(self, feature_dim, noise_dim):
        super(NoiseFeatureFusion, self).__init__()
        self.fc = nn.Linear(feature_dim, noise_dim)

    def forward(self, gan_feature, noise, r):
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
        fused_noise = r * noise + (1 - r) * gan_feature
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


def make_noise(batch, latent_dim, n_noise, device, gan_feature, fusion_module, r):
    if n_noise == 1:
        noise = torch.randn(batch, latent_dim, device=device)
    else:
        noise = torch.randn(n_noise, batch, latent_dim, device=device).unbind(0)
        # for noise in noise:
        #     print(noise.shape)

    # 使用融合模块
    if fusion_module is not None:
        noise = [fusion_module(gan_feature, n, r) for n in noise]

    return noise


def mixing_noise(batch, latent_dim, prob, device, gan_feature, fusion_module, r):
    if prob > 0:
        return make_noise(batch, latent_dim, 2, device, gan_feature, fusion_module, r)

    else:
        return [make_noise(batch, latent_dim, 1, device, gan_feature, fusion_module, r)]


def set_grad_none(model, targets):
    for n, p in model.named_parameters():
        if n in targets:
            p.grad = None


def save_image(img, label, image_number, root_dir='/root/autodl-tmp/TGN/CDTG/result'):
    """
    Save an image to a specified directory.

    Args:
    img (Tensor): The image tensor.
    label (int): The label of the image.
    image_number (int): A unique identifier for the image.
    root_dir (str): The root directory where the image will be saved.
    """
    # Create the directory if it does not exist
    label_dir = os.path.join(root_dir, str(label))
    os.makedirs(label_dir, exist_ok=True)
    img.save(os.path.join(label_dir, f"{image_number}.jpg"))


def generation(args, loader, generator, discriminator, g_optim, d_optim, g_ema, device, Rmodel, fusion_module, r):
    loader = sample_data(loader)

    image_count = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        _, gan_feature = Rmodel(images)

        # Generate 5 images
        for _ in range(5):
            noise = mixing_noise(1, args.latent, args.mixing, device, gan_feature, fusion_module, r)
            fake_img, _ = g_ema(noise, labels)

            if len(fake_img.shape) == 4:
                fake_img = fake_img[0]

            # 将Tensor转换为PIL图像
            # 首先取消归一化
            fake_img = fake_img * 0.5 + 0.5
            # 确保图像值在0到1之间
            fake_img = torch.clamp(fake_img, 0, 1)
            # 转换为PIL图像
            fake_img_pil = transforms.ToPILImage()(fake_img.cpu())

            # Save the generated image
            save_image(fake_img_pil, labels.item(), image_count)
            image_count += 1
        if image_count == 2000:
            print("Over")
            break


if __name__ == "__main__":
    device = "cuda"

    parser = argparse.ArgumentParser(description="StyleGAN2 trainer")

    # parser.add_argument("path", type=str, help="path to the lmdb dataset")
    parser.add_argument('--arch', type=str, default='stylegan2', help='model architectures')
    parser.add_argument(
        "--iter", type=int, default=200000, help="total training iterations"
    )
    parser.add_argument(
        "--batch", type=int, default=1, help="batch sizes for each gpus"
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

    parser.add_argument(
        "--folder_number",
        type=int,
        default=12,
        help="要生成的尾类标签"
    )

    parser.add_argument(
        "--r",
        type=float,
        default=0.05,
        help="噪声比例（0-1）"
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

    Rmodel = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=12).to(device)

    Rmodel.load_state_dict(
        torch.load('/root/autodl-tmp/TGN/KMDN/save_model_dill/RModel/RModel_epoch_1.pth',
                   map_location=torch.device('cuda:0')))

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

    dataset = DiorDataset(
        root_dir='/root/autodl-tmp/TGN/CDTG/datasets/dior/train/tail',
        folder_number=args.folder_number,
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
    generation(args, loader, generator, discriminator, g_optim, d_optim, g_ema, device, Rmodel, fusion_module, args.r)
