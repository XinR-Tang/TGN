import argparse

import torch
from torchvision import utils

import numpy as np

from swagan import Generator
from PIL import Image

torch.manual_seed(0)
np.random.seed(0)
import random

random.seed(0)


def make_image(tensor):
    return (
        tensor.detach()
        .clamp_(min=-1, max=1)
        .add(1)
        .div_(2)
        .mul(255)
        .type(torch.uint8)
        .permute(0, 2, 3, 1)
        .to("cpu")
        .numpy()
    )


if __name__ == "__main__":
    torch.set_grad_enabled(False)

    parser = argparse.ArgumentParser(description="Apply closed form factorization")

    parser.add_argument(
        "-i", "--index", type=int, default=0, help="index of eigenvector"
    )
    parser.add_argument(
        "-d",
        "--degree",
        type=float,
        default=5,
        help="scalar factors for moving latent vectors along eigenvector",
    )
    parser.add_argument(
        "--channel_multiplier",
        type=int,
        default=2,
        help='channel multiplier factor. config-f = 2, else = 1',
    )
    parser.add_argument("--ckpt", type=str, required=True, help="stylegan2 checkpoints")
    parser.add_argument(
        "--size", type=int, default=256, help="output image size of the generator"
    )
    parser.add_argument(
        "-n", "--n_sample", type=int, default=7, help="number of samples created"
    )
    parser.add_argument(
        "--truncation", type=float, default=0.7, help="truncation factor"
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="device to run the model"
    )
    parser.add_argument(
        "--out_prefix",
        type=str,
        default="factor",
        help="filename prefix to result samples",
    )
    parser.add_argument(
        "factor",
        type=str,
        help="name of the closed form factorization result factor file",
    )

    args = parser.parse_args()

    eigvec = torch.load(args.factor)["eigvec"].to(args.device)
    ckpt = torch.load(args.ckpt)
    g = Generator(args.size, 512, 8, channel_multiplier=args.channel_multiplier).to(args.device)
    g.load_state_dict(ckpt["g_ema"], strict=False)

    trunc = g.mean_latent(4096)

    latent = torch.randn(args.n_sample, 512, device=args.device)
    latent = g.get_latent(latent)
    # print(torch.load('inversion_codes/desert_001.pt').keys())
    # latent = torch.load('inversion_codes/desert_001.pt')['../NWPU-RESISC45-Imbalance/train/desert/desert_001.jpg']['latent']

    direction = args.degree * eigvec[:, args.index].unsqueeze(0)
    print(args.index)
    print(torch.mean(latent))
    print(torch.std(latent))
    print(torch.min(latent), torch.max(latent))
    img, _ = g([latent], truncation=args.truncation, truncation_latent=trunc, input_is_latent=True)
    print("BEFORE MAKE IMAGE:")
    print(img.shape)
    print(torch.min(img[0, 0, :, :]), torch.max(img[0, 0, :, :]))
    print(torch.min(img[0, 1, :, :]), torch.max(img[0, 1, :, :]))
    print(torch.min(img[0, 2, :, :]), torch.max(img[0, 2, :, :]))
    print(torch.std(img[0, 0, :, :]), torch.std(img[0, 1, :, :]), torch.std(img[0, 2, :, :]))
    print(torch.mean(img[0, 0, :, :]), torch.mean(img[0, 1, :, :]), torch.mean(img[0, 2, :, :]))
    img_arr = make_image(img)
    print("AFTER MAKE IMAGE:")
    print(img_arr.shape)
    temp_img = img_arr[0]
    print(np.min(temp_img[:, :, 0]), np.max(temp_img[:, :, 0]))
    print(np.min(temp_img[:, :, 1]), np.max(temp_img[:, :, 1]))
    print(np.min(temp_img[:, :, 2]), np.max(temp_img[:, :, 2]))
    print(np.std(temp_img[:, :, 0]), np.std(temp_img[:, :, 1]), np.std(temp_img[:, :, 2]))
    print(np.mean(temp_img[:, :, 0]), np.mean(temp_img[:, :, 1]), np.mean(temp_img[:, :, 2]))
    pil_img = Image.fromarray(img_arr[0])
    pil_img.save("test.png")

    img, _ = g(
        [latent],
        truncation=args.truncation,
        truncation_latent=trunc,
        input_is_latent=True,
    )
    img1, _ = g(
        [latent + direction],
        truncation=args.truncation,
        truncation_latent=trunc,
        input_is_latent=True,
    )
    img1_2, _ = g(
        [latent + 2 * direction],
        truncation=args.truncation,
        truncation_latent=trunc,
        input_is_latent=True,
    )
    img1_3, _ = g(
        [latent + 3 * direction],
        truncation=args.truncation,
        truncation_latent=trunc,
        input_is_latent=True,
    )
    img2, _ = g(
        [latent - direction],
        truncation=args.truncation,
        truncation_latent=trunc,
        input_is_latent=True,
    )
    img2_2, _ = g(
        [latent - 2 * direction],
        truncation=args.truncation,
        truncation_latent=trunc,
        input_is_latent=True,
    )
    img2_3, _ = g(
        [latent - 3 * direction],
        truncation=args.truncation,
        truncation_latent=trunc,
        input_is_latent=True,
    )

    for i in range(img.shape[0]):
        utils.save_image(img[i], f"comp{args.index}_ex{i}_img0.png", normalize=True, range=(0, 1))
        utils.save_image(img1[i], f"comp{args.index}_ex{i}_img+1.png", normalize=True, range=(0, 1))
        utils.save_image(img1_2[i], f"comp{args.index}_ex{i}_img+2.png", normalize=True, range=(0, 1))
        utils.save_image(img1_3[i], f"comp{args.index}_ex{i}_img+3.png", normalize=True, range=(0, 1))
        utils.save_image(img2[i], f"comp{args.index}_ex{i}_img-1.png", normalize=True, range=(0, 1))
        utils.save_image(img2_2[i], f"comp{args.index}_ex{i}_img-2.png", normalize=True, range=(0, 1))
        utils.save_image(img2_3[i], f"comp{args.index}_ex{i}_img-3.png", normalize=True, range=(0, 1))

    # grid = utils.save_image(
    #    torch.cat([img1_3, img1_2, img1, img, img2, img2_2, img2_3], 0),
    #    f"{args.out_prefix}_index-{args.index}_degree-{args.degree}.png",
    #    normalize=True,
    #    range=(0, 1),
    #    nrow=args.n_sample,
    # )

