# 放到clip软件包中
import os

import numpy as np
import torch
import torch.nn.functional as F

import clip
from PIL import Image
import torch
from torchvision.transforms.functional import to_pil_image

def batch_tensor_to_numpy(batch_tensor):
    # 确保输入是一个 4D tensor
    if not (isinstance(batch_tensor, torch.Tensor) and batch_tensor.ndimension() == 4):
        raise TypeError("Input tensor is not a 4D torch.Tensor.")

    # 转换批量中的每个图像
    numpy_images = [to_pil_image(batch_tensor[i]).convert("RGB") for i in range(batch_tensor.size(0))]
    numpy_images = [np.array(img) for img in numpy_images]

    return numpy_images

def numpy_to_pil(numpy_images):
    return [Image.fromarray(img) for img in numpy_images]

def get_clip_loss(PIL_image_input, text, labels, device):
    PIL_image_input = batch_tensor_to_numpy(PIL_image_input)
    PIL_image_input = numpy_to_pil(PIL_image_input)  # 将 numpy 数组转换为 PIL 图像
    model, preprocess = clip.load("ViT-B/32", device=device)
    image_features_list = []

    with torch.no_grad():
        for img in PIL_image_input:
            processed_img = preprocess(img).unsqueeze(0).to(device)
            image_features = model.encode_image(processed_img)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            image_features_list.append(image_features)

        # 将所有图像的特征堆叠成一个批量
        image_features_batch = torch.cat(image_features_list, dim=0)

        text_features = model.encode_text(text)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        logit_scale = np.exp(model.logit_scale.data.item())
        text_probs = (logit_scale * image_features_batch @ text_features.T).softmax(dim=-1)

        labels = torch.tensor(labels).to(device)
        loss = F.cross_entropy(text_probs, labels)

    print(loss.item())
    return loss

    # train
    # images_similarity = image_features @ image_features.T
    # texts_similarity = text_features @ text_features.T
    # targets = F.softmax((images_similarity + texts_similarity) / 2 * temperature, dim = -1
    # )
    # texts_Loss = F.cross_entropy(Logits, targets, reduction='none ')
    # images_Loss = F.cross_entropy(Logits.T, targets.T, reduction='none')
    # Loss = (images_Loss + texts_Loss) / 2.0  # ??

    #
    # if config.n_gpu > 1:
    #     loss = loss.mean()  # mean() to average on multi-gpu parallel training
    # if config.gradient_accumulation_steps > 1:
    #     loss = loss / config.gradient_accumulation_steps
    #     # 梯度累积是一种训练技巧，特别在GPU显存受限的情况下可以用来处理较大的batch size。
    #     # 梯度累积的原理是将batch size分成多个小batch，然后在每个小batch上分别计算梯度，最后将这些梯度累积起来再更新参数。


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("RN50", device=device)
    images = []
    labels = []
    path = r'D:\BaiduNetdiskDownload\DIOR\dior\ds\test\head'
    files = [r"0\11726_0.jpg", r"0\11726_1.jpg", r"2\11731_0.jpg",r"0\11726_0.jpg",r"0\11726_0.jpg"]
    file_path = [os.path.join(path, f) for f in files]

    for file in file_path:
        images.append(preprocess(Image.open(file).convert("RGB")))
        labels.append(int(file.split('\\')[-2]))
    print(labels)
    image_input = torch.tensor(np.stack(images)).to(device)

    # 首先生成每个类别的文本描述，现在数据集里有0-19种
    descriptions = {
        "0": "airplane",
        "1": "baseball field",
        "2": "harbor",
        "3": "ship",
        "4": "storage tank",
        "5": "tennis court",
        "6": "vehicle",
        "7": "windmill",
        "8": "airport",
        "9": "basketball court",
        "10": "bridge",
        "11": "chimney",
        "12": "dam",
        "13": "expressway-service-area",
        "14": "expressway-toll-station",
        "15": "golf field",
        "16": "ground track field",
        "17": "overpass",
        "18": "stadium",
        "19": "train station",
    }
    des = list(descriptions.values())
    text_descriptions = [f"A photo of a " + x for x in des]
    text = clip.tokenize(text_descriptions).to(device)

    clip_loss = get_clip_loss(image_input, text, labels)
# inference
#     logits_per_image, logits_per_text = model(image, text)  # logits为1x3和3x1的矩阵
#     probs = logits_per_image.softmax(dim=-1).cpu().numpy()
# print(image_features, text_features)
# print(logits_per_image, logits_per_text)
# print("Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]

# # 读取图像
# original_images = []
# images = []
# texts = []
#
# for label in labels:
#     image_file = os.path.join("images", label + ".jpg")
#     name = os.path.basename(image_file).split('.')[0]
#
#     image = Image.open(image_file).convert("RGB")
#     original_images.append(image)
#     images.append(preprocess(image))
#     texts.append(name)
#
# image_input = torch.tensor(np.stack(images)).cuda()
#
# # 提取图像特征
# with torch.no_grad():
#     image_features = model.encode_image(image_input).float()
#     image_features /= image_features.norm(dim=-1, keepdim=True)
#
# # 计算余弦相似度（未缩放）
# similarity = text_features.cpu().numpy() @ image_features.cpu().numpy().T
#
# # 我们也可以对得到的余弦相似度计算softmax，得到每个预测类别的概率值，注意这里要对相似度进行缩放，区分度更好
# logit_scale = np.exp(model.logit_scale.data.item())
# text_probs = (logit_scale * image_features @ text_features.T).softmax(dim=-1)
# top_probs, top_labels = text_probs.cpu().topk(5, dim=-1)
