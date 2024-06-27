import os
import shutil


def count_category_samples(path_to_categories):
    """统计每个类别的样本数量"""
    category_counts = {}
    for category in os.listdir(path_to_categories):
        category_path = os.path.join(path_to_categories, category)
        if os.path.isdir(category_path):
            category_counts[category] = len(os.listdir(category_path))
    return category_counts


def split_categories(category_counts, threshold_percentage):
    """根据给定的百分比阈值划分头部类和尾部类"""
    threshold_category_count = int(len(category_counts) * (threshold_percentage / 100))
    sorted_categories = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)
    head_classes = [category for category, count in sorted_categories[:threshold_category_count]]
    tail_classes = [category for category, count in sorted_categories[threshold_category_count:]]
    return head_classes, tail_classes


def copy_categories(original_path, new_path, categories):
    """复制特定类别的文件夹到新路径"""
    for category in categories:
        source_path = os.path.join(original_path, category)
        destination_path = os.path.join(new_path, category)
        if os.path.isdir(source_path):
            shutil.copytree(source_path, destination_path)


def split_txt_file(txt_file_path, head_classes, tail_classes, new_head_path, new_tail_path):
    """根据类别划分txt文件"""
    with open(txt_file_path, "r") as file:
        lines = file.readlines()

    head_lines = [line for line in lines if line.split('/')[0] in head_classes]
    tail_lines = [line for line in lines if line.split('/')[0] in tail_classes]

    with open(os.path.join(new_head_path, "head_classes.txt"), "w") as file:
        file.writelines(head_lines)

    with open(os.path.join(new_tail_path, "tail_classes.txt"), "w") as file:
        file.writelines(tail_lines)


if __name__ == "__main__":
    folder_name = "datasets"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    original_path = r"/root/autodl-tmp/TGN/dior/train"
    new_base_path = r"/root/autodl-tmp/TGN/KMDN/datasets/dior/train"
    txt_file_path = r"/root/autodl-tmp/TGN/dior/anno/DIOR_train.txt"  # 假设txt文件在原始路径下
    new_head_path = r"/root/autodl-tmp/TGN/KMDN/datasets/dior/train/head"
    new_tail_path = r"/root/autodl-tmp/TGN/KMDN/datasets/dior/train/tail"

    # threshold_percentage = 33
    threshold_percentage = 40

    # 统计类别样本数
    category_counts = count_category_samples(original_path)

    # 划分类别
    head_classes, tail_classes = split_categories(category_counts, threshold_percentage)

    print("Head classes:", head_classes)
    print("Tail classes:", tail_classes)

    # 创建新路径
    new_head_path = os.path.join(new_base_path, "head")
    new_tail_path = os.path.join(new_base_path, "tail")
    os.makedirs(new_head_path, exist_ok=True)
    os.makedirs(new_tail_path, exist_ok=True)

    # 复制文件
    copy_categories(original_path, new_head_path, head_classes)
    copy_categories(original_path, new_tail_path, tail_classes)

    split_txt_file(txt_file_path, head_classes, tail_classes, new_head_path, new_tail_path)

    print("Finished!!!!!!")

    original_path = r"/root/autodl-tmp/TGN/dior/test"
    new_base_path = r"/root/autodl-tmp/TGN/KMDN/datasets/dior/test"
    txt_file_path = r"/root/autodl-tmp/TGN/dior/anno/DIOR_test.txt"  # 假设txt文件在原始路径下
    new_head_path = r"/root/autodl-tmp/TGN/KMDN/datasets/dior/test/head"
    new_tail_path = r"/root/autodl-tmp/TGN/KMDN/datasets/dior/test/tail"
    category_counts = count_category_samples(original_path)
    # 划分类别
    head_classes, tail_classes = split_categories(category_counts, threshold_percentage)

    print("Head classes:", head_classes)
    print("Tail classes:", tail_classes)

    # 创建新路径
    new_head_path = os.path.join(new_base_path, "head")
    new_tail_path = os.path.join(new_base_path, "tail")
    os.makedirs(new_head_path, exist_ok=True)
    os.makedirs(new_tail_path, exist_ok=True)

    # 复制文件
    copy_categories(original_path, new_head_path, head_classes)
    copy_categories(original_path, new_tail_path, tail_classes)

    split_txt_file(txt_file_path, head_classes, tail_classes, new_head_path, new_tail_path)

    print("Finished!!!!!!")