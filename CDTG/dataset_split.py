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


def copy_and_rename_categories(original_path, new_path, categories):
    """复制特定类别的文件夹到新路径，并确保文件夹名从0开始递增"""
    for idx, category in enumerate(categories):
        source_path = os.path.join(original_path, category)
        destination_path = os.path.join(new_path, str(idx))
        if os.path.isdir(source_path):
            shutil.copytree(source_path, destination_path)


if __name__ == "__main__":
    folder_name = "datasets"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    original_train_path = r"/root/autodl-tmp/TGN/dior/train"
    new_train_base_path = r"/root/autodl-tmp/TGN/CDTG/datasets/dior/train"
    threshold_percentage = 40

    # 统计类别样本数
    train_category_counts = count_category_samples(original_train_path)

    # 划分类别
    head_classes, tail_classes = split_categories(train_category_counts, threshold_percentage)

    print("Head classes:", head_classes)
    print("Tail classes:", tail_classes)

    # 创建新路径
    new_train_tail_path = os.path.join(new_train_base_path, "tail")
    os.makedirs(new_train_tail_path, exist_ok=True)

    # 复制并重命名文件夹
    copy_and_rename_categories(original_train_path, new_train_tail_path, tail_classes)

    print("Finished Train Split!!!!!!")

