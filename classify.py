import os
import shutil
from PIL import Image  # 用于检测图片是否损坏
from sklearn.model_selection import train_test_split


def is_image_corrupted(image_path):
    """检查图片是否损坏"""
    try:
        with Image.open(image_path) as img:
            img.verify()  # 验证图片完整性
        return False
    except (IOError, SyntaxError):
        return True


def remove_corrupted_images(folder_path):
    """删除指定文件夹中的损坏图片"""
    removed_count = 0
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if is_image_corrupted(file_path):
            os.remove(file_path)
            removed_count += 1
            print(f"已删除损坏文件: {file_path}")
    return removed_count


# 原始数据路径
data_dir = "D:/tomato"
class_names = ["斑枯病", "健康", "轮斑病"]  # 你的类别文件夹名称

# 输出路径（按需修改）
output_dir = "D:/tomato_split"
os.makedirs(output_dir, exist_ok=True)

# 创建划分后的文件夹
for split in ["train", "val", "test"]:
    for class_name in class_names:
        os.makedirs(os.path.join(output_dir, split, class_name), exist_ok=True)

# 划分比例
train_ratio = 0.6
val_ratio = 0.2
test_ratio = 0.2

# 初始化统计字典
stats = {
    'total': {'train': 0, 'val': 0, 'test': 0},
    'class_details': {class_name: {'train': 0, 'val': 0, 'test': 0}
                      for class_name in class_names}
}

# 遍历每个类别
for class_name in class_names:
    class_dir = os.path.join(data_dir, class_name)

    # 删除原始数据中的损坏图片
    print(f"\n正在检查类别 '{class_name}' 的损坏图片...")
    corrupted_count = remove_corrupted_images(class_dir)
    print(f"已删除 {corrupted_count} 张损坏图片")

    files = [f for f in os.listdir(class_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]  # 过滤图片文件

    # 第一次划分：训练集 vs 临时集
    train_files, temp_files = train_test_split(
        files, test_size=(val_ratio + test_ratio), random_state=42)

    # 第二次划分：验证集 vs 测试集
    val_files, test_files = train_test_split(
        temp_files, test_size=test_ratio / (val_ratio + test_ratio), random_state=42)

    # 复制文件到目标文件夹
    for f in train_files:
        src = os.path.join(class_dir, f)
        dst = os.path.join(output_dir, "train", class_name, f)
        shutil.copy(src, dst)
        stats['total']['train'] += 1
        stats['class_details'][class_name]['train'] += 1

    for f in val_files:
        src = os.path.join(class_dir, f)
        dst = os.path.join(output_dir, "val", class_name, f)
        shutil.copy(src, dst)
        stats['total']['val'] += 1
        stats['class_details'][class_name]['val'] += 1

    for f in test_files:
        src = os.path.join(class_dir, f)
        dst = os.path.join(output_dir, "test", class_name, f)
        shutil.copy(src, dst)
        stats['total']['test'] += 1
        stats['class_details'][class_name]['test'] += 1

# 打印统计信息
print("\n" + "=" * 50)
print("数据集划分统计信息:")
print(f"总图片数: {sum(stats['total'].values())}")
print(f"训练集: {stats['total']['train']} 张")
print(f"验证集: {stats['total']['val']} 张")
print(f"测试集: {stats['total']['test']} 张")

print("\n按类别统计:")
for class_name in class_names:
    print(f"\n类别 '{class_name}':")
    print(f"  训练集: {stats['class_details'][class_name]['train']} 张")
    print(f"  验证集: {stats['class_details'][class_name]['val']} 张")
    print(f"  测试集: {stats['class_details'][class_name]['test']} 张")

# 检查目标文件夹中的损坏图片
print("\n正在检查目标文件夹中的损坏图片...")
total_removed = 0
for split in ["train", "val", "test"]:
    for class_name in class_names:
        folder = os.path.join(output_dir, split, class_name)
        removed = remove_corrupted_images(folder)
        total_removed += removed
        if removed > 0:
            print(f"在 {split}/{class_name} 中移除了 {removed} 张损坏图片")

print(f"\n总共移除了 {total_removed} 张损坏图片")

print("\n划分完成！输出结构：")
print(f"""
{output_dir}/
├── train/
│   ├── 斑枯病/  ({stats['class_details']['斑枯病']['train']}张)
│   ├── 健康/    ({stats['class_details']['健康']['train']}张)
│   └── 轮斑病/  ({stats['class_details']['轮斑病']['train']}张)
├── val/
│   ├── 斑枯病/  ({stats['class_details']['斑枯病']['val']}张)
│   ├── 健康/    ({stats['class_details']['健康']['val']}张)
│   └── 轮斑病/  ({stats['class_details']['轮斑病']['val']}张)
└── test/
    ├── 斑枯病/  ({stats['class_details']['斑枯病']['test']}张)
    ├── 健康/    ({stats['class_details']['健康']['test']}张)
    └── 轮斑病/  ({stats['class_details']['轮斑病']['test']}张)
""")