import os
import torch
import json
import numpy as np
from PIL import Image
from torchvision import transforms
from model import MobileNetV1_ShuffleAtt
import matplotlib.pyplot as plt
from tqdm import tqdm
import csv
from datetime import datetime
from collections import defaultdict, Counter

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


class Config:
    """配置参数"""
    MODEL_PATH = 'best_model_v2.pth'
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    INPUT_FOLDER = 'D:/tomato_split/test'
    OUTPUT_FOLDER = 'prediction_results_3'
    SAVE_RESULTS = True
    SHOW_PLOTS = False
    ALLOWED_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')


def get_transform():
    """获取图像预处理转换（与训练时保持一致）"""
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


def load_class_indices(json_path='class_indices_v2.json'):
    """加载类别索引映射"""
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"类别索引文件不存在: {json_path}")

    with open(json_path, 'r', encoding='utf-8') as f:
        class_indices = json.load(f)

    return {int(k): v for k, v in class_indices.items()}


def load_model(model_path, class_indices):
    try:
        num_classes = len(class_indices)

        model = MobileNetV1_ShuffleAtt(
            num_classes=num_classes,
            label_smoothing=0.1
        ).to(Config.DEVICE)

        # 加载模型权重
        checkpoint = torch.load(model_path, map_location=Config.DEVICE)

        # 处理不同的保存格式
        if 'model_state' in checkpoint:
            model.load_state_dict(checkpoint['model_state'], strict=True)
            best_acc = checkpoint.get('accuracy', '未知')
            print(f"模型加载成功 | 最佳验证准确率: {best_acc:.2f}%" if isinstance(best_acc, float) else f"模型加载成功")
        else:
            model.load_state_dict(checkpoint, strict=True)
            print("模型加载成功（旧格式）")

        model.eval()
        return model

    except Exception as e:
        raise RuntimeError(f"模型加载失败: {str(e)}")


def get_true_label(img_path, input_folder, class_names):
    rel_path = os.path.relpath(img_path, input_folder)
    true_class = rel_path.split(os.sep)[0]

    if true_class not in class_names:
        raise ValueError(f"无效标签: {true_class} (允许的标签: {class_names})")
    return true_class


def predict_image(img_path, model, transform, class_indices, input_folder):
    """预测单张图像"""
    try:
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"文件不存在: {img_path}")

        img = Image.open(img_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(Config.DEVICE)

        assert img_tensor.shape == (1, 3, 224, 224), \
            f"输入形状错误: {img_tensor.shape}，预期 (1, 3, 224, 224)"

        class_names = list(class_indices.values())
        true_class = get_true_label(img_path, input_folder, class_names)

        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0].cpu().numpy()
            pred_idx = torch.argmax(outputs).item()
            pred_class = class_indices[pred_idx]

        return {
            'status': 'success',
            'filename': os.path.basename(img_path),
            'image_path': img_path,
            'true_class': true_class,
            'prediction': pred_class,
            'pred_idx': pred_idx,
            'probabilities': probabilities,
            'image': img,
            'image_tensor': img_tensor[0].cpu()
        }

    except Exception as e:
        return {
            'status': 'error',
            'message': f"{type(e).__name__}: {str(e)}",
            'image_path': img_path,
            'filename': os.path.basename(img_path) if img_path else '未知'
        }


def save_visualization(result, save_path, class_indices):
    """保存预测可视化结果"""
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])
    img_tensor = result['image_tensor'].clone()
    img_np = img_tensor * std.view(3, 1, 1) + mean.view(3, 1, 1)
    img_np = img_np.permute(1, 2, 0).numpy()
    img_np = np.clip(img_np, 0, 1)

    # 创建可视化图表
    fig = plt.figure(figsize=(12, 5))

    # 原始图像
    ax1 = fig.add_subplot(121)
    ax1.imshow(img_np)
    ax1.set_title(f"预测: {result['prediction']}\n真实: {result['true_class']}")
    ax1.axis('off')

    # 概率分布
    ax2 = fig.add_subplot(122)
    class_names = list(class_indices.values())
    colors = ['green' if i == result['pred_idx'] else 'gray' for i in range(len(class_names))]
    bars = ax2.barh(class_names, result['probabilities'], color=colors)
    ax2.set_xlim(0, 1)
    ax2.set_title('类别概率分布')

    # 添加概率值标签
    for bar in bars:
        width = bar.get_width()
        ax2.text(width + 0.02, bar.get_y() + bar.get_height() / 2,
                 f'{width:.2%}', ha='left', va='center')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')

    if Config.SHOW_PLOTS:
        plt.show()
    plt.close()

def collect_image_files(folder_path):
    """收集所有有效图像文件"""
    image_files = []
    total_files = 0

    for root, _, files in os.walk(folder_path):
        total_files += len([f for f in files if f.lower().endswith(Config.ALLOWED_EXTENSIONS)])

    with tqdm(total=total_files, desc="扫描图像文件") as pbar:
        for root, _, files in os.walk(folder_path):
            for file in files:
                if file.lower().endswith(Config.ALLOWED_EXTENSIONS):
                    full_path = os.path.join(root, file)
                    try:
                        with Image.open(full_path) as img:
                            img.verify()
                        image_files.append(full_path)
                    except Exception as e:
                        print(f"\n损坏的图像文件: {full_path} - {str(e)}")
                    finally:
                        pbar.update(1)

    if not image_files:
        raise RuntimeError(f"在 {folder_path} 中未找到任何有效图像文件")

    print(f"找到 {len(image_files)} 个有效图像文件 (共扫描 {total_files} 个文件)")
    return image_files


def predict_folder(folder_path, model, transform, class_indices):
    """预测文件夹中的所有图像"""
    os.makedirs(Config.OUTPUT_FOLDER, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = os.path.join(Config.OUTPUT_FOLDER, f"report_{timestamp}")
    os.makedirs(report_dir, exist_ok=True)

    image_files = collect_image_files(folder_path)
    class_names = list(class_indices.values())

    results = []
    error_log = []
    total_correct = 0
    class_correct = defaultdict(int)
    class_total = defaultdict(int)

    for img_path in tqdm(image_files, desc="预测进度"):
        result = predict_image(img_path, model, transform, class_indices, folder_path)

        if result['status'] == 'success':
            results.append(result)

            true_class = result['true_class']
            class_total[true_class] += 1

            if result['prediction'] == true_class:
                total_correct += 1
                class_correct[true_class] += 1

            if Config.SAVE_RESULTS:
                save_name = f"{os.path.splitext(result['filename'])[0]}_result.jpg"
                save_path = os.path.join(report_dir, save_name)
                save_visualization(result, save_path, class_indices)
        else:
            error_log.append(result)

    # 保存报告
    if Config.SAVE_RESULTS:
        save_report(report_dir, results, error_log, class_names, class_correct, class_total)

    return results, error_log


def save_report(report_dir, results, error_log, class_names, class_correct, class_total):
    """保存预测报告"""
    csv_path = os.path.join(report_dir, 'predictions.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
        fieldnames = ['filename', 'true_class', 'prediction', 'is_correct'] + class_names
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for r in results:
            row = {
                'filename': r['filename'],
                'true_class': r['true_class'],
                'prediction': r['prediction'],
                'is_correct': '正确' if r['prediction'] == r['true_class'] else '错误'
            }
            # 添加各类别概率
            for i, cls in enumerate(class_names):
                row[cls] = f"{r['probabilities'][i]:.4f}"
            writer.writerow(row)

    print("\n" + "=" * 60)
    print("预测结果汇总")
    print("=" * 60)
    print(f"总预测数: {len(results) + len(error_log)}")
    print(f"成功预测: {len(results)}")
    print(f"预测失败: {len(error_log)}")

    total_correct_count = 0
    for result in results:
        if result['prediction'] == result['true_class']:
            total_correct_count += 1

    if results:
        total_acc = total_correct_count / len(results) * 100
        print(f"总准确率: {total_acc:.2f}%")

        print("\n各类别预测详情:")
        for cls in class_names:
            total = class_total.get(cls, 0)
            correct = class_correct.get(cls, 0)
            wrong = total - correct if total > 0 else 0
            acc = correct / total * 100 if total > 0 else 0
            print(f"  {cls}: 总数={total}, 正确={correct}, 错误={wrong}, 准确率={acc:.2f}%")

    if error_log:
        error_path = os.path.join(report_dir, 'errors.csv')
        with open(error_path, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=['filename', 'image_path', 'message'])
            writer.writeheader()
            writer.writerows(error_log)

    print(f"\n详细报告保存至: {report_dir}")


def main():
    print("=" * 60)
    print("番茄叶片病害识别预测系统")
    print("=" * 60)
    print(f"使用设备: {Config.DEVICE}")
    print(f"模型路径: {Config.MODEL_PATH}")
    print(f"测试数据路径: {Config.INPUT_FOLDER}")

    try:
        class_indices = load_class_indices()
        print(f"类别信息: {list(class_indices.values())}")

        model = load_model(Config.MODEL_PATH, class_indices)

        transform = get_transform()

        results, errors = predict_folder(Config.INPUT_FOLDER, model, transform, class_indices)

        if results:
            sample = results[0]
            print("\n示例预测结果:")
            print(f"文件名: {sample['filename']}")
            print(f"真实类别: {sample['true_class']}")
            print(f"预测类别: {sample['prediction']}")
            print("概率分布:")
            for i, cls in enumerate(class_indices.values()):
                print(f"  {cls}: {sample['probabilities'][i]:.2%}")

    except Exception as e:
        print(f"\n预测过程出错: {str(e)}")

    finally:
        # 清理GPU内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("\n预测完成")


if __name__ == '__main__':
    main()