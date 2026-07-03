import torch
import torch.nn as nn
from torchvision import transforms, datasets
import torch.optim as optim
from model import MobileNetV1_ShuffleAtt
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import json
import gc
import pandas as pd
import shutil
import glob

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batch_size = 32
num_epochs = 100

data_transform = {
    "train": transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    "val": transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

def init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, 0, 0.01)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

def create_data_loaders(train_dir, val_dir):
    try:
        train_data = datasets.ImageFolder(train_dir, transform=data_transform["train"])
        val_data = datasets.ImageFolder(val_dir, transform=data_transform["val"])

        # 保存类别映射
        with open('class_indices_v2.json', 'w') as f:
            json.dump({v: k for k, v in train_data.class_to_idx.items()}, f, indent=4)

        print("\n训练集分布:")
        class_counts = {cls: 0 for cls in train_data.class_to_idx}
        for path, label in train_data.samples:
            cls_name = train_data.classes[label]
            class_counts[cls_name] += 1
        for cls, count in class_counts.items():
            print(f"{cls}: {count}张")

        return train_data, val_data

    except Exception as e:
        raise RuntimeError(f"数据加载失败: {str(e)}")

def create_loader(dataset, batch_size, shuffle=True):
    num_workers = 4 if torch.cuda.is_available() else 0
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        drop_last=True
    )

def train_epoch(model, loader, criterion, optimizer, device, epoch):
    """单epoch训练"""
    model.train()
    total_loss, total_correct, total_samples = 0, 0, 0
    pbar = tqdm(loader, desc=f'Epoch {epoch + 1}', dynamic_ncols=True)

    for inputs, targets in pbar:
        inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total_correct += predicted.eq(targets).sum().item()
        total_samples += targets.size(0)

        pbar.set_postfix({
            'bs': batch_size,
            'loss': f"{total_loss / (pbar.n + 1):.3f}",
            'acc': f"{100. * total_correct / total_samples:.1f}%"
        })

        if pbar.n % 50 == 0:
            gc.collect()
            torch.cuda.empty_cache()

    return total_loss / len(loader), 100. * total_correct / total_samples

def validate(model, loader, criterion, device):
    """验证过程"""
    model.eval()
    total_loss, total_correct = 0, 0
    class_correct = [0] * len(loader.dataset.classes)
    class_total = [0] * len(loader.dataset.classes)

    with torch.no_grad():
        for inputs, targets in tqdm(loader, desc='Validating'):
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total_correct += predicted.eq(targets).sum().item()

            for c in range(len(loader.dataset.classes)):
                class_correct[c] += ((predicted == targets) & (targets == c)).sum().item()
                class_total[c] += (targets == c).sum().item()

    print("\n验证集各类别准确率:")
    for i, (c, t) in enumerate(zip(class_correct, class_total)):
        print(f"{loader.dataset.classes[i]}: {c}/{t} = {100. * c / t:.1f}%")

    return total_loss / len(loader), 100. * total_correct / len(loader.dataset)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not torch.cuda.is_available():
        print("警告: 未检测到可用GPU，将使用CPU进行训练，这会非常慢！")
    print(f"使用设备: {device} | Batch Size: {batch_size}")

    # 数据加载
    train_data, val_data = create_data_loaders(
        'D:/tomato_split/train',
        'D:/tomato_split/val'
    )
    train_loader = create_loader(train_data, batch_size)
    val_loader = create_loader(val_data, batch_size, shuffle=False)

    print("🔄 初始化全新模型...")
    model = MobileNetV1_ShuffleAtt(num_classes=len(train_data.classes)).to(device)

    model.apply(init_weights)

    print(f"总参数量: {sum(p.numel() for p in model.parameters()):,}")

    print("🧪 测试初始随机权重准确率...")
    model.eval()
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            test_correct += predicted.eq(targets).sum().item()
            test_total += targets.size(0)
            if test_total > 100:
                break
    initial_acc = 100. * test_correct / test_total
    print(f"初始随机权重准确率: {initial_acc:.1f}%")

    if initial_acc > 30:
        print("⚠️ 警告：初始准确率过高，可能存在权重残留！")

    # 优化器配置
    optimizer = optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-4
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    history = {
        'epoch': [],
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'learning_rate': []
    }

    csv_path = 'D:/pycharm project/MobileNet v1/train_results_2.3.csv'

    # 早停策略
    patience = 10
    no_improve_epochs = 0
    best_acc = 0.0
    min_epochs = 50

    # 训练循环
    print("\n🚀 开始训练...")
    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()

        # 记录训练历史
        current_lr = optimizer.param_groups[0]['lr']
        history['epoch'].append(epoch + 1)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['learning_rate'].append(current_lr)

        pd.DataFrame(history).to_csv(csv_path, index=False)

        print(f"Epoch {epoch + 1:02d} | "
              f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.1f}% | "
              f"Val Loss: {val_loss:.4f} | Acc: {val_acc:.1f}% | "
              f"LR: {current_lr:.6f}")
        print(f"📊 训练记录已保存至: {csv_path}")

        # 保存最佳模型
        if val_acc > best_acc:
            improvement = val_acc - best_acc
            best_acc = val_acc
            no_improve_epochs = 0
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'model_config': {
                    'num_classes': len(train_data.classes),
                    'label_smoothing': 0.1,
                    'G': 8,
                },
                'optimizer_state': optimizer.state_dict(),
                'batch_size': batch_size,
                'accuracy': val_acc,
            }, 'best_model_v2.pth')
            print(f"✨ 保存最佳模型，准确率: {val_acc:.2f}% (提升: +{improvement:.2f}%)")
        else:
            no_improve_epochs += 1
            print(f"⏳ 连续 {no_improve_epochs}/{patience} 个epoch未提升")
        if epoch >= min_epochs and no_improve_epochs >= patience:
            print(f"🛑 早停触发：训练{epoch+1}轮后连续{patience}个epoch无提升")
            break
        elif no_improve_epochs >= patience:
            print(f"⏰ 已达到早停条件，但未达到最小训练轮数{min_epochs}，继续训练...")

    print(f"\n🎯 训练完成！最终记录保存在: {csv_path}")
    print(f"最佳验证准确率: {best_acc:.2f}%")


if __name__ == '__main__':
    main()
