import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from vit import ViT
import time
import logging
import csv
import argparse
from tqdm import tqdm
##nohup python train_vit.py --patch_size 4 --num_epochs 50 --dropout 0.1 --emb_dropout 0.1 --csv_file vit_patch4.csv --batch_size 64 --learning_rate 0.0003 > vit4.log 2>&1 &

##nohup python train_vit.py --patch_size 8 --num_epochs 50 --dropout 0.1 --emb_dropout 0.1 --csv_file vit_a10_patch8.csv --batch_size 128 --learning_rate 0.0003 > vit8.log 2>&1 &


def parse_args():
    parser = argparse.ArgumentParser(description="ViT Training Configuration")
    parser.add_argument("--image_size", type=int, default=32, help="Input image size")
    parser.add_argument("--patch_size", type=int, default=4, help="Patch size")
    parser.add_argument("--num_classes", type=int, default=10, help="Number of output classes")
    parser.add_argument("--dim", type=int, default=128, help="Embedding dimension")
    parser.add_argument("--depth", type=int, default=6, help="Number of Transformer layers")
    parser.add_argument("--heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--mlp_dim", type=int, default=256, help="MLP intermediate dimension")
    parser.add_argument("--pool", type=str, default="cls", choices=["cls", "mean"], help="Pooling method")
    parser.add_argument("--channels", type=int, default=3, help="Number of input channels")
    parser.add_argument("--dim_head", type=int, default=64, help="Dimension of each attention head")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--emb_dropout", type=float, default=0.1, help="Embedding dropout rate")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training and testing")
    parser.add_argument("--num_epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--epoch_step", type=int, default=5, help="Number of epochs after which to test")
    parser.add_argument("--csv_file", type=str, default="training_vit.csv", help="Path to save the CSV log file")
    return parser.parse_args()

def train(model, train_loader, test_loader, optimizer, criterion, device, num_epochs=30, epoch_step=5, csv_file="training_vit.csv"):
    model.train()
    epoch_times = []  # 用于记录每个 epoch 的时间
    total_start_time = time.time()  # 总训练时间开始计时
    test_results = []  # 用于记录测试结果（测试集的 loss 和 accuracy）

    # 初始化 CSV 文件
    with open(csv_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Epoch", "Train Loss", "Train Accuracy", "Train Time (s)", "Test Loss", "Test Accuracy", "Inference Time (s)"])

    for epoch in range(num_epochs):
        epoch_start_time = time.time()  # 当前 epoch 开始计时
        total_loss = 0
        correct = 0
        total = 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # 统计
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

            # 更新进度条
            loop.set_postfix(loss=loss.item(), accuracy=100. * correct / total)

        epoch_end_time = time.time()  # 当前 epoch 结束计时
        epoch_time = epoch_end_time - epoch_start_time
        epoch_times.append(epoch_time)

        # 打印和记录日志
        epoch_loss = total_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        print(f"Epoch {epoch+1}/{num_epochs}: Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}%, Time: {epoch_time:.4f}s")
        logging.info(f"Epoch {epoch+1}/{num_epochs}: Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}%, Time: {epoch_time:.4f}s")

        # 初始化测试相关变量
        test_loss = None
        test_acc = None
        inference_time = None

        # 每隔 epoch_step 测试一次模型
        if (epoch + 1) % epoch_step == 0 or (epoch + 1) == num_epochs:
            test_loss, test_acc, inference_time = test(model, test_loader, criterion, device)
            test_results.append((epoch + 1, test_loss, test_acc, inference_time))
            logging.info(f"Epoch {epoch+1}/{num_epochs} - Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}%, Inference Time: {inference_time:.4f}s")

        # 将结果写入 CSV 文件
        with open(csv_file, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([epoch + 1, epoch_loss, epoch_acc, epoch_time, test_loss, test_acc, inference_time])

    total_end_time = time.time()  # 总训练时间结束计时
    total_training_time = total_end_time - total_start_time

    # 打印和记录总训练时间
    print(f"Total Training Time: {total_training_time:.4f}s")
    logging.info(f"Total Training Time: {total_training_time:.4f}s")

    return epoch_times, total_training_time, test_results

# **测试函数**
def test(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    inference_start_time = time.time()

    with torch.no_grad():
        loop = tqdm(test_loader, desc="Testing")
        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            # 统计
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

    inference_end_time = time.time()
    inference_time = inference_end_time - inference_start_time

    test_loss = total_loss / len(test_loader)
    test_acc = 100. * correct / total
    print(f"Test Loss: {test_loss:.4f}, Accuracy: {test_acc:.4f}%, Inference Time: {inference_time:.4f}s")
    logging.info(f"Test Loss: {test_loss:.4f}, Accuracy: {test_acc:.4f}%, Inference Time: {inference_time:.4f}s")

    return test_loss, test_acc, inference_time

if __name__ == "__main__":
    args = parse_args()

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    file_handler = logging.FileHandler("training_vit.log")
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logging.basicConfig(level=logging.INFO, handlers=[file_handler, stream_handler])

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    test_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=False, transform=train_transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=False, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    logging.info(f"Using device: {device}")


    model = ViT(
    image_size=args.image_size,
    patch_size=args.patch_size,
    num_classes=args.num_classes,
    dim=args.dim,
    depth=args.depth,
    heads=args.heads,
    mlp_dim=args.mlp_dim,
    pool=args.pool,
    channels=args.channels,
    dim_head=args.dim_head,
    dropout=args.dropout,
    emb_dropout=args.emb_dropout
).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)

    logging.info("Starting training...")
    logging.info(f"Configuration: {vars(args)}")
    epoch_times, total_training_time, test_results = train(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        num_epochs=args.num_epochs,
        epoch_step=args.epoch_step,
        csv_file=args.csv_file
    )


    logging.info("Starting testing...")
    test_loss, test_acc, inference_time = test(model, test_loader, criterion, device)
    logging.info(f"Final Test Loss: {test_loss:.4f}, Final Test Accuracy: {test_acc:.4f}%")

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    model_path = f"model_vit_{args.csv_file.split('.')[0]}_{timestamp}.pth"
    torch.save(model, model_path)
    logging.info(f"Model saved as {model_path}")
    for i, t in enumerate(epoch_times):
        print(f"Epoch {i+1} Training Time: {t:.4f}s")
        logging.info(f"Epoch {i+1} Training Time: {t:.4f}s")

    print(f"Total Training Time: {total_training_time:.4f}s")
    print(f"Inference Time: {inference_time:.4f}s")
    logging.info(f"Total Training Time: {total_training_time:.4f}s")
    logging.info(f"Inference Time: {inference_time:.4f}s")
    for epoch, test_loss, test_acc, inference_time in test_results:
        logging.info(
            f"Epoch {epoch}: Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}%, Inference Time: {inference_time:.4f}s")

