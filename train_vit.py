import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts
from tqdm import tqdm
from vit import ViT, PerformerViT, LearnableKernel
import time
import logging
import csv
import pandas as pd
import os
from PIL import Image
from sklearn.preprocessing import LabelEncoder
import argparse
from tqdm import tqdm
## nohup python train_vit.py --model_type vit --patch_size 4 --num_epochs 100 --dropout 0.1 --emb_dropout 0.1 --csv_file vit_original.csv --batch_size 256 --learning_rate 0.0008 --model_name vit_model.pth --qkv_bias > vit4.log 2>&1 &

# nohup python train_vit.py --model_type performer --patch_size 4 --num_epochs 100 --dropout 0.1 --emb_dropout 0.1 --kernel_fn relu --generalized_attention True --nb_features 128 --csv_file performer_relu.csv --batch_size 256 --learning_rate 0.0008 --model_name performer_relu_model.pth --qkv_bias > performer-relu.log 2>&1 &

# nohup python train_vit.py --model_type performer --patch_size 4 --num_epochs 100 --dropout 0.1 --emb_dropout 0.1 --kernel_fn exp --nb_features 128 --csv_file  performer_exp.csv --batch_size 256 --learning_rate 0.0008 --model_name performer_exp_model.pth --qkv_bias > performer-exp.log 2>&1 &

# nohup python train_vit.py --model_type performer --patch_size 4 --num_epochs 250 --dropout 0.1 --emb_dropout 0.1 --kernel_fn learnable --nb_features 128 --csv_file  performer_fvariant.csv --batch_size 256 --learning_rate 0.0008 --model_name performer_fvariant_model.pth --qkv_bias > performer-fvariant.log 2>&1 &



def parse_args():
    parser = argparse.ArgumentParser(description="Model Training Configuration")
    parser.add_argument("--model_type", type=str, default="vit", choices=["vit", "performer"], help="Model type: vit or performer")
    parser.add_argument("--image_size", type=int, default=32, help="Input image size")
    parser.add_argument("--patch_size", type=int, default=4, help="Patch size")
    parser.add_argument("--num_classes", type=int, default=10, help="Number of output classes")
    parser.add_argument("--dim", type=int, default=128, help="Embedding dimension")
    parser.add_argument("--depth", type=int, default=6, help="Number of Transformer layers")
    parser.add_argument("--heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--mlp_dim", type=int, default=256, help="MLP intermediate dimension")
    parser.add_argument("--pool", type=str, default="cls", choices=["cls", "mean"], help="Pooling method")
    parser.add_argument("--channels", type=int, default=3, help="Number of input channels")
    parser.add_argument("--dim_head", type=int, default=32, help="Dimension of each attention head")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--emb_dropout", type=float, default=0.1, help="Embedding dropout rate")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training and testing")
    parser.add_argument("--num_epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--epoch_step", type=int, default=5, help="Number of epochs after which to test")
    parser.add_argument("--csv_file", type=str, default="training_log.csv", help="Path to save the CSV log file")
    parser.add_argument("--model_name", type=str, default="model.pth", help="File name to save the trained model")
    # Performer-specific arguments
    parser.add_argument("--nb_features", type=int, default=None, help="Number of random features for Performer")
    parser.add_argument("--generalized_attention", type=bool, default=False, help="Use generalized attention in Performer")
    parser.add_argument("--kernel_fn", type=str, default="relu", choices=["relu", "exp", "learnable"], help="Kernel function for Performer")
    parser.add_argument("--no_projection", action="store_true", help="Disable projection in Performer")
    parser.add_argument("--qkv_bias", action="store_true", help="Add bias to QKV projections")

    return parser.parse_args()

def create_model(args, device):
    if args.model_type == "vit":
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
            emb_dropout=args.emb_dropout,
            qkv_bias=args.qkv_bias
        )
    elif args.model_type == "performer":
        if args.kernel_fn == "relu":
            kernel_fn = torch.nn.ReLU()
            generalized_attention = True
        elif args.kernel_fn == "learnable":
            kernel_fn = LearnableKernel(args.nb_features)
            generalized_attention = True
        else:
            kernel_fn = None  # default Softmax Kernel
            generalized_attention = False

        model = PerformerViT(
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
            emb_dropout=args.emb_dropout,
            nb_features=args.nb_features,
            generalized_attention=generalized_attention,
            kernel_fn=kernel_fn,
            no_projection=args.no_projection,
            qkv_bias=args.qkv_bias
        )
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")

    return model.to(device)


def train(model, train_loader, test_loader, optimizer, scheduler, criterion, device, num_epochs=30, epoch_step=5, csv_file="training_vit.csv"):
    model.train()
    epoch_times = []
    total_start_time = time.time()
    test_results = []


    with open(csv_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Epoch", "Train Loss", "Train Accuracy", "Train Time (s)", "Test Loss", "Test Accuracy", "Inference Time (s)", "Learning Rate"])

    for epoch in range(num_epochs):
        epoch_start_time = time.time()
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


            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)


            loop.set_postfix(loss=loss.item(), accuracy=100. * correct / total)


        if scheduler is not None:
            scheduler.step()

        epoch_end_time = time.time()
        epoch_time = epoch_end_time - epoch_start_time
        epoch_times.append(epoch_time)


        epoch_loss = total_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch + 1}/{num_epochs}: Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}%, Time: {epoch_time:.4f}s, LR: {current_lr:.6f}")
        logging.info(
            f"Epoch {epoch + 1}/{num_epochs}: Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}%, Time: {epoch_time:.4f}s, LR: {current_lr:.6f}")
        for name, param in model.named_parameters():
            if 'kernel_fn' in name:
                if param.grad is not None:
                    logging.debug(f"Parameter: {name}, Grad Norm: {param.grad.norm()}")
                else:
                    logging.warning(f"Parameter: {name} has no grad")

        def log_gpu_memory_usage():
            memory_allocated = torch.cuda.memory_allocated(device)
            max_memory_allocated = torch.cuda.max_memory_allocated(device)
            memory_reserved = torch.cuda.memory_reserved(device)
            max_memory_reserved = torch.cuda.max_memory_reserved(device)

            print(f"Memory Allocated: {memory_allocated / 1024 ** 2:.2f} MiB")
            print(f"Max Memory Allocated: {max_memory_allocated / 1024 ** 2:.2f} MiB")
            print(f"Memory Reserved: {memory_reserved / 1024 ** 2:.2f} MiB")
            print(f"Max Memory Reserved: {max_memory_reserved / 1024 ** 2:.2f} MiB")

        log_gpu_memory_usage()

        test_loss = None
        test_acc = None
        inference_time = None


        if (epoch + 1) % epoch_step == 0 or (epoch + 1) == num_epochs:
            test_loss, test_acc, inference_time = test(model, test_loader, criterion, device)
            test_results.append((epoch + 1, test_loss, test_acc, inference_time))
            logging.info(
                f"Epoch {epoch + 1}/{num_epochs} - Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}%, Inference Time: {inference_time:.4f}s")


        with open(csv_file, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(
                [epoch + 1, epoch_loss, epoch_acc, epoch_time, test_loss, test_acc, inference_time, current_lr])

    total_end_time = time.time()
    total_training_time = total_end_time - total_start_time


    print(f"Total Training Time: {total_training_time:.4f}s")
    logging.info(f"Total Training Time: {total_training_time:.4f}s")

    return epoch_times, total_training_time, test_results


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

    logging.basicConfig(level=logging.DEBUG, handlers=[file_handler, stream_handler])

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

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    logging.info(f"Using device: {device}")

    model = create_model(args, device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=3e-5)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=5e-5)
    logging.info("Starting training...")
    logging.info(f"Configuration: {vars(args)}")
    epoch_times, total_training_time, test_results = train(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        scheduler=scheduler,

        criterion=criterion,
        device=device,
        num_epochs=args.num_epochs,
        epoch_step=args.epoch_step,
        csv_file=args.csv_file
    )
    with open(args.csv_file, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([])
        writer.writerow(["Args Information"])
        for key, value in vars(args).items():
            writer.writerow([key, value])

    logging.info("Starting testing...")
    test_loss, test_acc, inference_time = test(model, test_loader, criterion, device)
    logging.info(f"Final Test Loss: {test_loss:.4f}, Final Test Accuracy: {test_acc:.4f}%")

    model_path = args.model_name
    torch.save(model.state_dict(), model_path)
    logging.info(f"Model saved as {model_path}")

    print(f"Total Training Time: {total_training_time:.4f}s")
    print(f"Inference Time: {inference_time:.4f}s")
    logging.info(f"Total Training Time: {total_training_time:.4f}s")
    logging.info(f"Inference Time: {inference_time:.4f}s")
    for epoch, test_loss, test_acc, inference_time in test_results:
        logging.info(
            f"Epoch {epoch}: Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}%, Inference Time: {inference_time:.4f}s")

