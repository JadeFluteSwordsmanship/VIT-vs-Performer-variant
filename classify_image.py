import torch
from torchvision import transforms
from PIL import Image
import argparse

# CIFAR-10 类别
CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]


# 定义数据预处理
def preprocess_image(image_path, image_size=32):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),  # 调整到指定大小
        transforms.ToTensor(),  # 转换为 Tensor
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 标准化
    ])
    image = Image.open(image_path).convert("RGB")  # 确保是 RGB 格式
    return transform(image).unsqueeze(0)  # 添加 batch 维度


# 加载模型
def load_model(model_path, device="cuda" if torch.cuda.is_available() else "cpu"):
    model = torch.load(model_path, map_location=device)
    model.eval()
    return model


# 进行分类
def classify_image(model, image_tensor):
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = outputs.max(1)
        return predicted.item()


def parse_args():
    parser = argparse.ArgumentParser(description="CIFAR-10 Image Classification")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the input image")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model")
    return parser.parse_args()


# 主函数
if __name__ == "__main__":
    args = parse_args()

    # 加载模型
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.model_path, device)

    # 预处理图片
    image_tensor = preprocess_image(args.image_path).to(device)

    # 分类
    predicted_label = classify_image(model, image_tensor)
    print(f"Predicted Label: {predicted_label}")
    print(f"Predicted Category: {CIFAR10_CLASSES[predicted_label]}")


# python classify_image.py --image_path dog.jpg --model_path model_vit_vit_a10_patch4_20241201_085823.pth