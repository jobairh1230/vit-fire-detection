"""
ViT Fire & Smoke Detection — Inference Script
Author: Jobair Hossain

Usage:
    python src/inference.py --image path/to/image.jpg --weights path/to/vit_base_best.pth
"""
import argparse
import torch
from PIL import Image
from torchvision import transforms
from transformers import ViTImageProcessor, ViTForImageClassification

CLASS_NAMES = ['fire', 'neutral', 'smoke']
MODEL_NAME = 'google/vit-base-patch16-224'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_model(model_path):
    model = ViTForImageClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(CLASS_NAMES),
        ignore_mismatched_sizes=True,
        id2label={i: n for i, n in enumerate(CLASS_NAMES)},
        label2id={n: i for i, n in enumerate(CLASS_NAMES)},
    )
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    return model.to(DEVICE)


def build_transform():
    """Use ViT processor's exact normalization — don't hardcode mean/std."""
    proc = ViTImageProcessor.from_pretrained(MODEL_NAME)
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=proc.image_mean, std=proc.image_std),
    ])


def preprocess(image_path, transform):
    img = Image.open(image_path).convert('RGB')
    return transform(img).unsqueeze(0)


def predict(model, image_tensor):
    with torch.no_grad():
        outputs = model(pixel_values=image_tensor.to(DEVICE))
        probs = torch.softmax(outputs.logits, dim=1)[0]
        pred_idx = probs.argmax().item()
    return CLASS_NAMES[pred_idx], float(probs[pred_idx]), probs.cpu().tolist()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', required=True, help='Path to input image')
    parser.add_argument('--weights', required=True, help='Path to model weights (.pth)')
    args = parser.parse_args()

    model = load_model(args.weights)
    transform = build_transform()
    tensor = preprocess(args.image, transform)
    label, confidence, all_probs = predict(model, tensor)

    print(f"\nPrediction: {label.upper()}")
    print(f"Confidence: {confidence:.1%}")
    print("\nAll probabilities:")
    for name, prob in zip(CLASS_NAMES, all_probs):
        print(f"  {name}: {prob:.4f}")
