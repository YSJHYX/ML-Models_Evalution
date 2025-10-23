import torch 
import torchvision.models as models
import torch.nn.functional as F
from torchvision import transforms as T
from PIL import Image
import os
from glob import glob
from natsort import natsorted 
import json
import numpy as np
import matplotlib.pyplot as plt

# Check device used
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device Used: ", device)

# data pre-processing
transform = T.Compose([T.Resize((224,224)), T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229,0.224,0.225])])

models_dict={
    'AlexNet': models.alexnet(pretrained=True).to(device).eval(),
    'VGG16': models.vgg16(pretrained=True).to(device).eval(),
    'ResNet50': models.resnet50(pretrained=True).to(device).eval(),
    'InceptionV3': models.inception_v3(pretrained=True).to(device).eval(),
    'DenseNet121': models.densenet121(pretrained=True).to(device).eval(),
    'MobileNetV2': models.mobilenet_v2(pretrained=True).to(device).eval()
}

def load_imageNet_class_index(json_path):
    with open(json_path, 'r') as f:
        class_index = json.load(f)
    return class_index


def load_ground_truth_labels(image_label_dir):
    labels = {}
    for fileName in os.listdir(image_label_dir):
        if fileName.endswith(('.png', '.jpg', '.JPEG', 'jpeg')):
            parts = fileName.split('_')
            if len(parts) >= 3:
                image_id = parts[0]
                ground_truth_id = parts[1]
                ground_truth_name = '_'.join(parts[2:]).split('.')[0]
                labels[image_id] = {
                    'id': ground_truth_id,
                    'name': ground_truth_name
                }
    return labels

# image class prediction function
def predict_image(model, image_path, transform, device, class_index, top_k = 5):
    model.eval()
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = F.softmax(outputs, dim=1)
        top_probs, top_idxs = probabilities.topk(top_k)

    top_probs = top_probs.cpu().numpy()[0]
    top_idxs = top_idxs.cpu().numpy()[0]

    predictions = []
    for i in range(top_k):
        idx = str(top_idxs[i])
        if idx in class_index:
            class_id, label = class_index[idx]
            predictions.append({
                'rank': i+1,
                'class_id': class_id,
                'label': label,
                'probability': float(top_probs[i])
            })
    return predictions

def evaluate_models(models_dict, image_folder, labels, class_index, device):
    results = {}
    for model_name, model in models_dict.items():
        print(f"\n正在评估 {model_name}...")
        print("Evaluation Process... ...")
        models_results = {
            'predictions': {},
            'top1_accuracy': 0,
            'top5_accuracy': 0,
            'detailed results': []
        }

        correct_top1 = 0
        correct_top5 = 0
        total = 0

        for image_id, true_label in labels.items():
            image_path = os.path.join(image_folder, f"{image_id}.png")
            if not os.path.exists(image_path):
                for ext in ['.JPEG', '.jpg']:
                    alt_path = os.path.join(image_folder, f"{image_id}{ext}")
                    if os.path.exists(alt_path):
                        image_path = alt_path
                        break
            if os.path.exists(image_path):
                predictions = predict_image(model, image_path, transform, device, class_index, top_k=5)
                models_results['predictions'][image_id] = predictions

                true_class_id = true_label['id']
                top1_pred = predictions[0]['class_id'] if predictions else None
                top5_pred = [pred['class_id'] for pred in predictions]

                if top1_pred == true_class_id:
                    correct_top1 += 1
                if true_class_id in top5_pred:
                    correct_top5 += 1
                total += 1

                models_results['detailed results'].append({
                    'image_id': image_id,
                    'true_label': true_label,
                    'predictions': predictions,
                    'top1_correct': top1_pred == true_class_id,
                    'top5_correct': true_class_id in top5_pred
                })

        models_results['top1_accuracy'] = correct_top1 / total if total > 0 else 0
        models_results['top5_accuracy'] = correct_top5 / total if total > 0 else 0

        results[model_name] = models_results

        print(f"{model_name} - Top-1准确率: {models_results['top1_accuracy']:.4f}, Top-5准确率: {models_results['top5_accuracy']:.4f}")
    return results


def dataVisualization(results):
    model_names = list(results.keys())
    top1_accuracies = [results[model]['top1_accuracy'] for model in model_names]
    top5_accuracies = [results[model]['top5_accuracy'] for model in model_names]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
    ax1.bar(model_names, top1_accuracies, color='skyblue', alpha=0.7)
    ax1.set_title('Top-1 Accuracy Comparision')
    ax1.set_ylabel('Accuracy')
    ax1.set_ylim(0,1)
    ax1.tick_params(axis='x', rotation = 45)

    ax2.bar(model_names, top5_accuracies, color='lightgreen', alpha=0.7)
    ax2.set_title('Top-5 Accuracy Comparision')
    ax2.set_ylabel('Accuracy')
    ax2.set_ylim(0,1)
    ax2.tick_params(axis='x', rotation = 45)


    plt.tight_layout()
    plt.savefig('model_performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    images_paths = 'images'
    image_label_dir = 'images_label'
    class_index_path = 'images_label/imagenet_class_index.json'

    print("Starting Models Evaluations... ...")
    print("Loading ImageNet Class Index... ...")
    class_index = load_imageNet_class_index(class_index_path)
    labels = load_ground_truth_labels(image_label_dir)
    print(f"Found {len(labels)} images with ground truth labels.")
    results = evaluate_models(models_dict, images_paths, labels, class_index, device)

    print("Genratting Data Visualization... ...")
    dataVisualization(results)
    print("Performance chart has been saved as 'model_performance_comparison.png'.")


    print("----------Models Evalution Summary----------")
    for model_name, result in results.items():
        print(f"{model_name}:")
        print(f"  Top-1准确率: {result['top1_accuracy']:.4f}")
        print(f"  Top-5准确率: {result['top5_accuracy']:.4f}")
        print()
    print("Evaluation Completed.")

if __name__ == '__main__':
    main()
