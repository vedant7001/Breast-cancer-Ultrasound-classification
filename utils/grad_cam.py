import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import torchvision.transforms as transforms


def apply_gradcam(model, img_path, target_layer, target_class=None, device=None, input_size=(224, 224)):
    """
    Apply Grad-CAM to visualize which parts of the image the model focuses on
    
    Args:
        model (torch.nn.Module): Model to visualize
        img_path (str): Path to the input image
        target_layer: Target layer for visualization
        target_class (int, optional): Target class index (default: None, uses predicted class)
        device (str, optional): Device to use ('cuda' or 'cpu')
        input_size (tuple): Input image size
        
    Returns:
        tuple: (original_image, grad_cam_image, class_idx, class_prob)
    """
    try:
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.image import show_cam_on_image
    except ImportError:
        print("pytorch-grad-cam package is required for Grad-CAM visualization.")
        print("Install it with: pip install pytorch-grad-cam")
        return None, None, None, None
    
    # Set device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)
    
    # Load and preprocess image
    img = Image.open(img_path).convert('RGB')
    preprocess = transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = preprocess(img).unsqueeze(0).to(device)
    
    # Save original image for overlay
    orig_img = np.array(Image.open(img_path).convert('RGB').resize(input_size)) / 255.0
    
    # Move model to device
    model = model.to(device)
    model.eval()
    
    # Forward pass to get class prediction if not provided
    if target_class is None:
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            class_idx = torch.argmax(probabilities, dim=1).item()
            class_prob = probabilities[0, class_idx].item()
    else:
        class_idx = target_class
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            class_prob = probabilities[0, class_idx].item()
    
    # Initialize GradCAM
    cam = GradCAM(model=model, target_layers=[target_layer], use_cuda=device.type == 'cuda')
    
    # Generate heatmap
    grayscale_cam = cam(input_tensor=input_tensor, target_category=class_idx)
    grayscale_cam = grayscale_cam[0, :]
    
    # Overlay heatmap on original image
    visualization = show_cam_on_image(orig_img, grayscale_cam, use_rgb=True)
    
    return orig_img, visualization, class_idx, class_prob


def visualize_multiple_samples(model, img_paths, target_layer, class_names=None, device=None, save_dir=None):
    """
    Apply Grad-CAM to visualize multiple samples
    
    Args:
        model (torch.nn.Module): Model to visualize
        img_paths (list): List of image paths
        target_layer: Target layer for visualization
        class_names (list, optional): List of class names
        device (str, optional): Device to use
        save_dir (str, optional): Directory to save visualizations
        
    Returns:
        None
    """
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    plt.figure(figsize=(20, 5 * len(img_paths)))
    
    for i, img_path in enumerate(img_paths):
        # Apply Grad-CAM
        orig_img, grad_cam_img, class_idx, class_prob = apply_gradcam(
            model, img_path, target_layer, device=device
        )
        
        if orig_img is None:
            continue
        
        # Get class name
        if class_names and class_idx < len(class_names):
            class_name = class_names[class_idx]
        else:
            class_name = f"Class {class_idx}"
        
        # Plot original and Grad-CAM images
        plt.subplot(len(img_paths), 2, i*2+1)
        plt.imshow(orig_img)
        plt.title(f"Original: {os.path.basename(img_path)}")
        plt.axis('off')
        
        plt.subplot(len(img_paths), 2, i*2+2)
        plt.imshow(grad_cam_img)
        plt.title(f"Grad-CAM: {class_name} ({class_prob:.3f})")
        plt.axis('off')
    
    plt.tight_layout()
    
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'gradcam_visualization.png'), dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def get_target_layer(model, model_name):
    """
    Get the appropriate target layer for a model
    
    Args:
        model (torch.nn.Module): Model to extract layer from
        model_name (str): Name of the model architecture
        
    Returns:
        torch.nn.Module: Target layer for Grad-CAM
    """
    # For DenseNet models
    if model_name.lower().startswith('densenet'):
        # Last denselayer in the last dense block
        return model.features[-1]
    
    # For ResNet models
    elif model_name.lower().startswith('resnet'):
        # Last residual layer
        return model.conv_layers[-2]
    
    # For EfficientNet models
    elif model_name.lower().startswith('efficientnet'):
        # Last convolutional layer
        return list(model.features.children())[-2]
    
    else:
        raise ValueError(f"Unsupported model architecture: {model_name}")


if __name__ == "__main__":
    # Example usage
    import torch
    from models import get_model
    
    # Load model
    model = get_model('densenet', num_classes=2, pretrained=True)
    
    # Get target layer
    target_layer = get_target_layer(model, 'densenet')
    
    # Example image paths
    img_paths = [
        "path/to/image1.jpg",
        "path/to/image2.jpg"
    ]
    
    # Apply Grad-CAM to multiple images
    visualize_multiple_samples(
        model=model,
        img_paths=img_paths,
        target_layer=target_layer,
        class_names=["Normal", "Abnormal"],
        save_dir="results/gradcam"
    ) 