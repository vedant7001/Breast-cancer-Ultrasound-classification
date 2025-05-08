import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class DenseNetModel(nn.Module):
    def __init__(self, num_classes=2, pretrained=True, model_variant='densenet121'):
        """
        DenseNet model for breast ultrasound classification
        
        Args:
            num_classes (int): Number of classes for classification
            pretrained (bool): Whether to use pretrained weights
            model_variant (str): DenseNet variant ('densenet121' or 'densenet169')
        """
        super(DenseNetModel, self).__init__()
        
        if model_variant == 'densenet121':
            base_model = models.densenet121(weights='IMAGENET1K_V1' if pretrained else None)
            num_features = 1024
        elif model_variant == 'densenet169':
            base_model = models.densenet169(weights='IMAGENET1K_V1' if pretrained else None)
            num_features = 1664
        else:
            raise ValueError(f"Model variant '{model_variant}' not supported. Use 'densenet121' or 'densenet169'.")
        
        self.features = base_model.features
        
        # Replace the classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(num_features, num_classes)
        )
        
    def forward(self, x):
        features = self.features(x)
        out = self.classifier(features)
        return out


class ResNetModel(nn.Module):
    def __init__(self, num_classes=2, pretrained=True, model_variant='resnet50'):
        """
        ResNet model for breast ultrasound classification
        
        Args:
            num_classes (int): Number of classes for classification
            pretrained (bool): Whether to use pretrained weights
            model_variant (str): ResNet variant ('resnet18', 'resnet50', or 'resnet101')
        """
        super(ResNetModel, self).__init__()
        
        if model_variant == 'resnet18':
            base_model = models.resnet18(weights='IMAGENET1K_V1' if pretrained else None)
            num_features = 512
        elif model_variant == 'resnet50':
            base_model = models.resnet50(weights='IMAGENET1K_V1' if pretrained else None)
            num_features = 2048
        elif model_variant == 'resnet101':
            base_model = models.resnet101(weights='IMAGENET1K_V1' if pretrained else None)
            num_features = 2048
        else:
            raise ValueError(f"Model variant '{model_variant}' not supported. Use 'resnet18', 'resnet50', or 'resnet101'.")
        
        # Remove the final fully connected layer
        self.conv_layers = nn.Sequential(*list(base_model.children())[:-1])
        
        # Add a new classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(num_features, num_classes)
        )
        
    def forward(self, x):
        features = self.conv_layers(x)
        out = self.classifier(features)
        return out


class EfficientNetModel(nn.Module):
    def __init__(self, num_classes=2, pretrained=True, model_variant='efficientnet_b0'):
        """
        EfficientNet model for breast ultrasound classification
        
        Args:
            num_classes (int): Number of classes for classification
            pretrained (bool): Whether to use pretrained weights
            model_variant (str): EfficientNet variant ('efficientnet_b0' or 'efficientnet_b3')
        """
        super(EfficientNetModel, self).__init__()
        
        if model_variant == 'efficientnet_b0':
            base_model = models.efficientnet_b0(weights='IMAGENET1K_V1' if pretrained else None)
            num_features = 1280
        elif model_variant == 'efficientnet_b3':
            base_model = models.efficientnet_b3(weights='IMAGENET1K_V1' if pretrained else None)
            num_features = 1536
        else:
            raise ValueError(f"Model variant '{model_variant}' not supported. Use 'efficientnet_b0' or 'efficientnet_b3'.")
        
        # Remove the final classifier
        self.features = nn.Sequential(*list(base_model.children())[:-1])
        
        # Add a new classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(num_features, num_classes)
        )
        
    def forward(self, x):
        features = self.features(x)
        out = self.classifier(features)
        return out


def get_model(model_name, num_classes=2, pretrained=True, variant=None):
    """
    Factory function to get a model by name
    
    Args:
        model_name (str): Name of the model ('densenet', 'resnet', or 'efficientnet')
        num_classes (int): Number of output classes
        pretrained (bool): Whether to use pretrained weights
        variant (str, optional): Specific variant of the model
        
    Returns:
        nn.Module: The model instance
    """
    if model_name.lower() == 'densenet':
        variant = variant or 'densenet121'
        return DenseNetModel(num_classes=num_classes, pretrained=pretrained, model_variant=variant)
    elif model_name.lower() == 'resnet':
        variant = variant or 'resnet50'
        return ResNetModel(num_classes=num_classes, pretrained=pretrained, model_variant=variant)
    elif model_name.lower() == 'efficientnet':
        variant = variant or 'efficientnet_b0'
        return EfficientNetModel(num_classes=num_classes, pretrained=pretrained, model_variant=variant)
    else:
        raise ValueError(f"Model '{model_name}' not recognized. Use 'densenet', 'resnet', or 'efficientnet'.")


if __name__ == '__main__':
    # Example usage
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create models
    densenet = get_model('densenet', num_classes=2, pretrained=True)
    resnet = get_model('resnet', num_classes=2, pretrained=True)
    efficientnet = get_model('efficientnet', num_classes=2, pretrained=True)
    
    # Move to device
    densenet.to(device)
    resnet.to(device)
    efficientnet.to(device)
    
    # Print model summaries
    print(f"DenseNet parameters: {sum(p.numel() for p in densenet.parameters() if p.requires_grad)}")
    print(f"ResNet parameters: {sum(p.numel() for p in resnet.parameters() if p.requires_grad)}")
    print(f"EfficientNet parameters: {sum(p.numel() for p in efficientnet.parameters() if p.requires_grad)}")
    
    # Test forward pass
    x = torch.randn(2, 3, 224, 224).to(device)
    print(f"DenseNet output shape: {densenet(x).shape}")
    print(f"ResNet output shape: {resnet(x).shape}")
    print(f"EfficientNet output shape: {efficientnet(x).shape}") 