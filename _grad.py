import torch
import torch.nn as nn
import timm
from torchvision import transforms
import gradio as gr
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io


class HybridCNNTransformer(nn.Module):
    """
    Hybrid CNN + Transformer model for image classification.

    - CNN Backbone: EfficientNet-B0 pretrained on ImageNet for feature extraction.
    - Projection: 1x1 convolution to convert CNN features to embeddings for Transformer.
    - Transformer Encoder: stacks multi-head self-attention layers.
    - Classification head: Dense layer on CLS token output for final classes.

    You can customize this model by:
    - Adding dropout or normalization layers after CNN backbone.
    - Altering Transformer encoder parameters (num_layers, nhead, dropout).
    - Modifying classification head with more layers or activation.
    """

    def __init__(self, num_classes):
        super(HybridCNNTransformer, self).__init__()

        # 1. CNN Backbone - remove final classifier, keep feature extractor.
        self.cnn_backbone = timm.create_model(
            "efficientnet_b0", pretrained=False, num_classes=0, global_pool=""
        )
        cnn_feature_dim = self.cnn_backbone.num_features

        # add extra features under this one as discussed in the meeting lol
        # self.norm = nn.BatchNorm2d(cnn_feature_dim)
        # self.extra_conv = nn.Conv2d(cnn_feature_dim, cnn_feature_dim, kernel_size=3, padding=1)
        # self.cnn_dropout = nn.Dropout2d(0.2)

        # 2. Project CNN feature maps into transformer embedding dimension.
        embed_dim = 256
        self.projection = nn.Conv2d(cnn_feature_dim, embed_dim, kernel_size=1)

        # 3. Transformer Encoder setup
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=8, batch_first=True, dropout=0.2
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)

        # Learnable [CLS] token prepended to the sequence
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Dropout before classification head
        self.dropout = nn.Dropout(0.5)

        # Final classification layer
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        # propagation ✨
        x = self.cnn_backbone(x)
        # referring to the thing above, also add layers here coz why not
        # x = self.norm(x)
        # x = self.extra_conv(x)
        # x = self.cnn_dropout(x)

        # Project CNN features to transformer dimension
        x = self.projection(x)

        # Flatten spatial dimensions to sequence tokens
        b, c, h, w = x.shape
        x = x.flatten(2).permute(0, 2, 1)  # Shape: (B, Seq_len, Emb_dim)

        # Prepend CLS token to sequence (for global representation)
        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # Transformer Encoder forward pass
        x = self.transformer_encoder(x)

        # Extract CLS token output
        cls_output = x[:, 0]
        cls_output = self.dropout(cls_output)

        # Classification head produces final logits
        output = self.fc(cls_output)
        return output


class GradCAM:
    """
    Grad-CAM implementation for CNN feature maps.
    Since your model has a hybrid architecture, we'll apply Grad-CAM to the CNN backbone features.
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.feature_maps = None
        self.gradients = None

        # Register hooks
        self.target_layer.register_forward_hook(self.save_feature_maps)
        self.target_layer.register_full_backward_hook(self.save_gradients)

    def save_feature_maps(self, module, input, output):
        self.feature_maps = output

    def save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_cam(self, input_tensor, target_class):
        """Generate CAM for the target class"""
        # Forward pass
        output = self.model(input_tensor)

        # Zero gradients
        self.model.zero_grad()

        # Backward pass for target class
        target_score = output[0, target_class]
        target_score.backward(retain_graph=True)

        # Get gradients and feature maps
        gradients = self.gradients.detach().cpu()
        feature_maps = self.feature_maps.detach().cpu()

        # Calculate weights (global average pooling of gradients)
        weights = torch.mean(gradients, dim=[2, 3], keepdim=True)

        # Generate CAM
        cam = torch.sum(weights * feature_maps, dim=1).squeeze()

        # Apply ReLU and normalize
        cam = torch.relu(cam)
        cam = cam / (torch.max(cam) + 1e-7)  # Normalize to [0, 1]

        return cam.numpy()


def create_heatmap_overlay(original_image, cam, alpha=0.4):
    """
    Create heatmap overlay on original image
    """
    # Resize CAM to match original image size
    h, w = original_image.size[1], original_image.size[0]
    cam_resized = cv2.resize(cam, (w, h))

    # Convert to heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # Convert PIL to numpy
    original_np = np.array(original_image)

    # Create overlay
    overlay = heatmap * alpha + original_np * (1 - alpha)
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    return Image.fromarray(overlay)


# --- Setup and Configuration ---
class_names = ["Apple_Black_Rot", "Apple_Cedar_Rust", "Apple_Healthy", "Apple_Scab"]
IMG_SIZE = 224
MODEL_PATH = "best_model.pth"

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the same transformations used for validation/testing
val_test_transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# For Grad-CAM visualization (without normalization)
gradcam_transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
    ]
)

# --- Load the Model ---
model = HybridCNNTransformer(num_classes=len(class_names)).to(device)

# Handle different checkpoint formats
checkpoint = torch.load(MODEL_PATH, map_location=device)
if isinstance(checkpoint, dict):
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
else:
    model.load_state_dict(checkpoint)

model.eval()

# Initialize Grad-CAM - targeting the last conv layer of CNN backbone
# For EfficientNet, we'll use the conv_head layer
target_layer = model.cnn_backbone.conv_head
grad_cam = GradCAM(model, target_layer)


def predict(image):
    """
    Original prediction function - takes a PIL image and returns predictions
    """
    if image is None:
        return None

    # Apply transformations
    image_tensor = val_test_transform(image).unsqueeze(0).to(device)

    # Perform inference
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]

    # Create confidence dictionary
    confidences = {
        class_names[i]: float(probabilities[i]) for i in range(len(class_names))
    }

    return confidences


def predict_with_gradcam(image):
    """
    Enhanced prediction function with Grad-CAM visualization
    """
    if image is None:
        return None, None

    # Get basic predictions
    confidences = predict(image)

    # Prepare image for Grad-CAM
    gradcam_tensor = gradcam_transform(image).unsqueeze(0).to(device)
    gradcam_tensor.requires_grad_()

    # Get predicted class
    image_tensor = val_test_transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
    predicted_class = torch.argmax(probabilities).item()

    try:
        # Generate CAM
        cam = grad_cam.generate_cam(gradcam_tensor, predicted_class)

        # Create heatmap overlay
        heatmap_overlay = create_heatmap_overlay(image, cam)

        return confidences, heatmap_overlay

    except Exception as e:
        print(f"Grad-CAM generation failed: {e}")
        return confidences, None


# --- Create and Launch the Gradio Interface ---
# Create a tabbed interface with both basic and Grad-CAM functionality
with gr.Blocks(title="Plant Leaf Disease Classifier") as iface:
    gr.Markdown("# Plant Leaf Disease Classifier")
    gr.Markdown(
        "Upload an image of an apple leaf to classify its disease. This model can identify Black Rot, Cedar Rust, Scab, or Healthy leaves."
    )

    with gr.Tabs():
        # Basic prediction tab (original functionality)
        with gr.TabItem("Basic Prediction"):
            with gr.Row():
                with gr.Column():
                    image_input_basic = gr.Image(
                        type="pil", label="Upload a Leaf Image"
                    )

                with gr.Column():
                    prediction_output_basic = gr.Label(
                        num_top_classes=4, label="Predictions"
                    )

            # Auto-predict on image upload
            image_input_basic.change(
                fn=predict, inputs=image_input_basic, outputs=prediction_output_basic
            )

        # Grad-CAM analysis tab
        with gr.TabItem("Grad-CAM Analysis"):
            with gr.Row():
                with gr.Column():
                    image_input_cam = gr.Image(type="pil", label="Upload a Leaf Image")
                    analyze_btn = gr.Button("Analyze with Grad-CAM", variant="primary")

                with gr.Column():
                    prediction_output_cam = gr.Label(
                        num_top_classes=4, label="Predictions"
                    )
                    heatmap_output = gr.Image(
                        label="Grad-CAM Heatmap (Areas the model focuses on)",
                        type="pil",
                    )

            # Grad-CAM explanation
            gr.Markdown("""
            **Understanding the Heatmap:**
            - **Red areas**: Regions the model considers most important for its prediction
            - **Yellow areas**: Moderately important regions
            - **Blue/Green areas**: Less important regions
            - This shows where the CNN backbone focuses before passing features to the transformer
            """)

            # Event handler for Grad-CAM analysis
            analyze_btn.click(
                fn=predict_with_gradcam,
                inputs=image_input_cam,
                outputs=[prediction_output_cam, heatmap_output],
            )


# Launch the interface
if __name__ == "__main__":
    print("Launching Gradio interface... Go to the URL below in your browser.")
    iface.launch()
    """
    Hybrid CNN + Transformer model for image classification.

    - CNN Backbone: EfficientNet-B0 pretrained on ImageNet for feature extraction.
    - Projection: 1x1 convolution to convert CNN features to embeddings for Transformer.
    - Transformer Encoder: stacks multi-head self-attention layers.
    - Classification head: Dense layer on CLS token output for final classes.

    You can customize this model by:
    - Adding dropout or normalization layers after CNN backbone.
    - Altering Transformer encoder parameters (num_layers, nhead, dropout).
    - Modifying classification head with more layers or activation.
    """

    def __init__(self, num_classes):
        super(HybridCNNTransformer, self).__init__()

        # 1. CNN Backbone - remove final classifier, keep feature extractor.
        self.cnn_backbone = timm.create_model(
            "efficientnet_b0", pretrained=False, num_classes=0, global_pool=""
        )
        cnn_feature_dim = self.cnn_backbone.num_features

        # add extra features under this one as discussed in the meeting lol
        # self.norm = nn.BatchNorm2d(cnn_feature_dim)
        # self.extra_conv = nn.Conv2d(cnn_feature_dim, cnn_feature_dim, kernel_size=3, padding=1)
        # self.cnn_dropout = nn.Dropout2d(0.2)

        # 2. Project CNN feature maps into transformer embedding dimension.
        embed_dim = 256
        self.projection = nn.Conv2d(cnn_feature_dim, embed_dim, kernel_size=1)

        # 3. Transformer Encoder setup
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=8, batch_first=True, dropout=0.2
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)

        # Learnable [CLS] token prepended to the sequence
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Dropout before classification head
        self.dropout = nn.Dropout(0.5)

        # Final classification layer
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        # propagation ✨
        x = self.cnn_backbone(x)
        # referring to the thing above, also add layers here coz why not
        # x = self.norm(x)
        # x = self.extra_conv(x)
        # x = self.cnn_dropout(x)

        # Project CNN features to transformer dimension
        x = self.projection(x)

        # Flatten spatial dimensions to sequence tokens
        b, c, h, w = x.shape
        x = x.flatten(2).permute(0, 2, 1)  # Shape: (B, Seq_len, Emb_dim)

        # Prepend CLS token to sequence (for global representation)
        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # Transformer Encoder forward pass
        x = self.transformer_encoder(x)

        # Extract CLS token output
        cls_output = x[:, 0]
        cls_output = self.dropout(cls_output)

        # Classification head produces final logits
        output = self.fc(cls_output)
        return output


class GradCAM:
    """
    Grad-CAM implementation for CNN feature maps.
    Since your model has a hybrid architecture, we'll apply Grad-CAM to the CNN backbone features.
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.feature_maps = None
        self.gradients = None

        # Register hooks
        self.target_layer.register_forward_hook(self.save_feature_maps)
        self.target_layer.register_backward_hook(self.save_gradients)

    def save_feature_maps(self, module, input, output):
        self.feature_maps = output

    def save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_cam(self, input_tensor, target_class):
        """Generate CAM for the target class"""
        # Forward pass
        output = self.model(input_tensor)

        # Zero gradients
        self.model.zero_grad()

        # Backward pass for target class
        target_score = output[0, target_class]
        target_score.backward(retain_graph=True)

        # Get gradients and feature maps
        gradients = self.gradients.detach().cpu()
        feature_maps = self.feature_maps.detach().cpu()

        # Calculate weights (global average pooling of gradients)
        weights = torch.mean(gradients, dim=[2, 3], keepdim=True)

        # Generate CAM
        cam = torch.sum(weights * feature_maps, dim=1).squeeze()

        # Apply ReLU and normalize
        cam = torch.relu(cam)
        cam = cam / (torch.max(cam) + 1e-7)  # Normalize to [0, 1]

        return cam.numpy()


def create_heatmap_overlay(original_image, cam, alpha=0.4):
    """
    Create heatmap overlay on original image
    """
    # Resize CAM to match original image size
    h, w = original_image.size[1], original_image.size[0]
    cam_resized = cv2.resize(cam, (w, h))

    # Convert to heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # Convert PIL to numpy
    original_np = np.array(original_image)

    # Create overlay
    overlay = heatmap * alpha + original_np * (1 - alpha)
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    return Image.fromarray(overlay)


# --- Setup and Configuration ---
class_names = ["Apple_Black_Rot", "Apple_Cedar_Rust", "Apple_Healthy", "Apple_Scab"]
IMG_SIZE = 224
MODEL_PATH = "best_model.pth"

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the same transformations used for validation/testing
val_test_transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# For Grad-CAM visualization (without normalization)
gradcam_transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
    ]
)

# --- Load the Model ---
model = HybridCNNTransformer(num_classes=len(class_names)).to(device)

# Handle different checkpoint formats
checkpoint = torch.load(MODEL_PATH, map_location=device)
if isinstance(checkpoint, dict):
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
else:
    model.load_state_dict(checkpoint)

model.eval()

# Initialize Grad-CAM - targeting the last conv layer of CNN backbone
# For EfficientNet, we'll use the conv_head layer
target_layer = model.cnn_backbone.conv_head
grad_cam = GradCAM(model, target_layer)


def predict_with_gradcam(image, show_gradcam=True):
    """
    Takes a PIL image and returns predictions with optional Grad-CAM visualization
    """
    if image is None:
        return None, None

    # Prepare image for inference
    image_tensor = val_test_transform(image).unsqueeze(0).to(device)

    # Get predictions
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]

    # Create confidence dictionary
    confidences = {
        class_names[i]: float(probabilities[i]) for i in range(len(class_names))
    }

    if not show_gradcam:
        return confidences, None

    # Generate Grad-CAM for the predicted class
    predicted_class = torch.argmax(probabilities).item()

    # Use unnormalized image for Grad-CAM
    gradcam_tensor = gradcam_transform(image).unsqueeze(0).to(device)
    gradcam_tensor.requires_grad_()

    try:
        # Generate CAM
        cam = grad_cam.generate_cam(gradcam_tensor, predicted_class)

        # Create heatmap overlay
        heatmap_overlay = create_heatmap_overlay(image, cam)

        return confidences, heatmap_overlay

    except Exception as e:
        print(f"Grad-CAM generation failed: {e}")
        return confidences, None


def predict_basic(image):
    """Basic prediction without Grad-CAM for faster inference"""
    confidences, _ = predict_with_gradcam(image, show_gradcam=False)
    return confidences


def predict_with_heatmap(image):
    """Prediction with Grad-CAM heatmap"""
    confidences, heatmap = predict_with_gradcam(image, show_gradcam=True)
    return confidences, heatmap


# --- Create Gradio Interface with Tabs ---
with gr.Blocks(title="Plant Leaf Disease Classifier with Grad-CAM") as iface:
    gr.Markdown("# Plant Leaf Disease Classifier")
    gr.Markdown(
        "Upload an image of an apple leaf to classify its disease. The model can identify Black Rot, Cedar Rust, Scab, or Healthy leaves."
    )

    with gr.Tabs():
        # Basic Prediction Tab
        with gr.TabItem("Basic Prediction"):
            with gr.Row():
                with gr.Column():
                    image_input_basic = gr.Image(type="pil", label="Upload Leaf Image")
                    predict_btn_basic = gr.Button("Predict", variant="primary")

                with gr.Column():
                    prediction_output_basic = gr.Label(
                        num_top_classes=4, label="Disease Classification"
                    )

        # Grad-CAM Visualization Tab
        with gr.TabItem("Grad-CAM Analysis"):
            with gr.Row():
                with gr.Column():
                    image_input_cam = gr.Image(type="pil", label="Upload Leaf Image")
                    predict_btn_cam = gr.Button(
                        "Analyze with Grad-CAM", variant="primary"
                    )

                with gr.Column():
                    prediction_output_cam = gr.Label(
                        num_top_classes=4, label="Disease Classification"
                    )
                    heatmap_output = gr.Image(
                        label="Grad-CAM Heatmap (Areas the model focuses on)",
                        type="pil",
                    )

            gr.Markdown("""
            **Grad-CAM Interpretation:**
            - 🔴 Red areas: Regions the model considers most important for the prediction
            - 🟡 Yellow areas: Moderately important regions  
            - 🔵 Blue areas: Less important regions
            - The heatmap shows where the CNN backbone is focusing before the transformer processes the features
            """)

    # Event handlers
    predict_btn_basic.click(
        fn=predict_basic, inputs=image_input_basic, outputs=prediction_output_basic
    )

    predict_btn_cam.click(
        fn=predict_with_heatmap,
        inputs=image_input_cam,
        outputs=[prediction_output_cam, heatmap_output],
    )

    # Auto-predict on image upload for basic tab
    image_input_basic.change(
        fn=predict_basic, inputs=image_input_basic, outputs=prediction_output_basic
    )


# Launch the interface
if __name__ == "__main__":
    print("Launching Gradio interface with Grad-CAM visualization...")
    print("Go to the URL shown below in your browser.")
    iface.launch()
