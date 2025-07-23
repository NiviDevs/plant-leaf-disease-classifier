import torch
import torch.nn as nn
import timm
from torchvision import transforms
import gradio as gr


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

# --- 2. Setup and Configuration ---
# Define the same parameters and transformations as in your notebook
class_names = ["Apple_Black_Rot", "Apple_Cedar_Rust", "Apple_Healthy", "Apple_Scab"]
IMG_SIZE = 224
MODEL_PATH = (
    "best_model.pth"  # Make sure your saved model file is in the same directory
)

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

# --- 3. Load the Model ---
# Initialize the model architecture
model = HybridCNNTransformer(num_classes=len(class_names)).to(device)

# Load the saved weights
# Note: map_location=device ensures the model loads correctly whether you're on a CPU or GPU machine
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

# Set the model to evaluation mode
model.eval()


# --- 4. Prediction Function ---
# This function takes a PIL image and returns a dictionary of class confidences
def predict(image):
    """
    Takes a PIL image, preprocesses it, and returns a dictionary of
    class names to their predicted probabilities.
    """
    if image is None:
        return None

    # Gradio provides the image as a PIL Image object.
    # We apply the same transformations as our test set.
    image_tensor = val_test_transform(image).unsqueeze(0).to(device)

    # Perform inference
    with torch.no_grad():
        outputs = model(image_tensor)
        # Apply softmax to get probabilities
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]

    # Create a dictionary of class names and their confidence scores
    confidences = {
        class_names[i]: float(probabilities[i]) for i in range(len(class_names))
    }

    return confidences


# --- 5. Create and Launch the Gradio Interface ---
# Define the Gradio interface components
# Input: An image upload box
# Output: A label component to display the predictions
# We also provide a title, description, and some example images for users to try.
iface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload a Leaf Image"),
    outputs=gr.Label(num_top_classes=4, label="Predictions"),
    title="Plant Leaf Disease Classifier",
    description="Upload an image of an apple leaf to classify its disease. This model can identify Black Rot, Cedar Rust, Scab, or a Healthy leaf.",
    examples=[
        # Add paths to a few example images if you have them.
        # If not, you can remove the 'examples' list.
        # e.g., ['path/to/healthy_leaf.jpg', 'path/to/scab_leaf.jpg']
    ],
)

# Launch the web interface
if __name__ == "__main__":
    print("Launching Gradio interface... Go to the URL below in your browser.")
    iface.launch()
