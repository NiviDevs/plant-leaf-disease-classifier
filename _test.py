import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import os
import multiprocessing


def main():
    # 1. SETUP AND CONFIGURATION
    # ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ==============================================================================
    # UPDATED: Point this to the parent directory containing train, val, test folders
    # Your structure should be:
    # /data/
    #   ├── train/
    #   │   ├── Apple___Apple_scab/
    #   │   └── ...
    #   ├── val/
    #   │   └── ...
    #   └── test/
    #       └── ...
    # ==============================================================================
    DATA_ROOT = "."  # <-- CHANGE THIS if your 'data' folder is elsewhere

    TRAIN_DIR = os.path.join(DATA_ROOT, "data", "train")
    VAL_DIR = os.path.join(DATA_ROOT, "data", "val")
    TEST_DIR = os.path.join(DATA_ROOT, "data", "test")

    # Model parameters
    IMG_SIZE = 224
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-4
    EPOCHS = 20

    # 2. DATA PREPARATION (REVISED FOR SPLIT FOLDERS)
    # ---
    # Data augmentation for the training set
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # No augmentation for validation and testing sets
    val_test_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Create datasets for each split
    train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
    val_dataset = datasets.ImageFolder(VAL_DIR, transform=val_test_transform)
    test_dataset = datasets.ImageFolder(TEST_DIR, transform=val_test_transform)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2
    )

    class_names = train_dataset.classes
    print(f"Classes found: {class_names}")
    print(f"Training images: {len(train_dataset)}")
    print(f"Validation images: {len(val_dataset)}")
    print(f"Test images: {len(test_dataset)}")

    # 3. HYBRID MODEL DEFINITION (CNN + TRANSFORMER)
    # ---
    class HybridCNNTransformer(nn.Module):
        def __init__(self, num_classes):
            super(HybridCNNTransformer, self).__init__()
            self.cnn_backbone = timm.create_model(
                "efficientnet_b0", pretrained=True, num_classes=0, global_pool=""
            )
            cnn_feature_dim = self.cnn_backbone.num_features

            embed_dim = 256  # Transformer embedding dimension
            self.projection = nn.Conv2d(cnn_feature_dim, embed_dim, kernel_size=1)

            encoder_layer = nn.TransformerEncoderLayer(
                d_model=embed_dim, nhead=8, batch_first=True, dropout=0.2
            )
            self.transformer_encoder = nn.TransformerEncoder(
                encoder_layer, num_layers=4
            )

            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.dropout = nn.Dropout(0.5)
            self.fc = nn.Linear(embed_dim, num_classes)

        def forward(self, x):
            x = self.cnn_backbone(x)
            x = self.projection(x)
            b, c, h, w = x.shape
            x = x.flatten(2).permute(0, 2, 1)
            cls_tokens = self.cls_token.expand(b, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
            x = self.transformer_encoder(x)
            cls_output = x[:, 0]
            cls_output = self.dropout(cls_output)
            output = self.fc(cls_output)
            return output

    model = HybridCNNTransformer(num_classes=len(class_names)).to(device)

    # 4. TRAINING & VALIDATION LOOP
    # ---
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    best_val_accuracy = 0.0

    for epoch in range(EPOCHS):
        # --- Training Phase ---
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # --- Validation Phase ---
        model.eval()
        val_corrects = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                val_corrects += torch.sum(preds == labels.data)

        epoch_val_accuracy = val_corrects.double() / len(val_dataset)
        print(f"Epoch {epoch + 1}/{EPOCHS} | Val Acc: {epoch_val_accuracy:.4f}")

        # Save the model if it has the best validation accuracy so far
        if epoch_val_accuracy > best_val_accuracy:
            best_val_accuracy = epoch_val_accuracy
            torch.save(model.state_dict(), "best_model.pth")
            print("   -> Validation accuracy improved, model saved!")

    # 5. FINAL EVALUATION ON THE TEST SET
    # ---
    print("\n--- Loading best model for final testing ---")
    model.load_state_dict(torch.load("best_model.pth"))
    model.eval()

    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Print Classification Report
    print("\nClassification Report (Test Set):")
    print(
        classification_report(all_labels, all_preds, target_names=class_names, digits=4)
    )

    # Plot Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix (Test Set)")
    plt.show()


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
