import torch
from PIL import Image
from torch import nn, save, load
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Normalize, Compose 
import matplotlib.pyplot as plt
from pathlib import Path

IMAGE_SIZE = 28
NUM_CLASSES = 10
BATCH_SIZE = 32 
LEARNING_RATE = 0.001 
NUM_EPOCHS = 10
MODEL_STATE_FILENAME = "model_state.pt"

transform = Compose([
    ToTensor(),
    Normalize((0.1307,), (0.3081,)) # Mean and Std Dev for MNIST
])

train_dataset = datasets.MNIST(root="data", download=True, train=True, transform=transform)
test_dataset = datasets.MNIST(root="data", download=True, train=False, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

class ImageClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            # layers
            nn.Conv2d(1, 32, kernel_size=(3, 3), padding=1), # 1 input, 32 filters, padding to keep 28x28
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)), # Output: (32, 14, 14)
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1), # Padding to keep 14x14
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)), # Output: (64, 7, 7)
            nn.Flatten(),
            # Correct flattened size after two 2x2 MaxPools: (28 / 2) / 2 = 7
            # output : 10 numbers (probability)
            nn.Linear(64 * 7 * 7, NUM_CLASSES),
        )
    def forward(self, x):
        return self.model(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
clf = ImageClassifier().to(device)
opt = Adam(clf.parameters(), lr=LEARNING_RATE)
loss_fn = nn.CrossEntropyLoss() 

def predict(model, image_path, device):
    model.eval() # Set model to evaluation mode
    try:
        image = Image.open(image_path).convert('L') # grayscale
        image_tensor = transform(image).unsqueeze(0).to(device) 

        with torch.no_grad():
            output = model(image_tensor)
            probabilities = torch.softmax(output, dim=1) 
            predicted_index = torch.argmax(probabilities).item()

        print(f"\n--- Prediction for {Path(image_path).name} ---")
        print("Network Output Activations (probability for each digit 0-9):")
        for i, val in enumerate(probabilities.flatten().cpu().numpy()):
            print(f"  Digit {i}: {val * 100:.2f}%")

        print(f"\nPredicted digit: {predicted_index}")

    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
    except Exception as e:
        print(f"Error during prediction: {e}")

def evaluate_model(model, data_loader, device, epoch=None):
    model.eval() 
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()

    accuracy = 100 * correct / total
    epoch_str = f"Epoch {epoch}: " if epoch is not None else ""
    print(f"{epoch_str}Test Accuracy: {accuracy:.2f}% ({correct}/{total})")
    model.train() # set model back to training mode if it's being used in a loop
    return accuracy

if __name__ == "__main__":
    model_save_path = Path.cwd() / MODEL_STATE_FILENAME
    loss_list = []
    accuracy_list = []
    epochs_ran = []

    if model_save_path.exists():
        print(f"Loading pre-trained model from {model_save_path}")
        clf.load_state_dict(load(model_save_path))
        clf.eval()
        print("Model loaded. Skipping training for now.")
        final_accuracy = evaluate_model(clf, test_loader, device)
        print(f"Final Test Accuracy: {final_accuracy:.2f}%")
    else:
        print("No pre-trained model found. Starting training...")
        clf.train() 
        for epoch in range(NUM_EPOCHS):
            current_loss = 0.0 
            for batch_idx, (x, y) in enumerate(train_loader):
                x, y = x.to(device), y.to(device) 

                y_pred = clf(x)
                loss = loss_fn(y_pred, y)

                opt.zero_grad()
                loss.backward()
                opt.step()
                
                current_loss += loss.item()

            avg_loss_per_epoch = current_loss / len(train_loader)
            print(f"Epoch {epoch} loss: {avg_loss_per_epoch:.4f}")
            # collect metrics
            loss_list.append(avg_loss_per_epoch)
            accuracy = evaluate_model(clf, test_loader, device, epoch)
            accuracy_list.append(accuracy)
            epochs_ran.append(epoch)

        with open(model_save_path, "wb") as f:
            save(clf.state_dict(), f)
        print(f"Model state saved to {model_save_path}")

        # Plot Loss
        plt.subplot(1, 2, 1) 
        plt.plot(epochs_ran, loss_list, marker='o', linestyle='-', color='blue')
        plt.title('Training Loss per Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)

        # Plot Accuracy
        plt.subplot(1, 2, 2) 
        plt.plot(epochs_ran, accuracy_list, marker='o', linestyle='-', color='red')
        plt.title('Test Accuracy per Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.grid(True)

        plt.tight_layout() 
        plt.show() 
    sample_image_path = Path(__file__).parent / "sample_digit.jpg"
    predict(clf, sample_image_path, device)