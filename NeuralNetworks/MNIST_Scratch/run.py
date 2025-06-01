import mnist_loader
from PIL import Image
import numpy as np
import network
from pathlib import Path

training_data, validation_data, testing_data = mnist_loader.load_data_wrapper()
# 784 input image pixels, 30 hidden neurons, 10 output neurons consisting of probability of images
net = network.Network([784, 30, 10])

# use stochastic gradient descent to learn from the MNIST training_data over 30 epochs, with a mini-batch size of 10, and a learning rate of η = 0.01
net.SGD(training_data, 30, 10, 0.01, test_data=testing_data)

# Predicting new image
def preprocess_image(image_path):
    """
    Loads an image, converts to grayscale, resizes to 28x28,
    normalizes pixels, and reshapes to (784, 1) numpy array.
    """
    try:
        img = Image.open(image_path).convert('L') # grayscale
        img = img.resize((28, 28)) 
        img_array = np.array(img) 

        # MNIST pixels are 0 (black) to 255 (white).
        # Flatten and transpose to (784, 1) column vector
        img_vector = np.reshape(img_array, (784, 1)) / 255.0
        return img_vector

    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        return None
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

image_path = Path(__file__).parent / "data" / "sample_digit.jpg"
processed_image = preprocess_image(image_path)
if processed_image is not None:
    output_activation = net.forward(processed_image)
    prediction = np.argmax(output_activation) # idx of neuron with highest activation
    print("\nNetwork Output Activations (confidence for each digit 0-9):")
    for i, val in enumerate(output_activation.flatten()):
            print(f"  Digit {i}: {val * 100:.2f}%")
    print(f"\nPredicted digit: {prediction}")
else:
    print("Could not make a prediction due to image processing error.")

