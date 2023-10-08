import torch
from torchvision import datasets
from torchvision.transforms import ToTensor

from neural_net import NeuralNetwork
from utils import get_device

if __name__ == "__main__":
    model_path = "models/model.pth"

    device = get_device()
    model = NeuralNetwork().to(device)
    model.load_state_dict(torch.load(model_path))

    classes = [
        "T-shirt/top",
        "Trouser",
        "Pullover",
        "Dress",
        "Coat",
        "Sandal",
        "Shirt",
        "Sneaker",
        "Bag",
        "Ankle boot",
    ]

    model.eval()
    # Download test data from open datasets.
    test_data = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor(),
    )
    x, y = test_data[0][0], test_data[0][1]
    with torch.no_grad():
        x = x.to(device)
        pred = model(x)
        predicted, actual = classes[pred[0].argmax(0)], classes[y]
        print(f'Predicted: "{predicted}", Actual: "{actual}"')
