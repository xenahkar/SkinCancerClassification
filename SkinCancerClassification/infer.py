import hydra
import torch
import torchvision.transforms as transforms
from model import ResNet18BinaryClassifier
from omegaconf import DictConfig
from PIL import Image
from trainer import SkinImageClassifier


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(config: DictConfig):
    image_path = config["infer"]["image_name"]
    image = Image.open(image_path).convert("RGB")

    transformer = transforms.Compose(
        [
            transforms.Resize(
                (config["model"]["image_height"], config["model"]["image_width"])
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                config["model"]["image_mean"], config["model"]["image_std"]
            ),
        ]
    )
    input_tensor = transformer(image).unsqueeze(0)

    model = ResNet18BinaryClassifier(config["model"]["num_classes"])
    module = SkinImageClassifier.load_from_checkpoint(
        f'{config["model"]["model_local_path"]}/{config["testing"]["checkpoint_name"]}',
        model=model,
        lr=config["training"]["lr"],
    )

    module.eval()
    with torch.no_grad():
        logits = module(input_tensor)
        probabilities = torch.softmax(logits, dim=1).squeeze(0)

    predicted_class = torch.argmax(probabilities).item()
    confidence = probabilities[predicted_class].item()
    class_names = ["benign", "malignant"]

    print(
        {
            "Skin Cancer Prediction": class_names[predicted_class],
            "Confidence": f"{confidence * 100:.2f} %",
        }
    )


if __name__ == "__main__":
    main()
