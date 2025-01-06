import torch
from torchvision.models import ResNet18_Weights, resnet18


class ResNet18BinaryClassifier(torch.nn.Module):
    """
    Бинарный классификатор с предобученной моделью ResNet-18
    (с замороженными слоями кроме последнего)
    """

    def __init__(self, num_classes=2):
        super(ResNet18BinaryClassifier, self).__init__()
        # Загружаем предобученную модель ResNet-18
        self.resnet18 = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

        # Замораживаем все слои, кроме последнего полносвязного слоя
        for param in self.resnet18.parameters():
            param.requires_grad = False

        # Заменяем последний полносвязный слой на новый для бинарной классификации
        in_features = self.resnet18.fc.in_features
        self.resnet18.fc = torch.nn.Sequential(
            torch.nn.Linear(in_features, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(256, num_classes),
        )

        # Разрешаем обновление градиентов для последнего слоя
        for param in self.resnet18.fc.parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.resnet18(x)
