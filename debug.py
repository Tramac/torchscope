import torchvision.models as models

from torchscope import scope

model = models.resnet18()
scope(model, (3, 224, 224))
