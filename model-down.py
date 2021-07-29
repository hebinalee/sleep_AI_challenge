###########################
## TO DOWNLOAD DNN MODELS 
###########################

import torchvision.models as models
from efficientnet_pytorch import EfficientNet

resnet18 = models.resnet18(pretrained=True)
resnet34 = models.resnet34(pretrained=True)
resnet50 = models.resnet50(pretrained=True)
resnet101 = models.resnet101(pretrained=True)
resnet152 = models.resnet152(pretrained=True)

#alexnet = models.alexnet(pretrained=True)
squeezenet1_0 = models.squeezenet1_0(pretrained=True)
squeezenet1_1 = models.squeezenet1_1(pretrained=True)

vgg11 = models.vgg11(pretrained=True)
vgg16 = models.vgg16(pretrained=True)
vgg19 = models.vgg19(pretrained=True)

densenet121 = models.densenet121(pretrained=True)
densenet169 = models.densenet169(pretrained=True)
densenet201 = models.densenet201(pretrained=True)
densenet161 = models.densenet161(pretrained=True)

inception = models.inception_v3(pretrained=True)

googlenet = models.googlenet(pretrained=True)

shufflenet = models.shufflenet_v2_x1_0(pretrained=True)

mobilenet = models.mobilenet_v2(pretrained=True)

resnext50_32x4d = models.resnext50_32x4d(pretrained=True)
resnext101_32x8d = models.resnext101_32x8d(pretrained=True)

wide_resnet50_2 = models.wide_resnet50_2(pretrained=True)
wide_resnet101_2 = models.wide_resnet101_2(pretrained=True)

mnasnet = models.mnasnet1_0(pretrained=True)

efficientnetb0 = EfficientNet.from_pretrained('efficientnet-b0', num_classes=5)
efficientnetb1 = EfficientNet.from_pretrained('efficientnet-b1', num_classes=5)
efficientnetb2 = EfficientNet.from_pretrained('efficientnet-b2', num_classes=5)
efficientnetb3 = EfficientNet.from_pretrained('efficientnet-b3', num_classes=5)
efficientnetb4 = EfficientNet.from_pretrained('efficientnet-b4', num_classes=5)
efficientnetb5 = EfficientNet.from_pretrained('efficientnet-b5', num_classes=5)
efficientnetb6 = EfficientNet.from_pretrained('efficientnet-b6', num_classes=5)
efficientnetb7 = EfficientNet.from_pretrained('efficientnet-b7', num_classes=5)
