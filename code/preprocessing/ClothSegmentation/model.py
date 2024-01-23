import torch
import torch.nn as nn
import torchvision.models as models

class classifier(nn.Module):
    def __init__(self):
        super(classifier, self).__init__()
        ResEncoder = models.resnet18(pretrained=True)
        self.backbone = torch.nn.Sequential(*(list(ResEncoder.children())[:-1])).cuda()

        self.cf = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, 2),
        )

    def weights_init(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear, nn.Conv3d)):
                nn.init.xavier_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Get input size
        x = self.backbone(x)

        b, c, h, w = x.shape
        x = x.reshape(b, -1)
        pred = self.cf(x)

        return pred
