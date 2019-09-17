import torch
import torchvision
import torch.optim as optim
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
class BCNN(torch.nn.Module):
    """B-CNN for CUB200.
    The B-CNN model is illustrated as follows.
    conv1^2 (64) -> pool1 -> conv2^2 (128) -> pool2 -> conv3^3 (256) -> pool3
    -> conv4^3 (512) -> pool4 -> conv5^3 (512) ->relu5_3-> bilinear pooling
    -> sqrt-normalize -> L2-normalize -> fc (200).
    The network accepts a 3*448*448 input, and the pool5 activation has shape
    512*28*28 since we down-sample 5 times.
    Attributes:
        features, torch.nn.Module: Convolution and pooling layers.
        fc, torch.nn.Module: 200.
    """
    def __init__(self,num_classes,is_all):
        """Declare all needed layers."""
        torch.nn.Module.__init__(self)
        self.is_all=is_all
        if self.is_all:
            self.features = torchvision.models.vgg16(pretrained=True).features
            self.features = torch.nn.Sequential(*list(self.features.children())
                                            [:-1])  # Remove max_pool5.
            self.fc = torch.nn.Linear(512*512, num_classes,bias=True)
        if not self.is_all:
            self.features = torchvision.models.vgg16(pretrained=True).features
            self.features = torch.nn.Sequential(*list(self.features.children())
            [:-1])  # Remove pool5.
            # Linear classifier.
            self.fc = torch.nn.Linear(512 ** 2, 200)

            # Freeze all previous layers.
            for param in self.features.parameters():
                param.requires_grad = False
            # Initialize the fc layers.
            torch.nn.init.kaiming_normal(self.fc.weight.data)
            if self.fc.bias is not None:
                torch.nn.init.constant(self.fc.bias.data, val=0)


    def forward(self, X):
        """Forward pass of the network.
        Args:
            X, torch.autograd.Variable of shape N*3*448*448.
        Returns:
            Score, torch.autograd.Variable of shape N*200.
        """
        N = X.size()[0]
        assert X.size() == (N, 3, 448, 448)
        X = self.features(X)
        assert X.size() == (N, 512, 28, 28)

        X =torch.reshape(X,(N,512,28*28))
        X = torch.bmm(X, torch.transpose(X, 1, 2)) / (28*28)  # Bilinear
        assert X.size() == (N, 512, 512)
        X=torch.reshape(X,(N,512*512))
        X = torch.sqrt(X + 1e-5)
        X = torch.nn.functional.normalize(X)
        X = self.fc(X)

        return X