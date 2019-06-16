import torch.nn as nn
import pdb
import torch

class AugResnet2(nn.Module):
    def __init__(self, resnet, outSize, dropout):
        super(AugResnet2, self).__init__()
        self.resnet_18 = resnet
        self.resnet_18.conv1 = nn.Conv2d(9, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.resnet_18.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.num_ftrs = self.resnet_18.fc.in_features
        self.resnet_18.fc = nn.Linear(self.num_ftrs, 512)


        if dropout:
            self.fc = nn.Sequential(
                    nn.Dropout(0.5),
                    nn.Linear(512+9, outSize)
                )
        else:
            self.fc = nn.Linear(512+9, outSize)

    def forward(self, x):
        mean_x = torch.squeeze(nn.AdaptiveAvgPool2d((1, 1))(x))

        cnn = self.resnet_18(x)
        out = torch.cat((cnn, mean_x), dim=1)
        #import pdb; pdb.set_trace()
        out = self.fc(out)
        return out
    
class AugResnet(nn.Module):
    def __init__(self, resnet, outSize, dropout):
        super(AugResnet, self).__init__()
        self.resnet_18 = resnet
        self.resnet_18.conv1 = nn.Conv2d(9, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.resnet_18.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.num_ftrs = self.resnet_18.fc.in_features
        self.resnet_18.fc = nn.Linear(self.num_ftrs, 512)

        self.mlp = nn.Sequential(
                    nn.Linear(9, 1024),
                    nn.ReLU(),
                    nn.Linear(1024, 512)
                    )

        if dropout:
            self.fc = nn.Sequential(
                    nn.Dropout(0.5),
                    nn.Linear(512+512, outSize)
                )
        else:
            self.fc = nn.Linear(512+512, outSize)

    def forward(self, x):
        mean_x = torch.squeeze(nn.AdaptiveAvgPool2d((1, 1))(x))
        mean_x = self.mlp(mean_x)

        cnn = self.resnet_18(x)
        out = torch.cat((cnn, mean_x), dim=1)
        #import pdb; pdb.set_trace()
        out = self.fc(out)
        return out


class AugmentedCNN(nn.Module):
    """ConvNet Augmented with AIA channel means"""
    def __init__(self):
        super(AugmentedCNN, self).__init__()
        self.convnet=nn.Sequential(
            nn.Conv2d(9,16,7),
            nn.LeakyReLU(),
            nn.Conv2d(16,32,7),
            nn.LeakyReLU(),
            nn.Conv2d(32,32,7),
            nn.LeakyReLU(),
            nn.Conv2d(32,16,7),
            nn.LeakyReLU())

        self.pool=nn.AdaptiveAvgPool2d((1,1))

        self.fc=nn.Sequential(
            nn.Linear(16+9,100),
            nn.LeakyReLU(),
            nn.Linear(100,14))

    def forward(self,aia_img):
        out=self.convnet(aia_img)
        out=self.pool(out)
        out=out.view(out.size(0),-1)
        aia_avg=self.pool(aia_img)
        aia_avg=aia_avg.view(out.size(0),-1)
        out=torch.cat([out,aia_avg],dim=1)
        out=self.fc(out)
        return out


class AvgLinearMap(nn.Module):
    """AVerage input, then do linear map"""
    def __init__(self, inplanes, outplanes):
        super(AvgLinearMap, self).__init__()
        self.downsample = nn.AdaptiveAvgPool2d((1,1))
        self.linear = nn.Linear(inplanes, outplanes)

    def forward(self, x): 
        x = self.downsample(x).view(x.size(0),-1)
        x = self.linear(x)
        return x



class AvgMLP(nn.Module):
    """AVerage input, then do a simple 1-hidden layer MLP"""
    def __init__(self, inplanes, hiddenplanes, outplanes, dropout=False):
        super(AvgMLP, self).__init__()
        self.downsample = nn.AdaptiveAvgPool2d((1,1))
        self.dropout = dropout
        if dropout:
            self.model = nn.Sequential(nn.Linear(inplanes, hiddenplanes), nn.Dropout(p=0.5),
                    nn.ReLU(), nn.Linear(hiddenplanes, outplanes))
        else:
            self.model = nn.Sequential(nn.Linear(inplanes, hiddenplanes), 
                    nn.ReLU(), nn.Linear(hiddenplanes, outplanes))

    def forward(self, x): 
        x = self.downsample(x).view(x.size(0),-1)
        x = self.model(x)
        return x
    
class DeepMLP(nn.Module):
    """Average input, then do a 3-hidden layer MLP"""
    def __init__(self, inplanes, hiddenplanes, outplanes, dropout=False):
        super(DeepMLP, self).__init__()
        self.downsample = nn.AdaptiveAvgPool2d((1,1))
        self.dropout = dropout
        if dropout:
            self.model = nn.Sequential(nn.Linear(inplanes, hiddenplanes), 
                    nn.ReLU(), nn.Linear(hiddenplanes, hiddenplanes), 
                    nn.ReLU(), nn.Linear(hiddenplanes, hiddenplanes) ,                  
                    nn.Dropout(p=0.5), nn.ReLU(), nn.Linear(hiddenplanes, outplanes))
        else:
            self.model = nn.Sequential(nn.Linear(inplanes, hiddenplanes), 
                    nn.ReLU(), nn.Linear(hiddenplanes, hiddenplanes), 
                    nn.ReLU(), nn.Linear(hiddenplanes, hiddenplanes),                  
                    nn.ReLU(), nn.Linear(hiddenplanes, outplanes))

    def forward(self, x): 
        x = self.downsample(x).view(x.size(0),-1)
        x = self.model(x)
        return x

class ChoppedAlexnetBN(nn.Module):
    def getLayers(self, numLayers):
        """Returns a list of layers + the feature size coming out"""
        layers = [nn.Conv2d(9, 64, kernel_size=11, stride=4, padding=2), nn.BatchNorm2d(64), nn.ReLU(inplace=True), ]
        if numLayers == 1:
            return (layers, 64)
        layers += [nn.MaxPool2d(kernel_size=3, stride=2), nn.Conv2d(64, 192, kernel_size=5, padding=2), nn.BatchNorm2d(192), nn.ReLU(inplace=True), ]
        if numLayers == 2:
            return (layers, 192)
        layers += [nn.MaxPool2d(kernel_size=3, stride=2), nn.Conv2d(192, 384, kernel_size=3, padding=1), nn.BatchNorm2d(384), nn.ReLU(inplace=True)]
        if numLayers == 3:
            return (layers,384)

        layers += [nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.BatchNorm2d(256)]
        return (layers,256)
        
    def __init__(self, numLayers, outSize, dropout):
        super(ChoppedAlexnetBN, self).__init__()
        layers, channelSize = self.getLayers(numLayers)
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        if (dropout):
            self.model = nn.Sequential(
                nn.Dropout(p=0.5),
                nn.Linear(channelSize, outSize)
                )
        else:
            self.model = nn.Linear(channelSize, outSize)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).view(x.size(0),-1)
        x = self.model(x)
        return x


class ChoppedAlexnetSpatial(nn.Module):
    """Spatial version of chopped alexnet
        
    This returns the feature map if asked for.
        
        """
    def getLayers(self, numLayers):
        """Returns a list of layers + the feature size coming out"""
        layers = [nn.Conv2d(9, 64, kernel_size=11, stride=4, padding=2), nn.ReLU(inplace=True), ]
        if numLayers == 1:
            return (layers, 64, 55)
        layers += [nn.MaxPool2d(kernel_size=3, stride=2), nn.Conv2d(64, 192, kernel_size=5, padding=2), nn.ReLU(inplace=True), ]
        if numLayers == 2:
            return (layers, 192, 29)
        layers += [nn.MaxPool2d(kernel_size=3, stride=2), nn.Conv2d(192, 384, kernel_size=3, padding=1), nn.ReLU(inplace=True)]
        if numLayers == 3:
            return (layers,384, 29)

        layers += [nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True)]
        return (layers,256)
        
    def __init__(self, numLayers, outSize, returnIntermediate=True):
        super(ChoppedAlexnetSpatial, self).__init__()
        layers, channelSize, channelDim = self.getLayers(numLayers)
        self.features = nn.Sequential(*(layers+[nn.Conv2d(channelSize, outSize, kernel_size=1)]))
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.returnIntermediate = returnIntermediate
        self.channelDim = channelDim 

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        f = self.features(x)
        x = self.pool(f).view(x.size(0),-1)
        if self.returnIntermediate:
            return x, f
        else:
            return x

class ChoppedAlexnet64BN(nn.Module):
    def getLayers(self, numLayers):
        """Returns a list of layers + the feature size coming out"""
        layers = [nn.Conv2d(9, 64, kernel_size=11, stride=4, padding=2), nn.BatchNorm2d(64), nn.ReLU(inplace=True), ]
        if numLayers == 1:
            return (layers, 64)
        layers += [nn.MaxPool2d(kernel_size=3, stride=2), nn.Conv2d(64, 64, kernel_size=5, padding=2), nn.BatchNorm2d(64), nn.ReLU(inplace=True), ]
        if numLayers == 2:
            return (layers, 64)
        layers += [nn.MaxPool2d(kernel_size=3, stride=2), nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True)]
        if numLayers == 3:
            return (layers,64)

        layers += [nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True)]
        return (layers,64)
        
    def __init__(self, numLayers, outSize, dropout):
        super(ChoppedAlexnet64BN, self).__init__()
        layers, channelSize = self.getLayers(numLayers)
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        if (dropout):
            self.model = nn.Sequential(
                nn.Dropout(p=0.5),
                nn.Linear(channelSize, outSize)
                )
        else:
            self.model = nn.Linear(channelSize, outSize)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).view(x.size(0),-1)
        x = self.model(x)
        return x



class ChoppedAlexnet64(nn.Module):
    def getLayers(self, numLayers):
        """Returns a list of layers + the feature size coming out"""
        layers = [nn.Conv2d(9, 64, kernel_size=11, stride=4, padding=2), nn.LeakyReLU(inplace=True), ]
        if numLayers == 1:
            return (layers, 64)
        layers += [nn.MaxPool2d(kernel_size=3, stride=2), nn.Conv2d(64, 64, kernel_size=5, padding=2), nn.LeakyReLU(inplace=True), ]
        if numLayers == 2:
            return (layers, 64)
        layers += [nn.MaxPool2d(kernel_size=3, stride=2), nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.LeakyReLU(inplace=True)]
        if numLayers == 3:
            return (layers,64)

        layers += [nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.LeakyReLU(inplace=True)]
        return (layers,256)
        
    def __init__(self, numLayers, outSize, dropout):
        super(ChoppedAlexnet64, self).__init__()
        layers, channelSize = self.getLayers(numLayers)
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        if (dropout):
            self.model = nn.Sequential(
                nn.Dropout(p=0.5),
                nn.Linear(channelSize, outSize)
                )
        else:
            self.model = nn.Linear(channelSize, outSize)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).view(x.size(0),-1)
        x = self.model(x)
        return x



class ChoppedAlexnet(nn.Module):
    def getLayers(self, numLayers):
        """Returns a list of layers + the feature size coming out"""
        layers = [nn.Conv2d(9, 64, kernel_size=11, stride=4, padding=2), nn.ReLU(inplace=True), ]
        if numLayers == 1:
            return (layers, 64)
        layers += [nn.MaxPool2d(kernel_size=3, stride=2), nn.Conv2d(64, 192, kernel_size=5, padding=2), nn.ReLU(inplace=True), ]
        if numLayers == 2:
            return (layers, 192)
        layers += [nn.MaxPool2d(kernel_size=3, stride=2), nn.Conv2d(192, 384, kernel_size=3, padding=1), nn.ReLU(inplace=True)]
        if numLayers == 3:
            return (layers,384)

        layers += [nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True)]
        return (layers,256)
        
    def __init__(self, numLayers, outSize, dropout):
        super(ChoppedAlexnet, self).__init__()
        layers, channelSize = self.getLayers(numLayers)
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        if (dropout):
            self.model = nn.Sequential(
                nn.Dropout(p=0.5),
                nn.Linear(channelSize, outSize)
                )
        else:
            self.model = nn.Linear(channelSize, outSize)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).view(x.size(0),-1)
        x = self.model(x)
        return x
        

class ChoppedVGG(nn.Module):
    def getLayers(self, numLayers):
        """Returns a list of layers + the feature size coming out"""
        layers = [nn.Conv2d(9, 64, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(64), 
                  nn.ReLU(inplace=True)]
        if numLayers == 1:
            return (layers, 64)
        layers += [nn.Conv2d(64, 64, kernel_size=3,stride=1, padding=1), nn.BatchNorm2d(64),
                   nn.ReLU(inplace=True), ]
        if numLayers == 2:
            return (layers, 64)
        layers += [nn.MaxPool2d(kernel_size=2, stride=2, padding = 0, dilation = 1, ceil_mode=False),
                    nn.Conv2d(64, 128, kernel_size=3, stride =1, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True)]
        if numLayers == 3:
            return (layers,128)
        layers += [nn.Conv2d(128, 128, kernel_size=3,stride=1, padding=1), nn.BatchNorm2d(128),
                   nn.ReLU(inplace=True)]
        return (layers,128)
        
    def __init__(self, numLayers, outSize, dropout):
        super(ChoppedVGG, self).__init__()
        layers, channelSize = self.getLayers(numLayers)
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        if (dropout):
            self.model = nn.Sequential(
                nn.Dropout(p=0.5),
                nn.Linear(channelSize, outSize)
                )
        else:
            self.model = nn.Linear(channelSize, outSize)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')


    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).view(x.size(0),-1)
        x = self.model(x)
        return x




