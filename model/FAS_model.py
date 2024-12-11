import torch
import torch.nn as nn
from torchvision.models.resnet import ResNet, BasicBlock

import numpy as np
from transformers import ViTModel, ViTConfig
import torchvision
from model.modified_vit.modeling_vit import ViTCrossModel

RESNET_CKPT = "./resnet18-f37072fd.pth"
VIT_CKPT = "./vit-base-patch16-224-in21k"

class Feature_Generator_CrossVIT(nn.Module):
    def __init__(self):
        super(Feature_Generator_CrossVIT, self).__init__()
        config = ViTConfig.from_pretrained(VIT_CKPT + '/config.json')
        model = ViTCrossModel(config)
        vit = ViTModel.from_pretrained(VIT_CKPT)
        for name, param in model.named_parameters():
            for name2, param_init in vit.named_parameters():
                if name == name2:
                    param.data = param_init.data
        self.vit = model
        self.resize = torchvision.transforms.Resize((224,224))
    def forward(self, input, encoder_hidden_states):
        input = self.resize(input)
        vit_input = {'pixel_values': input, 'encoder_hidden_states': encoder_hidden_states}
        vit_output = self.vit(**vit_input)
        sequence_output = vit_output[0]
        # take first token
        feature = sequence_output[:,0,:]
        return feature

class Feature_Embedder_VIT(nn.Module):
    def __init__(self):
        super(Feature_Embedder_VIT, self).__init__()
        self.bottleneck_layer_fc = nn.Linear(768, 512)
        self.bottleneck_layer_fc.weight.data.normal_(0, 0.005)
        self.bottleneck_layer_fc.bias.data.fill_(0.1)
        self.bottleneck_layer = nn.Sequential(
            self.bottleneck_layer_fc,
            nn.ReLU(),
            nn.Dropout(0.5)
        )
    def forward(self, input, norm_flag):
        feature = self.bottleneck_layer_fc(input)
        if (norm_flag):
            feature_norm = torch.norm(feature, p=2, dim=1, keepdim=True).clamp(min=1e-12) ** 0.5 * (2) ** 0.5
            feature = torch.div(feature, feature_norm)
        return feature

def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    model_path = RESNET_CKPT
    if pretrained:
        model.load_state_dict(torch.load(model_path))
        print("loading model: ", model_path)
    return model

class Feature_Generator_ResNet18(nn.Module):
    def __init__(self, pretrained=True):
        super(Feature_Generator_ResNet18, self).__init__()
        model_resnet = resnet18(pretrained=pretrained)
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
    def forward(self, input):
        feature = self.conv1(input)
        feature = self.bn1(feature)
        feature = self.relu(feature)
        feature = self.maxpool(feature)
        feature = self.layer1(feature)
        feature = self.layer2(feature)
        feature = self.layer3(feature)
        return feature

class Feature_Embedder_ResNet18_Cross(nn.Module):
    def __init__(self):
        super(Feature_Embedder_ResNet18_Cross, self).__init__()
        model_resnet = resnet18(pretrained=False)
        self.layer4 = model_resnet.layer4

    def forward(self, input):
        feature = self.layer4(input)
        feature = feature.reshape(-1,512,64).permute(0,2,1)
        return feature

def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output

class Classifier(nn.Module):
    def __init__(self, in_channel=512):
        super(Classifier, self).__init__()
        self.classifier_layer = nn.Linear(in_channel, 2)
        self.classifier_layer.weight.data.normal_(0, 0.01)
        self.classifier_layer.bias.data.fill_(0.0)

    def forward(self, input, norm_flag):
        if(norm_flag):
            self.classifier_layer.weight.data = l2_norm(self.classifier_layer.weight, axis=0)
            classifier_out = self.classifier_layer(input)
        else:
            classifier_out = self.classifier_layer(input)
        return classifier_out

class GradientReverseFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor, coeff=1.):
        ctx.coeff = coeff
        output = input * 1.0
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output.neg() * ctx.coeff, None

class GRL(nn.Module):
    def __init__(self):
        super(GRL, self).__init__()
        self.iter_num = 0
        self.alpha = 10
        self.low = 0.0
        self.high = 1.0
        self.max_iter = 4000  # be same to the max_iter of config.py

    def forward(self, input):
        self.iter_num += 1
        coeff = float(2.0 * (self.high - self.low) / (1.0 + np.exp(-self.alpha * self.iter_num / self.max_iter))
                         - (self.high - self.low) + self.low)
        return GradientReverseFunction.apply(input, coeff)

class Discriminator(nn.Module):
    def __init__(self, in_channel=512, n_domain=3, grl=True):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(in_channel, 512)
        self.fc1.weight.data.normal_(0, 0.01)
        self.fc1.bias.data.fill_(0.0)
        self.fc2 = nn.Linear(512, n_domain)
        self.fc2.weight.data.normal_(0, 0.3)
        self.fc2.bias.data.fill_(0.0)
        self.ad_net = nn.Sequential(
            self.fc1,
            nn.ReLU(),
            nn.Dropout(0.5),
            self.fc2
        )
        self.grl_layer = GRL()
        self.grl = grl

    def forward(self, feature):
        if self.grl:
            feature = self.grl_layer(feature)
        adversarial_out = self.ad_net(feature)
        return adversarial_out

class OA_Net(nn.Module):
    def __init__(self):
        super(OA_Net, self).__init__()
        self.feature_size = 512
        self.backbone = Feature_Generator_CrossVIT()
        self.embedder = Feature_Embedder_VIT()
        self.backbone_res = Feature_Generator_ResNet18()
        self.embedder_res = Feature_Embedder_ResNet18_Cross()

        self.classifier = Classifier(in_channel=self.feature_size)

    def forward(self, double_input, norm_flag=True):
        res = double_input[:,3:,]

        feature_res = self.backbone_res(res)
        mu_res = self.embedder_res(feature_res)

        input = double_input[:,:3,]
        feature = self.backbone(input, mu_res)
        embedding = self.embedder(feature, norm_flag)

        classifier_out = self.classifier(embedding, norm_flag)
        if self.training:
            return classifier_out, embedding
        else:
            return classifier_out
    
    def get_feature(self, double_input, norm_flag=True):
        input = double_input[:,:3,]
        feature = self.backbone(input)
        mu = self.embedder(feature, norm_flag)

        res = double_input[:,3:,]
        feature_res = self.backbone_res(res)
        mu_res = self.embedder_res(feature_res, norm_flag)

        embedding = torch.concat([mu, mu_res], dim=1)
        return embedding

