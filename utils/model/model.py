from utils.model import *

class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, train, label=False):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        if train:
            one_hot = torch.zeros(cosine.size(), device='cuda')
            one_hot.scatter_(1, label.cuda().view(-1, 1).long(), 1)
            output = (one_hot * phi) + ((1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
        else:
            output = cosine
        output *= self.s

        return output


class ArcNet(nn.Module):
    def __init__(self, net, nc):
        super().__init__()
        self.base = net
        self.arcface = ArcMarginProduct(512, nc, s=30, m=0.5, easy_margin=False)
    
    def forward(self, x, label):
        x = self.base(x)
        if self.training:
            x = self.arcface(x, self.training, label)
        else:
            x = self.arcface(x, self.training)
        return x

def unfreeze(model, idx_from_last):
    list_layers = list(model.named_children())[-idx_from_last:]

    for param in model.parameters():
        param.requires_grad = False

    for layer in list_layers:
        ic(layer)
        for param in layer[1].parameters():
            param.requires_grad = True