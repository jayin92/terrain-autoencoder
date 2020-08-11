import torch,torchvision
nn=torch.nn
class Encoder(nn.Module):
    def __init__(self,depth,innc,nc,outnc):
        super(Encoder,self).__init__()
        self.nLayer=depth
        self.layers=nn.ModuleList()
        self.outnc=outnc
        for i in range(self.nLayer):
            self.layers.append(
                nn.Sequential(
                nn.Conv2d(nc[i-1], nc[i], kernel_size=4,
                             stride=2, padding=1, bias=True),# batchNorm has bias
                #nn.BatchNorm2d(nc[i]),
                nn.LeakyReLU(0.1),
                nn.Conv2d(nc[i], nc[i], kernel_size=3,
                             stride=1, padding=1, bias=False),# batchNorm has bias
                nn.BatchNorm2d(nc[i]),
                nn.LeakyReLU(0.1)
            )if i!=0 else
                nn.Sequential(
                nn.Conv2d(innc, nc[i], kernel_size=3,
                             stride=1, padding=1, bias=True),# batchNorm has bias
                #nn.BatchNorm2d(nc[i]),
                nn.LeakyReLU(0.1),
                nn.Conv2d(nc[i], nc[i], kernel_size=3,
                             stride=1, padding=1, bias=False),# batchNorm has bias
                nn.BatchNorm2d(nc[i]),
                nn.LeakyReLU(0.1)
            )
                
                              )
    def forward(self,t):
        output=[]
        for i in range(self.nLayer):
            t=self.layers[i](t)
            output.append(t[:,:self.outnc[i]])
        return output

class Decoder(nn.Module):
    def __init__(self,depth,innernc,nc,outnc):
        super(Decoder,self).__init__()
        self.nLayer=depth
        self.layers=nn.ModuleList()
        for i in range(self.nLayer):
            self.layers.append(nn.Sequential(
                nn.ConvTranspose2d(nc[i], innernc[i-1], kernel_size=3,
                             stride=1, padding=1, bias=True),# batchNorm has bias
                #nn.BatchNorm2d(innernc[i-1]),
                nn.LeakyReLU(0.1),
                nn.ConvTranspose2d(innernc[i-1], innernc[i-1], kernel_size=4,
                             stride=2, padding=1, bias=False),# batchNorm has bias
                nn.BatchNorm2d(innernc[i-1]),
                nn.LeakyReLU(0.1)
            )if i!=0 else nn.Sequential(
                nn.ConvTranspose2d(nc[i], nc[i], kernel_size=3,
                             stride=1, padding=1, bias=True),# batchNorm has bias
                #nn.BatchNorm2d(innernc[i-1]),
                nn.LeakyReLU(0.1),
                nn.ConvTranspose2d(nc[i], outnc, kernel_size=3,
                             stride=1, padding=1, bias=True),# batchNorm has bias
            ))
    def forward(self,t1,t2):
        output=[]
        for i in range(self.nLayer-1,-1,-1):
            if i==self.nLayer-1:
                t=torch.cat([t1[i],t2[i]],dim=1)
            else:
                t=torch.cat([t,t1[i],t2[i]],dim=1)
            t=self.layers[i](t)
        return t
    
class Discriminator(nn.Module):
    def __init__(self,depth,innc,nc):
        super(Discriminator,self).__init__()
        self.nLayer=depth
        net=[]
        for i in range(self.nLayer):
            net.append(nn.Sequential(
                nn.Conv2d(nc[i-1]if i>0 else innc, nc[i], kernel_size=4,
                             stride=2, padding=1, bias=False),# batchNorm has bias
                nn.BatchNorm2d(nc[i]),
                nn.LeakyReLU(0.1)
            ))
        self.net=nn.Sequential(*net)
        self.fc=nn.Linear(nc[depth-1],1)
    def forward(self,t):
        return self.fc(self.net(t).mean(3).mean(2))   
class Mult(nn.Module):
    def __init__(self,nc):
        super(Mult,self).__init__()
        
        self.register_parameter(name='exp',
                                param=torch.nn.Parameter(torch.diag(torch.ones(nc)).unsqueeze(-1).unsqueeze(-1)))
                                
        #self.exp=torch.diag(torch.ones(nc)).unsqueeze(-1).unsqueeze(-1).to('cuda:1')
        '''self.register_parameter(name='weight',
                                param=torch.nn.Parameter(torch.ones(nc).unsqueeze(-1).unsqueeze(-1)))
                                '''
        self.register_parameter(name='bias',
                                param=torch.nn.Parameter(torch.zeros(nc).unsqueeze(-1).unsqueeze(-1)))
        self.relu=nn.ReLU()
    def forward(self,x):
        #return self.leaky_relu(x.unsqueeze(-3).pow(self.exp).prod(1)*self.weight+self.bias)
        x=self.relu(x)+0.1
        return x.unsqueeze(-3).pow(self.exp).prod(1)+self.bias

class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self,layers=[4,9,16,23]):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        lastLayer=0
        for l in layers:
            blocks.append(torchvision.models.vgg16(pretrained=True).features[lastLayer:l].eval())
            lastLayer=l
        for bl in blocks:
            for p in bl:
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.mean = torch.nn.Parameter(torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.std = torch.nn.Parameter(torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))

    def forward(self, input, target):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        input = (input-self.mean) / self.std
        target = (target-self.mean) / self.std
        loss = torch.tensor(0.)
        x = input
        y = target
        for block in self.blocks:
            x = block(x)
            y = block(y)
            loss += torch.nn.functional.l1_loss(x, y)
        return loss