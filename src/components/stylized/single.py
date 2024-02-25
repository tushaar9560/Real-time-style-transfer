import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, matrixSize=32):
        super(CNN, self).__init__()

        self.convs = nn.Sequential(nn.Conv2d(512,256,3,1,1),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(256,128,3,1,1),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(128, matrixSize,3,1,1)
                                   )
        # 32 x 8 x 8
        self.fc = nn.Linear(matrixSize*matrixSize, matrixSize*matrixSize)

    def forward(self, x):
        out = self.convs(x)
        b,c,h,w = out.size()
        out = out.view(b,c,-1)
        # 32 x 64
        out = torch.bmm(out, out.transpose(1,2).div(h*w))
        # 32 x 32
        out = out.view(out.size(0),-1)
        return self.fc(out)
    

class VAE(nn.Module):
    def __init__(self,z_dim):
        super(VAE, self).__init__()

        # 32 x 8 x 8
        self.encode = nn.Sequential(nn.Linear(512, 2*z_dim))
        self.bn = nn.BatchNorm1d(z_dim)
        self.decode = nn.Sequential(nn.Linear(z_dim,512),
                                    nn.BatchNorm1d(512),
                                    nn.ReLU(),
                                    nn.Linear(512,512),
                                    )
        
    def reparameterize(self, mu, logvar):
        mu = self.bn(mu)
        std = torch.exp(logvar)
        eps = torch.randn_like(std)

        return mu + std
    
    def forward(self, x):
        b,c,h = x.size()
        x = x.view(b,-1)

        z_q_mu, z_q_logvar = self.encode(x).chunk(2,dim=1)

        z_q = self.reparametize(z_q_mu, z_q_logvar)
        out = self.decode(z_q)
        out = out.view(b,c,h)

        KL = torch.sum(0.5 * (z_q_mu.pow(2) + z_q_logvar.exp().pow(2) - 1) - z_q_logvar)
        return out, KL
    

class MulLayer(nn.Module):
    def __init__(self,z_dim, matrixSize=32):
        super(MulLayer,self).__init__()
        self.snet = CNN(matrixSize)
        self.snet = CNN(matrixSize)
        self.VAE = VAE(z_dim)
        self.matrixSize = matrixSize
        self.compress = nn.Conv2d(512, matrixSize, 1,1,0)
        self.unzip = nn.Conv2d(matrixSize,512,1,1,0)
        self.transmatrix = None

    def forward(self, cF, sF, trans=True):
        cb,cc,ch,cw = cF.size()
        cFF = cF.view(cb,cc,-1)
        cMean = torch.mean(cFF,dim=2, keepdim=True)
        cMean = cMean.unsqueeze(3)
        cMean = cMean.expand_as(cF)
        cF = cF - cMean

        sb,sc,sh,sw = sF.size()
        sFF = sF.view(sb,sc,-1)
        sMean = torch.mean(sFF, dim=2, keepdim=True)
        sMean, KL = self.VAE(sMean)
        sMean = sMean.unsqueeze(3)
        sMeanC = sMean.expand_as(cF)
        sMeanS = sMean.expand_as(sF)
        sF = sF - sMeanS

        compress_content = self.compress(cF)
        b,c,h,w = compress_content.size()
        compress_content = compress_content.view(b,c,-1)

        if(trans):
            cMatrix = self.cnet(cF)
            sMatrix = self.snet(sF)

            sMatrix = sMatrix.view(sMatrix.size(0),self.matrixSize,self.matrixSize)
            cMatrix = cMatrix.view(cMatrix.size(0),self.matrixSize,self.matrixSize)
            transmatrix = torch.bmm(sMatrix,cMatrix)
            transfeature = torch.bmm(transmatrix,compress_content).view(b,c,h,w)
            out = self.unzip(transfeature.view(b,c,h,w))
            out = out + sMeanC
            return out, transmatrix, KL
        else:
            out = self.unzip(compress_content.view(b,c,h,w))
            out = out + cMean
            return out
