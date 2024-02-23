import torch 
import torch.nn as nn


class Encoder(nn.Module):
    #vgg
    def __init__(self):
        super(Encoder, self).__init__()
        # first block
        # 224 x 224
        self.conv_1_1 = nn.Conv2d(3,3,1,1,0)
        self.reflecPad_1_1 = nn.ReflectionPad2d((1,1,1,1))

        self.conv_1_2 = nn.Conv2d(3,64,3,1,0)
        self.relu_1_2 = nn.ReLU(inplace=True)
        # 224 x 224

        self.reflecPad_1_3 = nn.ReflectionPad2d((1,1,1,1))
        self.conv_1_3 = nn.Conv2d(64,64,3,1,0)
        self.relu_1_3 = nn.ReLU(inplace=True)

        self.maxPool_1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 112 x 112

        # second block
        self.reflecPad_2_1 = nn.ReflectionPad2d((1,1,1,1))
        self.conv_2_1 = nn.Conv2d(64,128,3,1,0)
        self.relu_2_1 = nn.ReLU(inplace=True)

        self.reflecPad_2_2 = nn.ReflectionPad2d((1,1,1,1))
        self.conv_2_2 = nn.Conv2d(128,128,3,1,0)
        self.relu_2_2 = nn.ReLU(inplace=True)

        self.maxPool_2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 56 x 56

        # third block
        self.reflecPad_3_1 = nn.ReflectionPad2d((1,1,1,1))
        self.conv_3_1 = nn.Conv2d(128,128,3,1,0)
        self.relu_3_1 = nn.ReLU(inplace=True)

        self.reflecPad_3_2 = nn.ReflectionPad2d((1,1,1,1))
        self.conv_3_2 = nn.Conv2d(128,256,3,1,0)
        self.relu_3_2 = nn.ReLU(inplace=True)

        self.reflecPad_3_3 = nn.ReflectionPad2d((1,1,1,1))
        self.conv_3_3 = nn.Conv2d(256,256, 3, 1, 0)
        self.relu_3_3 = nn.ReLU(inplace=True)

        self.reflecPad_3_4 = nn.ReflectionPad2d((1,1,1,1))
        self.conv_3_4 = nn.Conv2d(256,256,3,1,0)
        self.relu_3_4 = nn.ReLU(inplace=True)

        self.maxPool_3  = nn.MaxPool2d(kernel_size=2, stride = 2)
        # 28 x 28

        # fourth block
        self.reflecPad_4_1 = nn.ReflectionPad2d((1,1,1,1))
        self.conv_4_1 = nn.Conv2d(256,512,3,1,0)
        self.relu_4_1 = nn.ReLU(inplace=True)

        self.reflecPad_4_2 = nn.ReflectionPad2d((1,1,1,1))
        self.conv_4_2 = nn.Conv2d(512,512,3,1,0)
        self.relu_4_2 = nn.ReLU(inplace=True)

        self.reflecPad_4_3 = nn.ReflectionPad2d((1,1,1,1))
        self.conv_4_3 = nn.Conv2d(512,512,3,1,0)
        self.relu_4_3 = nn.ReLU(inplace=True)

        self.reflecPad_4_4 = nn.ReflectionPad2d((1,1,1,1))
        self.conv_4_4 = nn.Conv2d(512,512,3,1,0)
        self.relu_4_4 = nn.ReLU(inplace=True)

        self.maxpool_4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # fifth block
        self.reflecPad_5_1 = nn.ReflectionPad2d((1,1,1,1))
        self.conv_5_1 = nn.Conv2d(512,512,3,1,0)
        self.relu_5_1 = nn.ReLU(inplace=True)

    def forward(self, x , sF = None, contenV256 = None, styleV256 = None, matrixl1 = None, matrix21 = None, matrix31 = None):
        output = {}

        # first block
        out = self.conv_1_1(x)
        out = self.reflecPad_1_1(out)
        out = self.conv_1_2(out)
        out = self.relu_1_2(out)
        
        output['r11'] = out

        # out = self.reflecPad_1_3(out)
        out = self.reflecPad_3_2(out)
        out = self.conv_1_3(out)
        out = self.relu_1_3(out)

        output['r12'] = out

        out = self.maxPool_1(out)

        output['p1'] = out

        # second block
        out = self.reflecPad_2_1(out)
        out = self.conv_2_1(out)
        out = self.relu_2_1(out)

        output['r21'] = out

        # out = self.reflecPad_2_2(out)
        out = self.reflecPad_3_2(out)
        out = self.conv_2_2(out)
        out = self.relu_2_2(out)

        output['r22'] = out

        out = self.maxPool_2(out)

        output['p2'] = out

        # third block
        out = self.reflecPad_3_1(out)
        out = self.conv_3_1(out)
        out = self.relu_3_1(out)
        
        output['r31'] = out
        
        if(styleV256 is not None):
            feature = matrix31(output['r31'], sF['r31'], contenV256, styleV256)
            out = self.reflecPad_3_2(feature)
        else:
            out = self.reflecPad_3_2(out)
        out = self.conv_3_2(out)
        out = self.relu_3_2(out)

        output['r32'] = out

        out = self.reflecPad_3_3(out)
        out = self.conv_3_3(out)
        out = self.relu_3_3(out)

        output['r33'] = out

        out = self.reflecPad_3_4(out)
        out = self.conv_3_4(out)
        out = self.relu_3_4(out)

        output['r34'] = out

        out = self.maxPool_3(out)
        
        output['p3'] = out

        # fourth block
        out = self.reflecPad_4_1(out)
        out = self.conv_4_1(out)
        out = self.relu_4_1(out)

        output['r41'] = out

        out = self.reflecPad_4_2(out)
        out = self.conv_4_2(out)
        out = self.relu_4_2(out)

        output['r42'] = out

        out = self.reflecPad_4_3(out)
        out = self.conv_4_3(out)
        out = self.relu_4_3(out)

        output['r43'] = out

        out = self.reflecPad_4_4(out)
        out = self.conv_4_4(out)
        out = self.relu_4_4(out)

        output['r44'] = out

        out = self.maxpool_4(out)
        
        output['p4'] = out

        # fifth block
        out = self.reflecPad_5_1(out)
        out = self.conv_5_1(out)
        out = self.relu_5_1(out)

        output['r51'] = out

        return output