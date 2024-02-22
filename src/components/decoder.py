import torch
import torch.nn as nn


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        # decoder
        # first block
        self.reflecPad_1_1 = nn.ReflectionPad2d((1,1,1,1))
        self.conv_1_1 = nn.Conv2d(512,256,3,1,0)
        self.relu_1_1 = nn.ReLU(inplace=True)
        # 28 x 28

        self.unpool_1 = nn.UpsamplingNearest2d(scale_factor=2)
        # 56 x 56

        # second block
        self.reflecPad_2_1 = nn.ReflectionPad2d((1,1,1,1))
        self.conv_2_1 = nn.Conv2d(256,256,3,1,0)
        self.relu_2_1 = nn.ReLU(inplace=True)
        # 56 x 56

        self.reflecPad_2_2 = nn.ReflectionPad2d((1,1,1,1))
        self.conv_2_2 = nn.Conv2d(256,256,3,1,0)
        self.relu_2_2 = nn.ReLU(inplace=True)

        self.reflecPad_2_3 = nn.ReflectionPad2d((1,1,1,1))
        self.conv_2_3 = nn.Conv2d(256,256,3,1,0)
        self.relu_2_3 = nn.ReLU(inplace=True)

        self.reflecPad_2_4 = nn.ReflectionPad2d((1,1,1,1))
        self.conv_2_4 = nn.Conv2d(256,128,3,1,0)
        self.relu_2_4 = nn.ReLU(inplace=True)

        self.unpool_2 = nn.UpsamplingNearest2d(scale_factor=2)
        # 112 x 112

        # third block
        self.reflecPad_3_1 = nn.ReflectionPad2d((1,1,1,1))
        self.conv_3_1 = nn.Conv2d(128,128,3,1,0)
        self.relu_3_1 = nn.ReLU(inplace=True)
        # 112 x 112

        self.reflecPad_3_2 = nn.ReflectionPad2d((1,1,1,1))
        self.conv_3_2 = nn.Conv2d(128,64,3,1,0)
        self.relu_3_2 = nn.ReLU(inplace=True)

        self.unpool_3 = nn.UpsamplingNearest2d(scale_factor=2)
        # 224 x 224

        # fourth block
        self.reflecPad_4_1 = nn.ReflectionPad2d((1,1,1,1))
        self.conv_4_1 = nn.Conv2d(64,64,3,1,0)
        self.relu_4_1 = nn.ReLU(inplace=True)
        # 224 x 224

        self.reflecPad_4_2 = nn.ReflectionPad2d((1,1,1,1))
        self.conv_4_2 = nn.Conv2d(64,3,3,1,0)

    def forward(self,x):
        # first block
        out = self.reflecPad_1_1(x)
        out = self.conv_1_1(out)
        out = self.relu_1_1(out)
        out = self.unpool_1(out)

        # second block
        out = self.reflecPad_2_1(out)
        out = self.conv_2_1(out)
        out = self.relu_2_1(out)
        out = self.reflecPad_2_2(out)
        out = self.conv_2_2(out)
        out = self.relu_2_2(out)
        out = self.reflecPad_2_3(out)
        out = self.conv_2_3(out)
        out = self.relu_2_3(out)
        out = self.reflecPad_2_4(out)
        out = self.conv_2_4(out)
        out = self.relu_2_4(out)
        out = self.unpool_2(out)

        # third block
        out = self.reflecPad_3_1(out)
        out = self.conv_3_1(out)
        out = self.relu_3_1(out)
        out = self.reflecPad_3_2(out)
        out = self.conv_3_2(out)
        out = self.relu_3_2(out)
        out = self.unpool_3(out)

        # fourth block
        out = self.reflecPad_4_1(out)
        out = self.conv_4_1(out)
        out = self.relu_4_1(out)
        out = self.reflecPad_4_2(out)
        out = self.conv_4_2(out)

        return out