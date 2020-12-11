import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class HorizonNet_v2(nn.Module):
    '''
    The architecture of HorizonNet_v2.
    input : (n x 3 x 512 x 1024) Panorama and (n x 3 x 512 x 1024) Manhattan Line map.
    output: (n x 3 x 1 x 1024)
    args:
        out_scale : Dowsampling scale.
        step_cols : The output scale of RNNs.
        channels : The number of output channels of encoder.
        trans_blocks : The Module that operated at output blocks of encoder.
    '''
    def __init__(self):
        super(HorizonNet_v2, self).__init__()
        self.out_scale = 2
        self.step_cols = 4
        self.rnn_input_size = 2048
        self.rnn_hidden_size = 512
        # Encoder for panorama
        self.feature_extractor = Encoder()
        # Encoder for Manhattan line map
        self.feature_extractor_line = Encoder()
        self.channels = [64, 128, 256, 512]

        # Downsampling feature maps from 4 blocks
        self.trans_blocks=[]
        for c in self.channels:
            self.trans_blocks.append(Transform(c, c//2, c//self.out_scale))
        self.trans_blocks=nn.ModuleList(self.trans_blocks)


        self.bi_rnn = nn.LSTM(input_size=self.rnn_input_size,hidden_size=self.rnn_hidden_size,num_layers=2,dropout=0.5,batch_first=False,bidirectional=True)
        self.drop_out = nn.Dropout(0.5)
        self.linear = nn.Linear(in_features=2 * self.rnn_hidden_size, out_features=3 * self.step_cols)

        # Initialize the parameters that adopted by original HorizonNet.
        self.linear.bias.data[0 * self.step_cols:1 * self.step_cols].fill_(-1)
        self.linear.bias.data[1 * self.step_cols:2 * self.step_cols].fill_(-0.478)
        self.linear.bias.data[2 * self.step_cols:3 * self.step_cols].fill_(0.425)

        self.x_mean.requires_grad = False
        self.x_std.requires_grad = False


    # Normalization values adopted by original HorizonNet.
    x_mean = torch.FloatTensor(np.array([0.485, 0.456, 0.406])[None, :, None, None])
    x_std = torch.FloatTensor(np.array([0.229, 0.224, 0.225])[None, :, None, None])
    def _prepare_x(self, x):
        '''
        Normalize the input data.
        '''
        if self.x_mean.device != x.device:
            self.x_mean = self.x_mean.to(x.device)
            self.x_std = self.x_std.to(x.device)
        return (x[:, :3] - self.x_mean) / self.x_std

    def forward(self, x, line):

        if x.shape[2] != 512 or x.shape[3] != 1024 or line.shape[2] != 512 or line.shape[3] != 1024:
            raise NotImplementedError()

        # Encoders: parallel input with two ResNet34
        x = self._prepare_x(x) # Normalization
        # Extract features from two images.
        conv_list = self.feature_extractor(x)
        conv_list_line = self.feature_extractor_line(line)
        # Downsample and concatenate
        out_c=x.shape[3] // self.step_cols
        feature = torch.cat([ f(b, out_c).reshape(x.shape[0], -1, out_c) for f, b in zip(self.trans_blocks, conv_list)], dim=1)
        feature_line = torch.cat([ f(b, out_c).reshape(line.shape[0], -1, out_c) for f, b in zip(self.trans_blocks, conv_list_line)], dim=1)
        feature = torch.cat((feature, feature_line), dim=1)

        # Decoder: Rnn
        feature = feature.permute(2, 0, 1)  # [w, b, c*h]
        output, hidden = self.bi_rnn(feature)  # [seq_len, b, num_directions * hidden_size]
        output = self.drop_out(output)
        output = self.linear(output)  # [seq_len, b, 3 * step_cols]
        output = output.view(output.shape[0], output.shape[1], 3, self.step_cols)  # [seq_len, b, 3, step_cols]
        output = output.permute(1, 2, 0, 3)  # [b, 3, seq_len, step_cols]
        output = output.contiguous().view(output.shape[0], 3, -1)  # [b, 3, seq_len*step_cols]
        # output.shape => B x 3 x W
        cor = output[:, :1]  # B x 1 x W
        bon = output[:, 1:]  # B x 2 x W
        return bon, cor



class Encoder(nn.Module):
    '''
    The architecture of Encoder that adopted ResNet34 as backbone and output the feature map from
    4 blocks.
    '''
    def __init__(self):
        super(Encoder, self).__init__()
        # Get the backbone of encoder
        self.encoder = models.resnet34(pretrained=True)

    def forward(self, x):
        # Collect both high resolution information and low resolution information
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        x = self.encoder.relu(x)
        x = self.encoder.maxpool(x)
        '''
        The output shape of 4 blocks:
        1:  torch.Size([1, 64, 128, 256])
        2:  torch.Size([1, 128, 64, 128])
        3:  torch.Size([1, 256, 32, 64])
        4:  torch.Size([1, 512, 16, 32])
        '''
        block1 = self.encoder.layer1(x)
        block2 = self.encoder.layer2(block1)
        block3 = self.encoder.layer3(block2)
        block4 = self.encoder.layer4(block3)

        features=[block1, block2, block3, block4]
        return features



class Transform(nn.Module):
    '''
    This Module aims at uniting the output feature maps of four output blocks of encoder.
    '''
    def __init__(self,in_c, out_c, out_fin, ks=3 ):
        super(Transform, self).__init__()
        assert ks % 2 == 1
        self.seq=nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=ks, stride=(2, 1), padding=ks // 2),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_c, out_c, kernel_size=ks, stride=(2, 1), padding=ks // 2),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_c, out_c//2, kernel_size=ks, stride=(2, 1), padding=ks // 2),
            nn.BatchNorm2d(out_c//2),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_c//2, out_fin, kernel_size=ks, stride=(2, 1), padding=ks // 2),
            nn.BatchNorm2d(out_fin),
            nn.ReLU(inplace=True),
        )
    def forward(self, x, out=256):
        x = self.seq(x)
        x = torch.cat([x[..., -1:], x, x[..., :1]], 3)
        x = F.interpolate(x, size=(x.shape[2], out + out // x.shape[3]), mode='bilinear', align_corners=False)
        x = x[..., (out // x.shape[3]) : -(out // x.shape[3])]
        return x