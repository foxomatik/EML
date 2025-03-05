import torch
import torch.nn as nn
import torch.nn.functional as F

# Fine-tuned Model
class TinyYoloV2(nn.Module):
    def __init__(self, num_classes=20):
        super().__init__()

        anchors = ((1.08, 1.19),
            (3.42, 4.41),
            (6.63, 11.38),
            (9.42, 5.11),
            (16.62, 10.52),)

        self.register_buffer("anchors", torch.tensor(anchors))
        self.num_classes = num_classes

        self.pad = nn.ReflectionPad2d((0, 1, 0, 1))

        self.conv1 = nn.Conv2d(3, 16, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)

        self.conv2 = nn.Conv2d(16, 32, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32, 64, 3, 1, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(64)

        self.conv4 = nn.Conv2d(64, 128, 3, 1, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(128)

        self.conv5 = nn.Conv2d(128, 256, 3, 1, 1, bias=False)
        self.bn5 = nn.BatchNorm2d(256)

        self.conv6 = nn.Conv2d(256, 512, 3, 1, 1, bias=False)
        self.bn6 = nn.BatchNorm2d(512)

        self.conv7 = nn.Conv2d(512, 1024, 3, 1, 1, bias=False)
        self.bn7 = nn.BatchNorm2d(1024)

        self.conv8 = nn.Conv2d(1024, 1024, 3, 1, 1, bias=False)
        self.bn8 = nn.BatchNorm2d(1024)

        self.conv9 = nn.Conv2d(1024, len(anchors) * (5 + num_classes), 1, 1, 0)

    def forward(self, x, yolo=True):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.leaky_relu(x, negative_slope=0.1, inplace=True)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.leaky_relu(x, negative_slope=0.1, inplace=True)

        x = self.conv3(x)
        x = self.bn3(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.leaky_relu(x, negative_slope=0.1, inplace=True)

        x = self.conv4(x)
        x = self.bn4(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.leaky_relu(x, negative_slope=0.1, inplace=True)

        x = self.conv5(x)
        x = self.bn5(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.leaky_relu(x, negative_slope=0.1, inplace=True)

        x = self.conv6(x)
        x = self.bn6(x)
        x = self.pad(x)
        x = F.max_pool2d(x, kernel_size=2, stride=1)
        x = F.leaky_relu(x, negative_slope=0.1, inplace=True)

        x = self.conv7(x)
        x = self.bn7(x)
        x = F.leaky_relu(x, negative_slope=0.1, inplace=True)

        x = self.conv8(x)
        x = self.bn8(x)
        x = F.leaky_relu(x, negative_slope=0.1, inplace=True)

        x = self.conv9(x)
        if yolo:
            nB, _, nH, nW = x.shape

            x = x.view(nB, self.anchors.shape[0], -1, nH, nW).permute(0, 1, 3, 4, 2)

            anchors = self.anchors.to(dtype=x.dtype, device=x.device)
            range_y, range_x, = torch.meshgrid(
                    torch.arange(nH, dtype=x.dtype, device=x.device),
                    torch.arange(nW, dtype=x.dtype, device=x.device)
            )
            anchor_x, anchor_y = anchors[:, 0], anchors[:, 1]

            x = torch.cat([
                (x[:, :, :, :, 0:1].sigmoid() + range_x[None, None, :, :, None]) / nW, #x center
                (x[:, :, :, :, 1:2].sigmoid() + range_y[None, None, :, :, None]) / nH, #y center
                (x[:, :, :, :, 2:3].exp() * anchor_x[None, :, None, None, None])/ nW, # Width
                (x[:, :, :, :, 3:4].exp() * anchor_y[None, :, None, None, None]) /nH, # Height
                x[:, :, :, :, 4:5].sigmoid(), #confidence
                x[:, :, :, :, 5:].softmax(-1),], -1)
        
        return x

class PrunedTinyYoloV2(TinyYoloV2):
    def __init__(self, num_classes=20):
        super(TinyYoloV2, self).__init__()
        self.num_classes = num_classes
        anchors = ((1.08, 1.19),
            (3.42, 4.41),
            (6.63, 11.38),
            (9.42, 5.11),
            (16.62, 10.52),)

        self.register_buffer("anchors", torch.tensor(anchors))
        self.num_classes = num_classes

        self.pad = nn.ReflectionPad2d((0, 1, 0, 1))

        self.conv1 = nn.Conv2d(3, 16, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)

        self.conv2 = nn.Conv2d(16, 32, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32, 64, 3, 1, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(64)

        self.conv4 = nn.Conv2d(64, 128, 3, 1, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(128)

        self.conv5 = nn.Conv2d(128, 256, 3, 1, 1, bias=False)
        self.bn5 = nn.BatchNorm2d(256)

        self.conv6 = nn.Conv2d(256, 512, 3, 1, 1, bias=False)
        self.bn6 = nn.BatchNorm2d(512)

        self.conv7 = nn.Conv2d(512, 1024, 3, 1, 1, bias=False)
        self.bn7 = nn.BatchNorm2d(1024)

        self.conv8 = nn.Conv2d(1024, 1024, 3, 1, 1, bias=False)
        self.bn8 = nn.BatchNorm2d(1024)

        self.conv9 = nn.Conv2d(1024, len(anchors) * (5 + num_classes), 1, 1, 0)

        self._register_load_state_dict_pre_hook(self._sd_hook)

    def _sd_hook(self, state_dict, prefix, *_):
        anchors = ((1.08, 1.19),
            (3.42, 4.41),
            (6.63, 11.38),
            (9.42, 5.11),
            (16.62, 10.52),)
        for key in state_dict:
            if 'conv' in key and 'weight' in key:
                n = int(key.split('conv')[1].split('.weight')[0])

                dim_in = state_dict[f'conv{n}.weight'].shape[1]
                dim_out = state_dict[f'conv{n}.weight'].shape[0]
    
                conv = nn.Conv2d(dim_in, dim_out, 3, 1, padding=1, bias=False)
                if n == 1: self.conv1 = conv
                elif n == 2: self.conv2 = conv
                elif n == 3: self.conv3 = conv
                elif n == 4: self.conv4 = conv
                elif n == 5: self.conv5 = conv
                elif n == 6: self.conv6 = conv
                elif n == 7: self.conv7 = conv
                elif n == 8: self.conv8 = conv
                elif n == 9: self.conv9 = nn.Conv2d(dim_in, len(anchors) * (5 + self.num_classes), 1, 1, 0)
            
            if 'bn' in key and 'weight' in key:
                n = int(key.split('bn')[1].split('.weight')[0])

                dim_in = state_dict[f'bn{n}.weight'].shape[0]
    
                bn = nn.BatchNorm2d(dim_in)
                if n == 1: self.bn1 = bn
                elif n == 2: self.bn2 = bn
                elif n == 3: self.bn3 = bn
                elif n == 4: self.bn4 = bn
                elif n == 5: self.bn5 = bn
                elif n == 6: self.bn6 = bn
                elif n == 7: self.bn7 = bn
                elif n == 8: self.bn8 = bn
        pass

# Fused + Fine-tuned Model
class FusedTinyYoloV2(nn.Module):
    def __init__(self, num_classes=20):
        super().__init__()

        anchors = ((1.08, 1.19),
            (3.42, 4.41),
            (6.63, 11.38),
            (9.42, 5.11),
            (16.62, 10.52),)

        self.register_buffer("anchors", torch.tensor(anchors))
        self.num_classes = num_classes

        self.pad = nn.ReflectionPad2d((0, 1, 0, 1))

        self.conv1 = nn.Conv2d(3, 16, 3, 1, 1)

        self.conv2 = nn.Conv2d(16, 32, 3, 1, 1)

        self.conv3 = nn.Conv2d(32, 64, 3, 1, 1)

        self.conv4 = nn.Conv2d(64, 128, 3, 1, 1)

        self.conv5 = nn.Conv2d(128, 256, 3, 1, 1)

        self.conv6 = nn.Conv2d(256, 512, 3, 1, 1)

        self.conv7 = nn.Conv2d(512, 1024, 3, 1, 1)

        self.conv8 = nn.Conv2d(1024, 1024, 3, 1, 1)

        self.conv9 = nn.Conv2d(1024, len(anchors) * (5 + num_classes), 1, 1, 0)

    def forward(self, x, yolo=True):

        x = self.conv1(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.leaky_relu(x, negative_slope=0.1, inplace=True)

        x = self.conv2(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.leaky_relu(x, negative_slope=0.1, inplace=True)

        x = self.conv3(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.leaky_relu(x, negative_slope=0.1, inplace=True)

        x = self.conv4(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.leaky_relu(x, negative_slope=0.1, inplace=True)

        x = self.conv5(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.leaky_relu(x, negative_slope=0.1, inplace=True)

        x = self.conv6(x)
        x = self.pad(x)
        x = F.max_pool2d(x, kernel_size=2, stride=1)
        x = F.leaky_relu(x, negative_slope=0.1, inplace=True)

        x = self.conv7(x)
        x = F.leaky_relu(x, negative_slope=0.1, inplace=True)

        x = self.conv8(x)
        x = F.leaky_relu(x, negative_slope=0.1, inplace=True)

        x = self.conv9(x)
        if yolo:
            nB, _, nH, nW = x.shape

            x = x.view(nB, self.anchors.shape[0], -1, nH, nW).permute(0, 1, 3, 4, 2)

            anchors = self.anchors.to(dtype=x.dtype, device=x.device)
            range_y, range_x, = torch.meshgrid(
                    torch.arange(nH, dtype=x.dtype, device=x.device),
                    torch.arange(nW, dtype=x.dtype, device=x.device)
            )
            anchor_x, anchor_y = anchors[:, 0], anchors[:, 1]

            x = torch.cat([
                (x[:, :, :, :, 0:1].sigmoid() + range_x[None, None, :, :, None]) / nW, #x center
                (x[:, :, :, :, 1:2].sigmoid() + range_y[None, None, :, :, None]) / nH, #y center
                (x[:, :, :, :, 2:3].exp() * anchor_x[None, :, None, None, None])/ nW, # Width
                (x[:, :, :, :, 3:4].exp() * anchor_y[None, :, None, None, None]) /nH, # Height
                x[:, :, :, :, 4:5].sigmoid(), #confidence
                x[:, :, :, :, 5:].softmax(-1),], -1)

        return x

class PrunedFusedTinyYoloV2(FusedTinyYoloV2):
    def __init__(self, num_classes=20):
        super(FusedTinyYoloV2, self).__init__()
        self.num_classes = num_classes
        anchors = ((1.08, 1.19),
            (3.42, 4.41),
            (6.63, 11.38),
            (9.42, 5.11),
            (16.62, 10.52),)

        self.register_buffer("anchors", torch.tensor(anchors))
        self.num_classes = num_classes

        self.pad = nn.ReflectionPad2d((0, 1, 0, 1))

        self.conv1 = nn.Conv2d(3, 16, 3, 1, 1)

        self.conv2 = nn.Conv2d(16, 32, 3, 1, 1)

        self.conv3 = nn.Conv2d(32, 64, 3, 1, 1)

        self.conv4 = nn.Conv2d(64, 128, 3, 1, 1)

        self.conv5 = nn.Conv2d(128, 256, 3, 1, 1)

        self.conv6 = nn.Conv2d(256, 512, 3, 1, 1)

        self.conv7 = nn.Conv2d(512, 1024, 3, 1, 1)

        self.conv8 = nn.Conv2d(1024, 1024, 3, 1, 1)

        self.conv9 = nn.Conv2d(1024, len(anchors) * (5 + num_classes), 1, 1, 0)

        self._register_load_state_dict_pre_hook(self._sd_hook)

    def _sd_hook(self, state_dict, prefix, *_):
        anchors = ((1.08, 1.19),
            (3.42, 4.41),
            (6.63, 11.38),
            (9.42, 5.11),
            (16.62, 10.52),)
        for key in state_dict:
            if 'conv' in key and 'weight' in key:
                n = int(key.split('conv')[1].split('.weight')[0])

                dim_in = state_dict[f'conv{n}.weight'].shape[1]
                dim_out = state_dict[f'conv{n}.weight'].shape[0]
    
                conv = nn.Conv2d(dim_in, dim_out, 3, 1, padding=1)
                if n == 1: self.conv1 = conv
                elif n == 2: self.conv2 = conv
                elif n == 3: self.conv3 = conv
                elif n == 4: self.conv4 = conv
                elif n == 5: self.conv5 = conv
                elif n == 6: self.conv6 = conv
                elif n == 7: self.conv7 = conv
                elif n == 8: self.conv8 = conv
                elif n == 9: self.conv9 = nn.Conv2d(dim_in, len(anchors) * (5 + self.num_classes), 1, 1, 0)
            
        pass