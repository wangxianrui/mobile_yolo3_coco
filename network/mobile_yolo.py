import torch
import torch.nn as nn
import network.mobilenet as mobilenet


class Mobile_YOLO(nn.Module):
    def __init__(self, config):
        super(Mobile_YOLO, self).__init__()
        self.upsample = torch.nn.Upsample(scale_factor=2, mode='nearest')
        #  backbone
        #  load backbone state_dict
        self.backbone = mobilenet.mobilenetv2(config.backbone_pretrained)
        _out_filters = self.backbone.layers_out_filters
        #  embedding0
        final_out_filter0 = len(config.anchors[0]) * (5 + config.classes_num)
        self.embedding0 = self._make_embedding(_out_filters[-1], 512, 256)
        self.result0 = nn.Conv2d(_out_filters[-1], final_out_filter0, kernel_size=1)
        #  embedding1
        final_out_filter1 = len(config.anchors[1]) * (5 + config.classes_num)
        self.embedding1 = self._make_embedding(256 + _out_filters[-2], 256, 128)
        self.result1 = nn.Conv2d(256 + _out_filters[-2], final_out_filter1, kernel_size=1)
        #  embedding2
        final_out_filter2 = len(config.anchors[2]) * (5 + config.classes_num)
        self.embedding2 = self._make_embedding(128 + _out_filters[-3], 128, 64)
        self.result2 = nn.Conv2d(128 + _out_filters[-3], final_out_filter2, kernel_size=1)

    def _make_embedding(self, in_filter, middle_filter1, middle_filter2):
        module = nn.Sequential(
            # op1
            nn.Sequential(
                nn.Conv2d(in_filter, middle_filter1, kernel_size=1, bias=False),
                nn.BatchNorm2d(middle_filter1),
                nn.ReLU6(),
            ),
            nn.Sequential(
                nn.Conv2d(middle_filter1, middle_filter1, kernel_size=3, padding=1, groups=middle_filter1, bias=False),
                nn.BatchNorm2d(middle_filter1),
                nn.ReLU6(),
            ),
            nn.Sequential(
                nn.Conv2d(middle_filter1, in_filter, kernel_size=1, bias=False),
                nn.BatchNorm2d(in_filter),
            ),
            # op2
            nn.Sequential(
                nn.Conv2d(in_filter, middle_filter2, kernel_size=1, bias=False),
                nn.BatchNorm2d(middle_filter2),
                nn.ReLU6(),
            ),
            nn.Sequential(
                nn.Conv2d(middle_filter2, middle_filter2, kernel_size=3, padding=1, groups=middle_filter2, bias=False),
                nn.BatchNorm2d(middle_filter2),
                nn.ReLU6(),
            ),
            nn.Sequential(
                nn.Conv2d(middle_filter2, in_filter, kernel_size=1, bias=False),
                nn.BatchNorm2d(in_filter),
            )
        )
        return module

    def forward(self, x):
        def _branch(_embedding, _in):
            for i, e in enumerate(_embedding):
                _in = e(_in)
                if i == 3:
                    out_branch = _in
            return _in, out_branch

        #  backbone
        x52, x26, x13 = self.backbone(x)
        #  yolo branch 0
        x0_in = x13
        out0, out0_branch = _branch(self.embedding0, x0_in)
        out0 = self.result0(out0)
        #  yolo branch 1
        x1_in = self.upsample(out0_branch)
        x1_in = torch.cat([x1_in, x26], 1)
        out1, out1_branch = _branch(self.embedding1, x1_in)
        out1 = self.result1(out1)
        #  yolo branch 2
        x2_in = self.upsample(out1_branch)
        x2_in = torch.cat([x2_in, x52], 1)
        out2, out2_branch = _branch(self.embedding2, x2_in)
        out2 = self.result2(out2)
        return out0, out1, out2


if __name__ == "__main__":
    import training.config_train as config

    m = Mobile_YOLO(config).cuda()
    torch.save(m.state_dict(), 'test.pth')
    x = torch.randn(1, 3, 416, 416).cuda()
    y0, y1, y2 = m(x)
    print(y0.size())
    print(y1.size())
    print(y2.size())
