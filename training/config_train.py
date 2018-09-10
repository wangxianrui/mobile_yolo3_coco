from tensorboardX import SummaryWriter

# log
writer = SummaryWriter('log')

# backbone
backbone_pretrained = "weights/mobilenetv2.pth"
# backbone_pretrained = "weights/darknet53.pth"
# backbone_pretrained = "weights/shufflenetv2.pth"

# checkpoints
start_epoch = 0
checkpoint = ''

# yolo
anchors = [[[0.27884615, 0.21634615], [0.375, 0.47596154], [0.89663462, 0.78365385]],
           [[0.07211538, 0.14663462], [0.14903846, 0.10817308], [0.14182692, 0.28605769]],
           [[0.02403846, 0.03125], [0.03846154, 0.07211538], [0.07932692, 0.05528846]]]
classes_num = 80

# optimizer
optim_type = "sgd"
lr = 1e-2
backbone_lr = 0.1 * lr
weight_decay = 4e-05
milestones = [20, 40, 60, 80, 100]
epochs = milestones[-1]

# network
batch_size = 8
train_path = "/home/wxrui/DATA/coco/coco/train.txt"
img_w = 832
img_h = 832
device_ids = [0, 1, 2, 3]

# save
save_path = 'output'
