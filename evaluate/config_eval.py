# yolo
anchors = [[[0.27884615, 0.21634615], [0.375, 0.47596154], [0.89663462, 0.78365385]],
           [[0.07211538, 0.14663462], [0.14903846, 0.10817308], [0.14182692, 0.28605769]],
           [[0.02403846, 0.03125], [0.03846154, 0.07211538], [0.07932692, 0.05528846]]]

classes_num = 80
iou_thres = 0.5
conf_thres = 0.3
nms_thres = 0.5
bbox_per = 100

# network
batch_size = 4
eval_path = "/home/wxrui/DATA/coco/coco/minival2014.txt"
labelmap = "evaluate/labelmap.json"
img_w = 512
img_h = 512
device_ids = [0, 1, 2, 3]

# checkpoint
backbone_pretrained = ''
checkpoint = 'weights/coco.pth'
