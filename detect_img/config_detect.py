# yolo
anchors = [[[116, 90], [156, 198], [373, 326]],
           [[30, 61], [62, 45], [59, 119]],
           [[10, 13], [16, 30], [33, 23]]]
classes_num = 80
conf_thres = 0.5

# network
batch_size = 16
img_h = 832
img_w = 832
device_ids = [0, 1, 2, 3]

# path
classes_names_path = "/home/wxrui/DATA/coco/coco.names"
image_path = 'detect_img/images'
save_path = 'detect_img/output'

# checkpoint
backbone_pretrained = ''
checkpoint = 'weights/coco.pth'
