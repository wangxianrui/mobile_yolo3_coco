import os
import sys

sys.path.append('.')
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator
import random
import torch
import detect_img.config_detect as config
import network.mobile_yolo as mobile_yolo
import network.yolo_loss as yolo_loss
import network.utils as utils

cmap = plt.get_cmap('tab20b')
colors = [cmap(i) for i in np.linspace(0, 1, 20)]


def detect():
    # net
    net = mobile_yolo.Mobile_YOLO(config)
    net = torch.nn.DataParallel(net.cuda())
    net.eval()

    # checkpoint
    net.load_state_dict(torch.load(config.checkpoint))

    yolo_losses = []
    for i in range(3):
        yolo_losses.append(
            yolo_loss.YOLOLoss(config.anchors[i], config.classes_num, (config.img_w, config.img_h)))

    # prepare images path
    images_name = os.listdir(config.image_path)
    images_path = [os.path.join(config.image_path, name) for name in images_name]
    if len(images_path) == 0:
        raise Exception("no image found in {}".format(config.image_path))

    # Start inference
    batch_size = config.batch_size
    for step in range(0, len(images_path), batch_size):
        # preprocess
        images = []
        images_origin = []
        for path in images_path[step * batch_size: (step + 1) * batch_size]:
            print("processing: {}".format(path))
            image = cv2.imread(path, cv2.IMREAD_COLOR)
            if image is None:
                print("read path error: {}. skip it.".format(path))
                continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            images_origin.append(image)  # keep for save result
            image = cv2.resize(image, (config.img_w, config.img_h), interpolation=cv2.INTER_LINEAR)
            image = image.astype(np.float32)
            image /= 255.0
            image = np.transpose(image, (2, 0, 1))
            image = image.astype(np.float32)
            images.append(image)
        images = np.asarray(images)
        images = torch.from_numpy(images).cuda()

        # inference
        with torch.no_grad():
            outputs = net(images)
            output_list = []
            for i in range(3):
                output_list.append(yolo_losses[i](outputs[i]))
            output = torch.cat(output_list, 1)
            batch_detections = utils.non_max_suppression(output, config.classes_num, config.conf_thres)

        # write result images. Draw bounding boxes and labels of detections
        classes = open(config.classes_names_path, "r").read().split("\n")[:-1]
        if not os.path.isdir(config.save_path):
            os.makedirs(config.save_path)
        for idx, detections in enumerate(batch_detections):
            plt.figure()
            fig, ax = plt.subplots(1)
            ax.imshow(images_origin[idx])
            if detections is not None:
                unique_labels = detections[:, -1].cpu().unique()
                n_cls_preds = len(unique_labels)
                bbox_colors = random.sample(colors, n_cls_preds)
                for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
                    color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
                    # Rescale coordinates to original dimensions
                    ori_h, ori_w = images_origin[idx].shape[:2]
                    pre_h, pre_w = config.img_h, config.img_w
                    box_h = ((y2 - y1) / pre_h) * ori_h
                    box_w = ((x2 - x1) / pre_w) * ori_w
                    y1 = (y1 / pre_h) * ori_h
                    x1 = (x1 / pre_w) * ori_w
                    # Create a Rectangle patch
                    bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, edgecolor=color, facecolor='none')
                    # Add the bbox to the plot
                    ax.add_patch(bbox)
                    # Add label
                    plt.text(x1, y1, s=classes[int(cls_pred)], color='white',
                             verticalalignment='top',
                             bbox={'color': color, 'pad': 0})
            # Save generated image with detections
            plt.axis('off')
            plt.gca().xaxis.set_major_locator(NullLocator())
            plt.gca().yaxis.set_major_locator(NullLocator())
            plt.savefig(config.save_path + '/{}_{}.jpg'.format(step, idx), bbox_inches='tight', pad_inches=0.0)
            plt.close()


#

if __name__ == '__main__':
    # CUDA_VISIBLE_DEVICES
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, config.device_ids))
    detect()
