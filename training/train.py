import sys
import os
import time
import torch
import torch.utils.data
import torch.optim

sys.path.append('.')
import training.config_train as config
import network.mobile_yolo as mobile_yolo
import network.shuffle_yolo as shuffle_yolo
import network.dark_yolo as dark_yolo
import network.yolo_loss as yolo_loss
import data.coco_dataset as coco_dataset


def train():
    # DataLoader
    dataloader = torch.utils.data.DataLoader(
        coco_dataset.COCODataset(config.train_path, (config.img_w, config.img_h), is_training=True),
        batch_size=config.batch_size, shuffle=True, pin_memory=True)

    # net and optimizer
    net = mobile_yolo.Mobile_YOLO(config)
    optimizer = _get_optimizer(config, net)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=config.milestones, gamma=0.1)
    net = torch.nn.DataParallel(net.cuda())

    # checkpoints
    if config.checkpoint:
        print('lodding checkpoint:', config.checkpoint)
        checkpoint = torch.load(config.checkpoint)
        net.load_state_dict(checkpoint)

    yolo_losses = []
    for i in range(3):
        yolo_losses.append(
            yolo_loss.YOLOLoss(config.anchors[i], config.classes_num, (config.img_w, config.img_h)))

    print('Start training...')

    net.train()
    global_step = config.start_epoch * len(dataloader)

    for epoch in range(config.start_epoch, config.epochs):
        for step, samples in enumerate(dataloader):
            images, labels = samples["image"], samples["label"]
            start_time = time.time()

            # Forward and backward
            outputs = net(images)
            losses_name = ["total_loss", "x", "y", "w", "h", "conf", "cls"]
            losses = [[] for i in range(len(losses_name))]
            for i in range(3):
                _loss_item = yolo_losses[i](outputs[i], labels)
                for j, l in enumerate(_loss_item):
                    losses[j].append(l)
            losses = [sum(l) for l in losses]
            loss = losses[0]
            print(losses)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step > 0 and step % 10 == 0:
                _loss = loss.item()
                duration = float(time.time() - start_time)
                example_per_second = config.batch_size / duration
                lr = optimizer.param_groups[0]['lr']
                print("epoch [%.3d] iter = %d loss = %.2f example/sec = %.3f lr = %.5f " %
                      (epoch, step, _loss, example_per_second, lr))
                for i, name in enumerate(losses_name):
                    value = _loss if i == 0 else losses[i]
                    config.writer.add_scalar(name, value, global_step)
            if step > 0 and step % 1000 == 0:
                print('saving model to %s/model%s.pth' % (config.save_path, global_step))
                torch.save(net.state_dict(), '%s/model%s.pth' % (config.save_path, global_step))

            global_step += 1
        lr_scheduler.step()


def _get_optimizer(config, net):
    base_params = list(map(id, net.backbone.parameters()))
    logits_params = filter(lambda p: id(p) not in base_params, net.parameters())
    params = [
        {"params": logits_params, "lr": config.lr},
        {"params": net.backbone.parameters(), "lr": config.backbone_lr},
    ]

    # Initialize optimizer class
    if config.optim_type == "adam":
        optimizer = torch.optim.Adam(params, weight_decay=config.weight_decay)
    elif config.optim_type == "amsgrad":
        optimizer = torch.optim.Adam(params, weight_decay=config.weight_decay, amsgrad=True)
    elif config.optim_type == "rmsprop":
        optimizer = torch.optim.RMSprop(params, weight_decay=config.weight_decay)
    else:
        optimizer = torch.optim.SGD(params, momentum=0.9, weight_decay=config.weight_decay)

    return optimizer


if __name__ == '__main__':
    # CUDA_VISIBLE_DEVICES
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, config.device_ids))
    train()
