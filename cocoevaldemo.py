import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
import skimage.io as io
import pylab

pylab.rcParams['figure.figsize'] = (10.0, 8.0)

annType = 'bbox'
prefix = 'instances'
print('Running demo for *%s* results.' % annType)

# initialize COCO ground truth api
annFile = '/home/wxrui/DATA/coco/coco/annotations/instances_minival2014.json'
cocoGt = COCO(annFile)

# initialize COCO detections api
resFile = 'result.json'
cocoDt = cocoGt.loadRes(resFile)

imgIds = sorted(cocoGt.getImgIds())
imgIds = imgIds[0:100]
imgId = imgIds[np.random.randint(100)]

# running evaluation
cocoEval = COCOeval(cocoGt, cocoDt, annType)
cocoEval.params.imgIds = imgIds
cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()
