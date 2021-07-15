import argparse
import time
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import sys

sys.path.insert(0, './yolo5')
from yolo5.models.experimental import attempt_load
from yolo5.utils.datasets import LoadStreams, LoadImages,letterbox
from yolo5.utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from yolo5.utils.plots import plot_one_box
from yolo5.utils.torch_utils import select_device, load_classifier, time_synchronized
dictionarInaltimeMinimaSemn = {}#{'Crosswalk': 35,'Pedestrian': 145,
# 'Car':100, 'Stop':40, 'Priority':40, 'Traffic_Light':130}


class Detector:
    def __init__(self,weights='best.pt',imgsz = 640,source='0'):
        global np

        set_logging()
        self.device = select_device('0')
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA

        # Load model
        self.model = attempt_load(weights, map_location=self.device)  # load FP32 model
        self.stride = int(self.model.stride.max())  # model stride
        self.imgsz = check_img_size(imgsz, s=self.stride)  # check img_size
        if self.half:
            self.model.half()  # to FP16

        # vid_path, vid_writer = None, None
        self.source = source
        self.im0s = source
        self.img = letterbox(source, imgsz, stride=self.stride)[0]
        self.img = self.img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        self.img = np.ascontiguousarray(self.img) #114,114

        # Get names and colors
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.names]
        # Run inference
        if self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, imgsz, imgsz).to(self.device).type_as(next(self.model.parameters())))  # run once
        self.label = None
        self.labelSign = None
        self.punctCentral = [0,0]

#source='0', weights="../modelYolo/best.pt", conf=0.25,save_txt=False,imgsz=640
    def detect(self, source='0', weights='best.pt', view_img=False, save_txt=False, imgsz = 640, devicu='',
               augment=False, conf_thres=0.6, iou_thres=0.65):

        im0s=source
        self.labelHeight= 0
        self.label = None
        self.labelSign = None
        self.punctCentral = [0,0]

        img = letterbox(im0s, imgsz, stride=self.stride)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        if True:                           #return path, img, img0, self.cap
        #for path, img, im0s, vid_cap in dataset:  # in loc sa ia imaginea din dataseet, sa ii dam imaginea din main
            img = torch.from_numpy(img).to(self.device)
            img = img.half() if self.half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0

            # import numpy as np
            # self.mask = np.zeros(self.im0s.shape[:2], dtype="uint8")
            # cv2.rectangle(self.mask, (400, 0), (640, 450), 255, -1)
            # self.im0s = cv2.bitwise_and(self.im0s, self.im0s,
            #                        mask=self.mask)  # masca nu are efect, probabil parametrii pentru detectie sunt in tensor
            #print(img.ndimension())
            if img.ndimension() == 3:
                img = img.unsqueeze(0)


            # Inference
            self.t1 = time_synchronized()
            # self.model
            self.pred = self.model(img, augment=False)[0]

            # Apply NMS
            self.pred = non_max_suppression(self.pred, conf_thres, iou_thres, classes=None, agnostic=False)
            self.t2 = time_synchronized()

            # Process detections
            for i, det in enumerate(self.pred):  # detections per image

                self.p, self.s, self.im0, self.frame = None, '', im0s, source
                # print("6, in for, im0 ", type(self.im0), self.im0s.shape)
                #print("7, in for, frame ", type(self.frame), self.frame.shape)

                #p = Path(p)  # to Path
                #save_path = str(save_dir / p.name)  # img.jpg
                #txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
                self.s += '%gx%g ' % img.shape[2:]  # print string
                self.gn = torch.tensor(self.im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], self.im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        self.s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        # if save_txt:  # Write to file
                        #     xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        #     line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                        #     with open(txt_path + '.txt', 'a') as f:
                        #         f.write(('%g ' * len(line)).rstrip() % line + '\n')
                        # view_img = True
                        # if view_img:  # Add bbox to image
                        c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
                        self.punctCentral = [(c1[0]+c2[1])/2, (c1[1]+c2[1])/2]

                        self.label = f'{self.names[int(cls)]} {conf:.2f}'
                        self.labelSign = self.names[int(cls)]

                        self.labelHeight=abs(c1[1]-c2[1])
                        if self.labelHeight==0:
                            self.labelSign=None
                            self.label = None
                        #print("Area {}".format(area))
                        #x,y,w,h=[*xydasdasdsaxy]
                        #print(x,y,w,h)
                        #xywh = xyxy2xywh([x,y,w,h])

                        # testare clasa actiune
                        # print(self.label)

                        plot_one_box(xyxy, self.im0, label=self.label, color=self.colors[int(cls)], line_thickness=3)
                        # TODO: discriminare clasa dupa suprafata; ce face cand vede 2 semne?
                else:
                    self.label=None
                    self.labelSign=None
                # Print time (inference + NMS)
                #print(f'{self.s}Done. ({self.t2 - self.t1:.3f}s)')
                view_img = False
                # Stream results
                if view_img:

                    cv2.imshow("clasificat", im0s)
                    cv2.waitKey(1)  # 1 millisecond

        #print(f'Done. ({time.time() - t0:.3f}s)')

    def getResult(self):
        return self.labelSign, self.labelHeight

    def semnPunctCentral(self):
        return self.punctCentral