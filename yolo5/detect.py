import argparse
import time
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages,letterbox
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized

class Detector:
    def __init__(self,weights='best.pt',imgsz = 640,source='0'):
        global np
        # if ('numpy' in str(type(source))):
        #     print("numpy!")
        #     webcam = False
        # else:
        #     webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith('http') or source.i
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
        #print("init 1 ", type(source), self.source.shape)
        cv2.imshow('bau',self.source)
        self.im0s = source
        #print("init 2 ", type(self.im0s), self.im0s.shape)
        self.img = letterbox(source, imgsz, stride=self.stride)[0]
        #print("init 3 ", type(self.img), self.img.shape)
        self.img = self.img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        # self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB) # nu merge
        #print("init 4 ", type(self.img), self.img.shape)
        self.img = np.ascontiguousarray(self.img) #114,114
        #print("init 5 ", type(self.img), self.img.shape)

        # if webcam:
        #    view_img = check_imshow()
        #    cudnn.benchmark = True  # set True to speed up constant image size inference
        #    dataset = LoadStreams(source, img_size=imgsz, stride=stride)  # aici ia imaginea din feedul camerei
        # else:
        #    save_img = True
        #    dataset = LoadImages(source, img_size=imgsz, stride=stride)

        # for cale,img1,img0,vid_cap in dataset:
        #    cv2.imshow("im1",img1)
        #    cv2.imshow(("img0",img0))
        # Get names and colors
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.names]
        # Run inference
        if self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, imgsz, imgsz).to(self.device).type_as(next(self.model.parameters())))  # run once


#source='0', weights="../modelYolo/best.pt", conf=0.25,save_txt=False,imgsz=640
    def clasifica(self,source='0', weights='best.pt', view_img=False, save_txt=False, imgsz = 640,devicu='',augment=False,conf_thres=0.25,iou_thres=0.45):
        # print("1 sursa ", type(source))
        im0s=source
       # img=source

        img = letterbox(im0s, imgsz, stride=self.stride)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)


        if True:                           #return path, img, img0, self.cap
        #for path, img, im0s, vid_cap in dataset:  # in loc sa ia imaginea din dataseet, sa ii dam imaginea din main
            img = torch.from_numpy(img).to(self.device)
            img = img.half() if self.half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            # cv2.imshow("wqe",img)
            # print(type(img))  # img e un torch.Tensor
            # print(type(self.im0s))  # im0s e numpy.ndarray
            #print(dataset)
            # im0s = im0s[200: , 620:]  # daca se face crop nu se mai afiseaza unde trebuie chenarele
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
            self.model
            self.pred = self.model(img, augment=False)[0]

            # Apply NMS
            self.pred = non_max_suppression(self.pred, conf_thres, iou_thres, classes=None, agnostic=False)

            self.t2 = time_synchronized()



            # Process detections
            for i, det in enumerate(self.pred):  # detections per image

                # if self.webcam:  # batch_size >= 1
                #     self.p, self.s, self.im0, self.frame = self.path[i], '%g: ' % i, self.im0s[i].copy(), self.dataset.count
                # else:
                self.p, self.s, self.im0, self.frame = None, '', im0s, source  # ??? source?
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
                        view_img = True
                        if view_img:  # Add bbox to image
                            self.label = f'{self.names[int(cls)]} {conf:.2f}'
                            # testare clasa actiune
                            print(self.label)
                            plot_one_box(xyxy, self.im0, label=self.label, color=self.colors[int(cls)], line_thickness=3)


                # Print time (inference + NMS)
                print(f'{self.s}Done. ({self.t2 - self.t1:.3f}s)')
                view_img = False
                # Stream results
                if view_img:

                    cv2.imshow("clasificat", im0s)
                    cv2.waitKey(1)  # 1 millisecond




        #print(f'Done. ({time.time() - t0:.3f}s)')

def detect():
    source, weights, view_img, save_txt, imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://'))
    print(type(source))
    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)  # aici ia imaginea din feedul camerei
    else:
        save_img = True
        dataset = LoadImages(source, img_size=imgsz, stride=stride)
    #for cale,img1,img0,vid_cap in dataset:
        #cv2.imshow("im1",img1)
        #cv2.imshow('img' , img0)
    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()
    for path, img, im0s, vid_cap in dataset: # in loc sa ia imaginea din dataseet, sa ii dam imaginea din main
        print("1 d img ", type(img), img.shape)
        img = torch.from_numpy(img).to(device)
        print("2 d img ", type(img), img.shape)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        print("3 d img ", type(img), img.shape)
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        print("4 d img ", type(img), img.shape)
        # cv2.imshow("wqe",img)
        print(type(img))  # img e un torch.Tensor
        print(type(im0s)) # im0s e numpy.ndarray
        print (dataset)
        # im0s = im0s[200: , 620:]  # daca se face crop nu se mai afiseaza unde trebuie chenarele
        # import numpy as np
        # mask = np.zeros(im0s.shape[:2], dtype="uint8")
        # cv2.rectangle(mask, (550,200), (640,450), 255, -1)
        # im0s = cv2.bitwise_and(im0s, im0s, mask=mask)  # masca nu are efect, probabil parametrii pentru detectie sunt in tensor
        cv2.imshow("im0s", im0s)  # im0s e imaginea pe care trebuie facut crop !!!
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or view_img:  # Add bbox to image
                        label = f'{names[int(cls)]} {conf:.2f}'
                        # testare clasa actiune
                        print(label)
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

            # Print time (inference + NMS)
            print(f'{s}Done. ({t2 - t1:.3f}s)')
            view_img=True
            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            save_img  =False # ca sa nu mai salveze videoclipul de fiecare data
            try:
                if save_img:
                    pass
            except:
                save_img=False
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        fourcc = 'mp4v'  # output video codec
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                    vid_writer.write(im0)
    save_img = False
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {save_dir}{s}")

    print(f'Done. ({time.time() - t0:.3f}s)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='data/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    opt = parser.parse_args()
    print(opt)
#    check_requirements()
    cap = cv2.VideoCapture(0)
    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            #from PIL import Image

            #img = Image.open('stop.jpg')
            img = cv2.imread('5.jpg')
            # img.show()
            import numpy
            # np_im = numpy.array(img)
            #cv2.imshow('geee',np_im)
            #img = cv2.imread('stop.jpg',)
            #cv2.imshow('bau2',img)
            # img = cv2.resize(img,(640,480), interpolation=cv2.INTER_AREA)
            #data = numpy.asarray(img)
            # data =img
            #print("forma ", data.shape, type(data))

            detector = Detector(source=img ,weights='../best.pt')
            detector.clasifica(img)
            #detector.clasifica(source=cv2.imread('stop.jpg'))
            #detect()
