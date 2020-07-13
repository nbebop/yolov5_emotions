import argparse
import os

import torch.backends.cudnn as cudnn
import pandas as pd

#from emotion_prediction_ops.utils import google_utils
#from emotion_prediction_ops.utils.datasets import *
#from emotion_prediction_ops.utils.utils import *
from utils import google_utils
from utils.datasets import *
from utils.utils import *

def detect_obj(weights='weights/yolov5x.pt', output='../frames_output', img_size=640,
            conf_thres=0.4, iou_thres=0.6, fourcc='mp4v', device='', classes=None,
            agnostic_nms=False, augment=False, save_img=False, frame_folder='../frames',
            img_to_save=['kTHNpusq654_11.jpg', 'OZLUa8JUR18_55.jpg', 'hT_nvWreIhg_67.jpg',
            'Ahha3Cqe_fk_34.jpg', '8SbUC-UaAxE_108.jpg', 'J9NQFACZYEU_23.jpg', 'kTHNpusq654_0.jpg', 'kTHNpusq654_98.jpg', 'kTHNpusq654_108.jpg']):

    imgsz = img_size

    ##### prepare model #####
    device = torch_utils.select_device(device)
    if not os.path.exists(output): os.makedirs(output)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    google_utils.attempt_download(weights)
    model = torch.load(weights, map_location=device)['model'].float()  # load to FP32

    # update model if SourceChangeWarning
    #torch.save(torch.load(weights, map_location=device), weights)
    #model.fuse()
    model.to(device).eval()

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names #classnames
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    df_classes = tuple(names)
    final_predictions = list()
    obj_df_colnames = pd.concat([pd.Series(['media_id', 'frame_nr']), pd.Series(df_classes)])

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once

    #####  Loop on video folder then on frames #####
    for video_folder in os.listdir(frame_folder):
        media_id = video_folder
        for frame in os.listdir(os.path.join(frame_folder, video_folder)):

            temp_result = np.zeros(len(names))
            frame_nr = frame[frame.rfind('_')+1:-4]

            img_name = os.path.join(frame_folder, video_folder, frame)
            imgsz = check_img_size(imgsz, s=model.model[-1].stride.max())  # check img_size
            save_path = os.path.join(output, frame)
            save_img=True if frame in img_to_save else False
            if half: model.half()  # to FP16

            #read img
            img0 = cv2.imread(img_name)  # BGR
            # Padded resize
            img = letterbox(img0, new_shape=img_size)[0]
            # Convert
            img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
            img = np.ascontiguousarray(img)

            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3: img = img.unsqueeze(0)

            # Inference
            t1 = torch_utils.time_synchronized()
            pred = model(img, augment=augment)[0]
            # Apply NMS - bounding boxes for prediction
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes=classes, agnostic=agnostic_nms)
            t2 = torch_utils.time_synchronized()

            # Print time (inference + NMS) print('Done. (%.3fs)' % (t2 - t1))

            # Process detections
            for i, det in enumerate(pred):  # detections per image
                s = '%gx%g ' % img.shape[2:]
                gn = torch.tensor(img0.shape)[[1, 0, 1, 0]]  # normalization gain whwh

                if det is not None and len(det):
                    # Rescale boxes from img_size to img0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

                    #for c in det[:, -1].unique():
                    #    n = (det[:, -1] == c).sum()  # nr of detections per class
                    #    s += '%g %ss, ' % (n, names[int(c)])  # add to string


                    for *xyxy, conf, cl in det:
                        if temp_result[int(cl)] <= float(conf):
                            temp_result[int(cl)] = float(conf)
                        if save_img:
                            label = '%s %.2f' % (names[int(cl)], conf)
                            plot_one_box(xyxy, img0, label=label, color=colors[int(cl)], line_thickness=3)
                    #temp_result = np.divide(temp_result, len(det))
                    print(temp_result)

            #print(s)
            print('done: ', save_path)

            if save_img:
                cv2.imwrite(save_path, img0)
                print('saved: ', save_path)
            final_predictions.append([media_id, frame_nr]+temp_result.tolist())
            break
        break
    df = pd.DataFrame(final_predictions, columns=obj_df_colnames)
    df.to_csv('obj_det_values.csv', index=False)

#if __name__ == '__main__':
#    with torch.no_grad():
#        detect_obj(img_size=416, conf_thres=0.000001)

        # Update all models
        # for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt', 'yolov3-spp.pt']:
        #    detect()
        #    create_pretrained(opt.weights, opt.weights)
