import os
import cv2
import numpy as np
from sort import Sort
import balloon

weights_path = 'D:\CH2\MyWork\Mask-Rcnn-Sort\mask_rcnn_balloon.h5'
imagepath = ''
logpath = balloon.DEFAULT_LOGS_DIR
videopath = 'D:\CH2\MyWork\Mask-Rcnn-Sort\CatAndBalloon.mp4'
sort_max_age = 5
sort_min_hit = 3


class InferenceConfig(balloon.BalloonConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
config1= InferenceConfig()
config1.display()
def Main():
    # Load weights
    model = balloon.modellib.MaskRCNN(mode="inference", config=config1, model_dir=logpath)
    print("Loading weights ", weights_path)
    model.load_weights(weights_path, by_name=True)
    mot_tracker = Sort(sort_max_age, sort_min_hit)

    # evaluate
    vcapture = cv2.VideoCapture(videopath)
    width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = vcapture.get(cv2.CAP_PROP_FPS)
    # Define codec and create video writer
    file_name = "splash_{:%Y%m%dT%H%M%S}.avi".format(balloon.datetime.datetime.now())
    vwriter = cv2.VideoWriter(file_name,
                              cv2.VideoWriter_fourcc(*'MJPG'),
                              fps, (width, height))
    colours = np.random.rand(32, 3) * 255

    count = 0
    success = True
    while success:
        print("frame: ", count)
        # Read next image
        success, image = vcapture.read()
        if success:
            # Detect objects
            r = model.detect([image], verbose=0)[0]
            result = r['rois']
            result = np.array(result)  # 变成array矩阵
            print(result)
            det = result[:, 0:5]
            '''
            #print(det)
            det[:, 0] = det[:, 0] * width
            det[:, 1] = det[:, 1] * height
            det[:, 2] = det[:, 2] * width
            det[:, 3] = det[:, 3] * height
            print(det)
            将图片位置交给sort处理，det格式为[[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
            '''
            trackers = mot_tracker.update(det)
            print(trackers)
            for d in trackers:
                xmin = int(d[1])
                ymin = int(d[0])
                xmax = int(d[3])
                ymax = int(d[2])
                label = int(d[4])
                cv2.rectangle(image, (xmin, ymin), (xmax, ymax),
                              (int(colours[label % 32, 0]), int(colours[label % 32, 1]), int(colours[label % 32, 2])),
                              2)
                cv2.imshow("小猫追气球", image)
                cv2.waitKey(50)
            vwriter.write(image)
            count += 1
    vwriter.release()
    print("Saved to ", file_name)

if __name__ == '__main__':
    Main()





