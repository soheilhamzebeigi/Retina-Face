from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import os
import numpy as np
import tensorflow as tf
import time

from modules.models import RetinaFaceModel
from modules.utils import (set_memory_growth, load_yaml,
                           pad_input_image, recover_pad_output)
from resnet56_predict import Predict


flags.DEFINE_string('cfg_path', './configs/retinaface_res50.yaml',
                    'config file path')
flags.DEFINE_string('gpu', '0', 'which gpu to use')
flags.DEFINE_string('img_path', '', 'path to input image')
flags.DEFINE_boolean('webcam', False, 'get image source from webcam or not')
flags.DEFINE_float('iou_th', 0.4, 'iou threshold for nms')
flags.DEFINE_float('score_th', 0.5, 'score threshold for nms')
flags.DEFINE_float('down_scale_factor', 1.0, 'down-scale factor for inputs')



def put_text(img, result, img_h):
    text = ""
    if result == 0:
        text = "Angry"
    elif result == 1:
        text = "Disgust"
    elif result == 2:
        text = "Fear"
    elif result == 3:
        text = "Happy"
    elif result == 4:
        text = "Sad"
    elif result == 5:
        text = "Surprise"
    elif result == 6:
        text = "Neutral"
    
    cv2.putText(img, text, (50, img_h - 50),
                cv2.FONT_HERSHEY_DUPLEX, 2, (255, 255, 255))

def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    resized = cv2.resize(image, dim, interpolation = inter)
    return resized

def draw_bbox_landm(img, ann, img_height, img_width):
    """draw bboxes and landmarks"""
    # bbox
    x1, y1, x2, y2 = int(ann[0] * img_width), int(ann[1] * img_height), \
                     int(ann[2] * img_width), int(ann[3] * img_height)
    return x1, x2, y1, y2

def main(_argv):
    # init
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu

    logger = tf.get_logger()
    logger.disabled = True
    logger.setLevel(logging.FATAL)
    set_memory_growth()

    cfg = load_yaml(FLAGS.cfg_path)

    # define network
    model = RetinaFaceModel(cfg, training=False, iou_th=FLAGS.iou_th,
                            score_th=FLAGS.score_th)
    predict = Predict('data/FER_ResNet_56_model.100.h5')

    # load checkpoint
    checkpoint_dir = './checkpoints/' + cfg['sub_name']
    checkpoint = tf.train.Checkpoint(model=model)
    if tf.train.latest_checkpoint(checkpoint_dir):
        checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
        print("[*] load ckpt from {}.".format(
            tf.train.latest_checkpoint(checkpoint_dir)))
    else:
        print("[*] Cannot find ckpt from {}.".format(checkpoint_dir))
        exit()

    if not FLAGS.webcam:
        if not os.path.exists(FLAGS.img_path):
            print(f"cannot find image path from {FLAGS.img_path}")
            exit()

        print("[*] Processing on single image {}".format(FLAGS.img_path))

        img_raw = cv2.imread(FLAGS.img_path)
        img_height_raw, img_width_raw, _ = img_raw.shape
        img = np.float32(img_raw.copy())

        if FLAGS.down_scale_factor < 1.0:
            img = cv2.resize(img, (0, 0), fx=FLAGS.down_scale_factor,
                             fy=FLAGS.down_scale_factor,
                             interpolation=cv2.INTER_LINEAR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # pad input image to avoid unmatched shape problem
        img, pad_params = pad_input_image(img, max_steps=max(cfg['steps']))

        # run model
        outputs = model(img[np.newaxis, ...]).numpy()

        # recover padding effect
        outputs = recover_pad_output(outputs, pad_params)

        # draw and save results
        save_img_path = os.path.join('out_' + os.path.basename(FLAGS.img_path))
        w=0
        for prior_index in range(len(outputs)):
            w += 1
            x1, x2, y1, y2 = draw_bbox_landm(img_raw, outputs[prior_index], img_height_raw, img_width_raw)
            crop_img = img_raw[y1:y2, x1:x2]
            if (crop_img.size != 0):
                crop_img = image_resize(crop_img, 48, 48)
                crop_img = cv2.resize(crop_img, (48, 48), interpolation=cv2.INTER_LINEAR)
                if(len(crop_img.shape)==3):
                  crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
                cv2.imwrite('{}.jpg'.format(w), crop_img)
                crop_img = crop_img.astype('float32') / 255.0
                crop_img = np.array(crop_img.reshape([1, crop_img.shape[0], crop_img.shape[1], 1]))
                result = predict.predict_emotion(crop_img)
                print(result)
                print("result for face {}: ".format(w), np.argmax(result, axis = 1))


    else:
        cam = cv2.VideoCapture(0)

        start_time = time.time()
        while True:
            _, frame = cam.read()
            if frame is None:
                print("no cam input")

            frame_height, frame_width, _ = frame.shape
            img = np.float32(frame.copy())
            if FLAGS.down_scale_factor < 1.0:
                img = cv2.resize(img, (0, 0), fx=FLAGS.down_scale_factor,
                                 fy=FLAGS.down_scale_factor,
                                 interpolation=cv2.INTER_LINEAR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # pad input image to avoid unmatched shape problem
            img, pad_params = pad_input_image(img, max_steps=max(cfg['steps']))

            # run model
            outputs = model(img[np.newaxis, ...]).numpy()

            # recover padding effect
            outputs = recover_pad_output(outputs, pad_params)

            # draw results
            for prior_index in range(len(outputs)):
                x1, x2, y1, y2 = draw_bbox_landm(frame, outputs[prior_index], frame_height, frame_width)
                crop_img = frame[y1:y2, x1:x2]
            if (crop_img.size != 0):
                crop_img = image_resize(crop_img, 48, 48)
                crop_img = cv2.resize(crop_img, (48, 48), interpolation=cv2.INTER_LINEAR)
                if(len(crop_img.shape)==3):
                  crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
                crop_img = crop_img.astype('float32') / 255.0
                crop_img = np.array(crop_img.reshape([1, crop_img.shape[0], crop_img.shape[1], 1]))
                result = predict.predict_emotion(crop_img)
                # print(result)
                # print("result for face {}: ".format(w), np.argmax(result, axis = 1))
                res = np.argmax(result, axis = 1)
                put_text(frame, res, frame_height)
                


            # calculate fps
            fps_str = "FPS: %.2f" % (1 / (time.time() - start_time))
            start_time = time.time()
            cv2.putText(frame, fps_str, (50, 50 ),
                        cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 2)

            # show frame
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) == ord('q'):
                exit()


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
