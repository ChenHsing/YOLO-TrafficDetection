# -*- coding: utf-8 -*-
"""
Class definition of YOLO_v3 style detection model on image and video
"""

import colorsys
import os
from timeit import default_timer as timer
import cv2
import numpy as np
from keras import backend as K
from keras.models import load_model
from keras.layers import Input
from PIL import Image, ImageFont, ImageDraw

from yolo3.model import yolo_eval, yolo_body, tiny_yolo_body
from yolo3.utils import letterbox_image
import os
from keras.utils import multi_gpu_model
import collections


class YOLO(object):
    _defaults = {
        "model_path": 'model_data/yolo.h5',
        "anchors_path": 'model_data/yolo_anchors.txt',
        "classes_path": 'model_data/coco_classes.txt',
        "score" : 0.3,
        "iou" : 0.35,
        "model_image_size" : (416, 416),
        "gpu_num" : 1,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults) # set up default values
        self.__dict__.update(kwargs) # and update with user overrides
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.sess = K.get_session()
        self.boxes, self.scores, self.classes = self.generate()

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        is_tiny_version = num_anchors==6 # default setting
        try:
            self.yolo_model = load_model(model_path, compile=False)
        except:
            self.yolo_model = tiny_yolo_body(Input(shape=(None,None,3)), num_anchors//2, num_classes) \
                if is_tiny_version else yolo_body(Input(shape=(None,None,3)), num_anchors//3, num_classes)
            self.yolo_model.load_weights(self.model_path) # make sure model, anchors and classes match
        else:
            assert self.yolo_model.layers[-1].output_shape[-1] == \
                num_anchors/len(self.yolo_model.output) * (num_classes + 5), \
                'Mismatch between model and given anchor and class sizes'

        print('{} model, anchors, and classes loaded.'.format(model_path))

        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))
        np.random.seed(10101)  # Fixed seed for consistent colors across runs.
        np.random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.
        np.random.seed(None)  # Reset seed to default.

        # Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = K.placeholder(shape=(2, ))
        if self.gpu_num>=2:
            self.yolo_model = multi_gpu_model(self.yolo_model, gpus=self.gpu_num)
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                len(self.class_names), self.input_image_shape,
                score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes

    def getColorList(self):
        dict = collections.defaultdict(list)

        # 红色
        lower_red = np.array([156, 43, 46])
        upper_red = np.array([180, 255, 255])
        color_list = []
        color_list.append(lower_red)
        color_list.append(upper_red)
        dict['red'] = color_list

        # 红色2
        lower_red = np.array([0, 43, 46])
        upper_red = np.array([10, 255, 255])
        color_list = []
        color_list.append(lower_red)
        color_list.append(upper_red)
        dict['red2'] = color_list

        # 橙色
        lower_orange = np.array([11, 43, 46])
        upper_orange = np.array([25, 255, 255])
        color_list = []
        color_list.append(lower_orange)
        color_list.append(upper_orange)
        dict['orange'] = color_list

        # 黄色
        lower_yellow = np.array([26, 43, 46])
        upper_yellow = np.array([34, 255, 255])
        color_list = []
        color_list.append(lower_yellow)
        color_list.append(upper_yellow)
        dict['yellow'] = color_list

        # 绿色
        lower_green = np.array([35, 43, 46])
        upper_green = np.array([77, 255, 255])
        color_list = []
        color_list.append(lower_green)
        color_list.append(upper_green)
        dict['green'] = color_list
        return dict

    def get_color(self,frame):
        print('go in get_color')
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        maxsum = -100
        color = None
        color_dict = self.getColorList()
        score = 0
        type = 'black'
        for d in color_dict:
            mask = cv2.inRange(hsv, color_dict[d][0], color_dict[d][1])
            # print(cv2.inRange(hsv, color_dict[d][0], color_dict[d][1]))
            #cv2.imwrite('images/triffic/' + f + d + '.jpg', mask)
            binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)[1]
            binary = cv2.dilate(binary, None, iterations=2)
            img, cnts, hiera = cv2.findContours(binary.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            sum = 0
            for c in cnts:
                sum += cv2.contourArea(c)

            if sum > maxsum:
                maxsum = sum
                color = d
            if sum > score:
                score = sum
                type = d
        return type



    def detect_image(self, image,path):
        print('class',self._get_class())

        start = timer()

        if self.model_image_size != (None, None):
            assert self.model_image_size[0]%32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1]%32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')

        print(image_data.shape)
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.


        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })

        print('Found {} boxes for {}'.format(len(out_boxes), 'img'))

        font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                    size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = (image.size[0] + image.size[1]) // 300
        thickness = 5
        print('thickness',thickness)
        print('out_classes',out_classes)
        my_class = ['traffic light']
        imgcv = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)

        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            print('predicted_class',predicted_class)
            if predicted_class not in my_class:
                continue
            box = out_boxes[i]
            score = out_scores[i]

            label = '{} {:.2f}'.format(predicted_class, score)

            draw = ImageDraw.Draw(image)

            label_size = draw.textsize(label, font)

            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
            print(label, (left, top), (right, bottom))
            img2 = imgcv[top:bottom, left:right]
            color = self.get_color(img2)
            cv2.imwrite('images/triffic/'+path+str(i) + '.jpg', img2)
            if color== 'red' or color == 'red2':
                cv2.rectangle(imgcv, (left, top), (right, bottom), color=(0, 0, 255),
                              lineType=2, thickness=8)
                cv2.putText(imgcv, '{0} {1:.2f}'.format(color, score),
                            (left, top - 15),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.2, (0, 0, 255), 4,
                            cv2.LINE_AA)
            elif color == 'green':
                cv2.rectangle(imgcv, (left, top), (right, bottom), color=(0, 255, 0),
                              lineType=2, thickness=8)
                cv2.putText(imgcv, '{0} {1:.2f}'.format(color, score),
                            (left, top - 15),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.2, (0, 255, 0), 4,
                            cv2.LINE_AA)
            else:
                cv2.rectangle(imgcv, (left, top), (right, bottom), color=(255, 0, 0),
                              lineType=2, thickness=8)
                cv2.putText(imgcv, '{0} {1:.2f}'.format(color, score),
                            (left, top - 15),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.2, (255, 0, 0), 4,
                            cv2.LINE_AA)

            print(imgcv.shape)


            # if top - label_size[1] >= 0:
            #     text_origin = np.array([left, top - label_size[1]])
            # else:
            #     text_origin = np.array([left, top + 1])
            #
            # # My kingdom for a good redistributable image drawing library.
            # for j in range(thickness):
            #     draw.rectangle(
            #         [left + j, top + j, right - j, bottom - j],
            #         outline=self.colors[c])
            # draw.rectangle(
            #     [tuple(text_origin), tuple(text_origin + label_size)],
            #     fill=self.colors[c])
            # draw.text(text_origin, label, fill=(0, 0, 0), font=font)
            # del draw

        end = timer()
        print(end - start)
        return imgcv

    def close_session(self):
        self.sess.close()



def detect_img(yolo, img_path,fname):
    img = Image.open(img_path)
    import time
    t1 = time.time()

    img = yolo.detect_image(img,fname)
    print('time: {}'.format(time.time() - t1))
    return img
    #yolo.close_session()



if __name__ == '__main__':

    yolo = YOLO()
    output = 'images/res3.avi'
    video_full_path = 'images/triffic3.mp4'

    cap = cv2.VideoCapture(video_full_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 1)  # 设置要获取的帧号

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = cap.get(cv2.CAP_PROP_FPS)
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    out = cv2.VideoWriter(output, fourcc, fps, size)
    ret = True
    count = 0
    while ret :
        count+=1
        ret, frame = cap.read()
        if not ret :
            print('结束')
            break
        image = Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
        image = yolo.detect_image(image,'pic')
        out.write(image)

    cap.release()
    out.release()
    cv2.destroyAllWindows()


