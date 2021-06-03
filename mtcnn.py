import os, sys
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import graph_util
from codecs import open
from PIL import Image, ImageDraw, ImageFont

def bigImgShow(boxes, labels, img):
    for b in boxes:
        b = [int(bb) for bb in b[:4]]
        cv2.rectangle(img, (b[0], b[1]), (b[2], b[3]), (255,80,80), 2)
    pilImg = Image.fromarray(img)
    draw = ImageDraw.Draw(pilImg)
    fSize = 30
    font = ImageFont.truetype("simhei.ttf", fSize, encoding="utf-8")
    for l, b in zip(labels, boxes):
        b = [int(bb) for bb in b[:4]]
        top = (b[0], b[1]-fSize-1)
        name = str(l)
        draw.text(top, name, (255, 50, 50), font=font)
    img = cv2.cvtColor(np.array(pilImg), cv2.COLOR_RGB2BGR)
    return img

def save_pb(sess, names, out_path):
    pb_graph = graph_util.convert_variables_to_constants(
        sess=sess,
        input_graph_def=tf.get_default_graph().as_graph_def(),
        output_node_names=names)
    with tf.gfile.GFile(out_path, "wb") as f:
        f.write(pb_graph.SerializeToString())
        print(names)
        print("%d ops in the final graph." % len(pb_graph.node))

class DetectFace(object):
    def __init__(self):
        self.sess, self.image, self.data, self.fCuts = self.load_pb()

    def getNames(self):
        return ['pnet/input', 'onet/boxes', 'onet/points', 'onet/Ms_inv', 'onet/FinalCuts']
    
    def load_pb(self, path='detect_face.pb'):
        with tf.Graph().as_default():
            config = tf.ConfigProto()  
            config.gpu_options.allow_growth=True  
            sess = tf.Session(config=config)
            pb_graph_def = tf.GraphDef()
            with open(path, "rb") as f:
                pb_graph_def.ParseFromString(f.read())
                tf.import_graph_def(pb_graph_def, name='')
            sess.run(tf.global_variables_initializer())
            image = sess.graph.get_tensor_by_name("pnet/input:0")
            boxes = sess.graph.get_tensor_by_name("onet/boxes:0")
            points = sess.graph.get_tensor_by_name("onet/points:0")
            Ms = sess.graph.get_tensor_by_name("onet/Ms_inv:0")
            fCuts = sess.graph.get_tensor_by_name("onet/FinalCuts:0")
            return sess, image, [boxes, points, Ms], fCuts
    
    def __call__(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cuts, data = self.sess.run([self.fCuts, self.data], {self.image:img})
        return cuts, data

def drawAll(data, results, img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    boxes = data
    labels = results
    bigImg = bigImgShow(boxes, labels, img)
    return bigImg

def cutByBoxes(img, boxes):
    boxes = list(boxes.astype(np.int32))
    cuts = []
    for b in boxes:
        w = b[2]-b[0]
        h = b[3]-b[1]
        new_w = int(h/112*96)
        new_x = int(b[0]+(w-new_w)/2)
        new_x = 0 if new_x<0 else new_x
        cut = img[b[1]:b[3], new_x:new_x+new_w, :]
        orig_cut = img[b[1]:b[3], b[0]:b[2], :]
        cut = cv2.resize(cut, (96, 112), interpolation=cv2.INTER_LINEAR)
        cuts.append(cut)
    return np.array(cuts, dtype=np.float32)

def border_filter(data, res):
    for i, b in enumerate(data[0]):
        w = b[2]-b[0]
        h = b[3]-b[1]
        if w<20 or h<25:
            res[i][0] = '其他'

def test_from_frames():
    detectFace = DetectFace()
    pics = ['frames/'+p for p in os.listdir('frames')]
    crop_size = 150
    save_dir = 'faces/dilireba_adv'
    cnt = 0
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for p in pics:
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        cuts, data = detectFace(img)
        if cuts is None or cuts.shape[0]==0:
            continue
        if cuts.shape[0]>3:
            continue
        for i in range(data[0].shape[0]):
            b = data[0][i][:4]
            w = b[2]-b[0]
            h = b[3]-b[1]
            if w<50 or h<50:
                continue
            c = np.array([(b[0]+b[2])/2, (b[1]+b[3])/2])
            lt = c-crop_size
            rb = c+crop_size
            max_shp = max(img.shape[0], img.shape[1])
            lt = lt.clip(0, max_shp).astype(np.int32)
            rb = rb.clip(0, max_shp).astype(np.int32)
            cut = img[lt[1]:rb[1], lt[0]:rb[0], :]
            sp = save_dir+'/'+str(cnt)+'_'+str(i)+'.jpg'
            cv2.imwrite(sp, cut)
        cnt += 1
        print(cnt, p)

def extract_faces(path = 'videos/dilireba_dou.mp4', dir_name=None, global_cnt=None):
    detectFace = DetectFace()
    cap = cv2.VideoCapture(path)
    start = 0
    intv = 2
    total = 5000
    cnt = 0
    pic_cnt = 0
    is_flip = False
    crop_size = 256
    name = path.split('/')[-1].split('.')[0]
    facePath = 'faces/'+name
    if dir_name is not None:
        facePath = 'faces/'+dir_name
    if not os.path.exists(facePath):
        os.makedirs(facePath)
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            if cnt>=start and cnt%intv==0:
                cv2.imshow('frame', frame)
                cv2.waitKey(1)
                cuts, data = detectFace(frame)
                if cuts is None or cuts.shape[0]==0:
                    cnt += 1
                    continue
                if cuts.shape[0]>3:
                    cnt += 1
                    continue
                iNum = 2 if cuts.shape[0]>=2 else 1
                for i in range(iNum):
                    b = data[0][i][:4]
                    w = b[2]-b[0]
                    h = b[3]-b[1]
                    if w<60 or h<60:
                        continue
                    c = np.array([(b[0]+b[2])/2, (b[1]+b[3])/2])
                    lt = c-crop_size
                    rb = c+crop_size
                    max_shp = max(frame.shape[0], frame.shape[1])
                    lt = lt.clip(0, max_shp).astype(np.int32)
                    rb = rb.clip(0, max_shp).astype(np.int32)
                    cut = frame[lt[1]:rb[1], lt[0]:rb[0], :]
                    fp = facePath+'/'+name+'_'+str(pic_cnt)+'_'+str(i)+'.jpg'
                    if global_cnt is not None:
                        fp = facePath+'/'+str(100000000+global_cnt)+'.jpg'
                        global_cnt += 1
                    cv2.imwrite(fp, cut)
                    if is_flip:
                        cut = cv2.flip(cut, 1)
                        fp_flip = facePath+'/'+name+'_'+str(pic_cnt)+'_'+str(i)+'_flip.jpg'
                        cv2.imwrite(fp_flip, cut)
                pic_cnt += 1
                if pic_cnt>total:
                    break
                # cv2.imshow('cut', cut)
                # cv2.waitKey(1)
            print(cnt, pic_cnt)
            cnt += 1
        else:
            break
    cap.release()
    return global_cnt

if __name__ == "__main__":
    extract_faces(path='videos/libingbing.mp4')