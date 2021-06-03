import os, sys
from PIL.Image import new
import cv2
import numpy as np
import random
sys.path.append('./CosFace_pytorch')
from mtcnn import DetectFace, drawAll
from cosface_pred import get_img_feature, get_distance
import math

detectFace = DetectFace()

def show(img, name='', wait=0):
    cv2.imshow(name, img)
    cv2.waitKey(wait)

def saveVideo(path, frames, frate):
    videoWriter = cv2.VideoWriter(path, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), frate, (frames[0].shape[1],frames[0].shape[0]))
    for f in frames:
        videoWriter.write(f)
    videoWriter.release()

def get_frame_id_map(feat_dict):
    fid_map = {}
    for k, v in feat_dict.items():
        for c, b, fid in v:
            if fid not in fid_map.keys():
                fid_map[fid] = [(c, b, k)]
            else:
                fid_map[fid].append((c, b, k))
    return fid_map

def get_box_iou(b0, b1):
    def compute_iou(rec1, rec2):
        """
        computing IoU
        :param rec1: (y0, x0, y1, x1), which reflects
                (top, left, bottom, right)
        :param rec2: (y0, x0, y1, x1)
        :return: scala value of IoU
        """
        # computing area of each rectangles
        S_rec1 = (rec1[2] - rec1[0]) * (rec1[3] - rec1[1])
        S_rec2 = (rec2[2] - rec2[0]) * (rec2[3] - rec2[1])
    
        # computing the sum_area
        sum_area = S_rec1 + S_rec2
    
        # find the each edge of intersect rectangle
        left_line = max(rec1[1], rec2[1])
        right_line = min(rec1[3], rec2[3])
        top_line = max(rec1[0], rec2[0])
        bottom_line = min(rec1[2], rec2[2])
    
        # judge if there is an intersect
        if left_line >= right_line or top_line >= bottom_line:
            return 0
        else:
            intersect = (right_line - left_line) * (bottom_line - top_line)
            return (intersect / (sum_area - intersect))*1.0
    x00, y00, x01, y01, _ = b0
    x10, y10, x11, y11, _ = b1
    return compute_iou((y00,x00,y01,x01), (y10,x10,y11,x11))

def merge_features_one_round(feat_dict):
    merge_flag = False
    for k, v in feat_dict.items():
        for ik, iv in feat_dict.items():
            if k==ik:
                break
            feats0 = iv.copy()
            feats1 = v.copy()
            random.shuffle(feats0)
            random.shuffle(feats1)
            feats0 = feats0[:20]
            feats1 = feats1[:10]
            pass_cnt = min(8, len(feats0), len(feats1))
            is_feat_cnt = 0
            for f0, _, _ in feats0:
                for f1, _, _ in feats1:
                    dist = get_distance(f0, f1)
                    if dist>0.5:
                        is_feat_cnt += 1
                    if is_feat_cnt==pass_cnt:
                        break
                if is_feat_cnt==pass_cnt:
                    break
            if is_feat_cnt==pass_cnt:
                add_feats = v.copy()
                iv += add_feats
                feat_dict[k] = []
                merge_flag = True
                break
    new_dict = {}
    id_cnt = 0
    for k, v in feat_dict.items():
        if len(v)>0:
            new_dict[id_cnt] = v
            id_cnt += 1
    return merge_flag, new_dict
            
def feature_filter(feat_dict):
    mFlag = True
    while mFlag:
        mFlag, feat_dict = merge_features_one_round(feat_dict)
    mFlag, feat_dict = merge_features_one_round(feat_dict)
    new_dict = {}
    id_cnt = 1
    for k, v in feat_dict.items():
        if len(v)>15:
            new_dict[id_cnt] = v
            id_cnt += 1
    return new_dict

def video_analysis(path):
    cap = cv2.VideoCapture(path)
    frate = cap.get(cv2.CAP_PROP_FPS)
    frame_id = 0
    feat_id = 0
    last_feats = {}
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frames.append((frame_id, frame))
            cuts, data = detectFace(frame)
            cuts = list(cuts)
            boxes = list(data[0])
            if len(last_feats)<=0:
                for c, b in zip(cuts, boxes):
                    score = b[4]
                    if score<0.98:
                        continue
                    fc = get_img_feature(c)
                    last_feats[feat_id] = [(fc, b, frame_id)]
                    feat_id += 1
            else:
                for c, b in zip(cuts, boxes):
                    score = b[4]
                    if score<0.98:
                        continue
                    fc = get_img_feature(c)
                    best_id = -1
                    best_dist = -1
                    best_iou = -1
                    for key in last_feats.keys():
                        fl = last_feats[key][-1][0]
                        bl = last_feats[key][-1][1]
                        iou = get_box_iou(bl, b)
                        dist = get_distance(fl, fc)
                        if dist>best_dist:
                            best_dist = dist
                            best_id = key
                            best_iou = iou
                    if best_dist>=0.7 or (best_dist>0.35 and best_iou>0.7):
                        last_feats[best_id].append((fc, b, frame_id))
                    else:
                        last_feats[feat_id] = [(fc, b, frame_id)]
                        feat_id += 1
            frame_id += 1
            print('process', frame_id)
        else:
            break
    cap.release()
    last_feats = feature_filter(last_feats)
    for k, v in last_feats.items():
        print(k, len(v))
    fid_map = get_frame_id_map(last_feats)
    new_frames = []
    for i, frame in frames:
        if i not in fid_map.keys():
            new_frames.append(frame)
            continue
        feat_list = fid_map[i]
        for c, b, k in feat_list:
            frame = drawAll([b], [k], frame)
        new_frames.append(frame)
    saveVideo('tmp.mp4', new_frames, frate)

video_analysis('videos/sample5.mp4')