# coding=utf-8
import sys
sys.path.append("./")
import cv2
import numpy as np
import mxnet as mx
from pathlib import Path
# from deploy.mtcnn_detector import MtcnnDetector
from deploy.C3AE_expand_pred import build_net3, model_refresh_without_nan
OP_FILE_PATH = Path(__file__).absolute()
import math

def gen_boundbox(box, landmark):
    # gen trible boundbox
    ymin, xmin, ymax, xmax = map(int, [box[1], box[0], box[3], box[2]])
    w, h = xmax - xmin, ymax - ymin
    nose_x, nose_y = (landmark[2], landmark[2+5])
    w_h_margin = abs(w - h)
    top2nose = nose_y - ymin
    # 包含五官最小的框
    return np.array([
        [(xmin - w_h_margin, ymin - w_h_margin), (xmax + w_h_margin, ymax + w_h_margin)],  # out
        [(nose_x - top2nose, nose_y - top2nose), (nose_x + top2nose, nose_y + top2nose)],  # middle
        [(nose_x - w//2, nose_y - w//2), (nose_x + w//2, nose_y + w//2)]  # inner box
    ])


def gen_face_legacy(detector, image, image_path="", only_one=True):
    ret = detector.detect_face(image)
    if not ret:
        raise Exception("cant detect facei: %s"%image_path)
    bounds, lmarks = ret
    if only_one and len(bounds) > 1:
        print("!!!!!,", bounds, lmarks)
        raise Exception("more than one face %s"%image_path)
    return ret

def gen_face(detector, image, image_path="", only_one=True):
    cuts, data = detector(image)
    if cuts.shape[0]<=0:
        raise Exception("cant detect facei: %s"%image_path)
    bounds, lmarks = data[0], data[1]
    lmarks = lmarks.transpose(0, 2, 1).reshape([-1, 10])
    if only_one and len(bounds) > 1:
        print("!!!!!,", bounds, lmarks)
        raise Exception("more than one face %s"%image_path)
    ret = (bounds.astype(np.int32), lmarks.astype(np.int32))
    return ret

class AgeGenderPredictor(object):
    def __init__(self, detectFaceModel):
        model_path = str(OP_FILE_PATH.parent / 'gen_age_model/c3ae_model_v2_fp16_white_se_132_4.208622-0.973')
        self.models = build_net3(12, using_SE=True, using_white_norm=True)
        self.models.load_weights(model_path)
        model_refresh_without_nan(self.models)
        self.mtcnn_detect = detectFaceModel

    def list2colmatrix(self, pts_list):
        """
            convert list to column matrix
        Parameters:
        ----------
            pts_list:
                input list
        Retures:
        -------
            colMat:

        """
        assert len(pts_list) > 0
        colMat = []
        for i in range(len(pts_list)):
            colMat.append(pts_list[i][0])
            colMat.append(pts_list[i][1])
        colMat = np.matrix(colMat).transpose()
        return colMat

    def find_tfrom_between_shapes(self, from_shape, to_shape):
        """
            find transform between shapes
        Parameters:
        ----------
            from_shape:
            to_shape:
        Retures:
        -------
            tran_m:
            tran_b:
        """
        assert from_shape.shape[0] == to_shape.shape[0] and from_shape.shape[0] % 2 == 0

        sigma_from = 0.0
        sigma_to = 0.0
        cov = np.matrix([[0.0, 0.0], [0.0, 0.0]])

        # compute the mean and cov
        from_shape_points = from_shape.reshape(int(from_shape.shape[0]/2), 2)
        to_shape_points = to_shape.reshape(int(to_shape.shape[0]/2), 2)
        mean_from = from_shape_points.mean(axis=0)
        mean_to = to_shape_points.mean(axis=0)

        for i in range(from_shape_points.shape[0]):
            temp_dis = np.linalg.norm(from_shape_points[i] - mean_from)
            sigma_from += temp_dis * temp_dis
            temp_dis = np.linalg.norm(to_shape_points[i] - mean_to)
            sigma_to += temp_dis * temp_dis
            cov += (to_shape_points[i].transpose() - mean_to.transpose()) * (from_shape_points[i] - mean_from)

        sigma_from = sigma_from / to_shape_points.shape[0]
        sigma_to = sigma_to / to_shape_points.shape[0]
        cov = cov / to_shape_points.shape[0]

        # compute the affine matrix
        s = np.matrix([[1.0, 0.0], [0.0, 1.0]])
        u, d, vt = np.linalg.svd(cov)

        if np.linalg.det(cov) < 0:
            if d[1] < d[0]:
                s[1, 1] = -1
            else:
                s[0, 0] = -1
        r = u * s * vt
        c = 1.0
        if sigma_from != 0:
            c = 1.0 / sigma_from * np.trace(np.diag(d) * s)

        tran_b = mean_to.transpose() - c * r * mean_from.transpose()
        tran_m = c * r

        return tran_m, tran_b

    def extract_image_chips(self, img, points, desired_size=256, padding=0):
        """
            crop and align face
        Parameters:
        ----------
            img: numpy array, bgr order of shape (1, 3, n, m)
                input image
            points: numpy array, n x 10 (x1, x2 ... x5, y1, y2 ..y5)
            desired_size: default 256
            padding: default 0
        Retures:
        -------
            crop_imgs: list, n
                cropped and aligned faces
        """
        crop_imgs = []
        for p in points:
            shape  =[]
            for k in range(int(len(p)/2)):
                shape.append(p[k])
                shape.append(p[k+5])

            if padding > 0:
                padding = padding
            else:
                padding = 0
            # average positions of face points
            mean_face_shape_x = [0.224152, 0.75610125, 0.490127, 0.254149, 0.726104]
            mean_face_shape_y = [0.2119465, 0.2119465, 0.628106, 0.780233, 0.780233]

            from_points = []
            to_points = []

            for i in range(int(len(shape)/2)):
                x = (padding + mean_face_shape_x[i]) / (2 * padding + 1) * desired_size
                y = (padding + mean_face_shape_y[i]) / (2 * padding + 1) * desired_size
                to_points.append([x, y])
                from_points.append([shape[2*i], shape[2*i+1]])

            # convert the points to Mat
            from_mat = self.list2colmatrix(from_points)
            to_mat = self.list2colmatrix(to_points)

            # compute the similar transfrom
            tran_m, tran_b = self.find_tfrom_between_shapes(from_mat, to_mat)

            probe_vec = np.matrix([1.0, 0.0]).transpose()
            probe_vec = tran_m * probe_vec

            scale = np.linalg.norm(probe_vec)
            angle = 180.0 / math.pi * math.atan2(probe_vec[1, 0], probe_vec[0, 0])

            from_center = [(shape[0]+shape[2])/2.0, (shape[1]+shape[3])/2.0]
            to_center = [0, 0]
            to_center[1] = desired_size * 0.4
            to_center[0] = desired_size * 0.5

            ex = to_center[0] - from_center[0]
            ey = to_center[1] - from_center[1]

            rot_mat = cv2.getRotationMatrix2D((from_center[0], from_center[1]), -1 * angle, scale)
            rot_mat[0][2] += ex
            rot_mat[1][2] += ey

            chips = cv2.warpAffine(img, rot_mat, (desired_size, desired_size))
            crop_imgs.append(chips)

        return crop_imgs

    def find_mid_face(self, img, bbox, ldmks):
        """
        如果图片中包含多个人脸，找到位于图片最中间的一个作为需要的人脸
        :param img:
        :param bbox:
        :param ldmks:
        :return:
        """
        if len(ldmks) == 1:
            return bbox, ldmks
        img_h, img_w, _ = img.shape
        img_center = (img_w / 2, img_h / 2)
        nose_distance_center = np.inf
        mid_idx = 0
        for idx, ldmk in enumerate(ldmks):
            # 找到鼻子里图片中心距离最小的那个人
            nose_x, nose_y = ldmk[2], ldmk[2+5]
            nose_distance_center_tmp = np.sqrt((nose_x - img_center[0]) ** 2 + (nose_y - img_center[1]) ** 2)
            if nose_distance_center_tmp < nose_distance_center:
                nose_distance_center = nose_distance_center_tmp
                mid_idx = idx
        return [bbox[mid_idx]], [ldmks[mid_idx]]

    def predict_age_gen(self, img, bbox=None, ldmks=None):
        """
        识别图片中人的年龄和性别
        :param img: opencv 读取出来的图片的numpy数组
        :return: {"gen": float, "age": str}
        """
        result_dict = {"gen": None, "age": None}
        try:
            if bbox is None or ldmks is None:
                bbox, ldmks = gen_face(self.mtcnn_detect, img, only_one=False)
            else:
                ldmks = ldmks.transpose(0, 2, 1).reshape([-1, 10]).astype(np.int32)
                bbox = bbox.astype(np.int32)
            # 只留下最中间的人脸
            bbox, ldmks = self.find_mid_face(img, bbox, ldmks)
            ret = self.extract_image_chips(img, ldmks, padding=0.4)
        except Exception as e:
            print(e)
            ret = None
        if not ret:
            return result_dict
        padding = 200
        new_bd_img = cv2.copyMakeBorder(img, padding, padding, padding, padding, cv2.BORDER_CONSTANT)
        for pidx, (box, ldmk) in enumerate(zip(bbox, ldmks)):
            trible_box = gen_boundbox(box, ldmk)
            tri_imgs = []
            for bb in trible_box:
                bb = bb + padding
                h_min, w_min = bb[0]
                h_max, w_max = bb[1]
                tri_imgs.append([cv2.resize(new_bd_img[w_min: w_max, h_min: h_max, :], (64, 64))])

            result = self.models.predict(tri_imgs)
            if result and len(result) == 3:
                age, _, gender = result
                age_label, gender_label = age[-1][-1], 'F' if gender[-1][0] > gender[-1][1] else 'M'
            elif result and len(result) == 2:
                # 不需要判断性别的情况
                age, _ = result
                age_label, gender_label = age[-1][-1], 'unknown'
            else:
                raise Exception(f'fatal result: {result}')
            result_dict['gen'] = gender_label
            result_dict['age'] = age_label
        return result_dict


if __name__ == '__main__':
    from mtcnn import DetectFace
    pred = AgeGenderPredictor(DetectFace())
    img_np = cv2.imread('rzr.png')
    out = pred.predict_age_gen(img_np)
    print(out)
