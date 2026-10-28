import os
import cv2
import numpy as np
from utils.evaluate import compute_sad_loss, compute_mse_loss, compute_gradient_loss, compute_connectivity_error
import argparse
import sys


# 以下为包装好的 Logger 类的定义
class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")  # 防止编码错误

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def evaluate(args):
    img_names = []
    mse_loss_unknown = []
    sad_loss_unknown = []
    grad_loss_unknown = []
    conn_loss_unknown = []

    for i, img in enumerate(os.listdir(args.pred_dir)):
        image_name = img[:-4] + ".png"
        if not ((os.path.isfile(os.path.join(args.pred_dir, img)) and
                 os.path.isfile(os.path.join(args.label_dir, image_name)) and
                 os.path.isfile(os.path.join(args.trimap_dir, image_name)))):
            print('[{}/{}] "{}" skipping'.format(i, len(os.listdir(args.label_dir)), image_name))
            continue

        pred_name = os.path.join(args.pred_dir, img)
        pred = cv2.imread(pred_name, 0).astype(np.float32)
        label = cv2.imread(os.path.join(args.label_dir, image_name), 0).astype(np.float32)
        trimap = cv2.imread(os.path.join(args.trimap_dir, image_name), 0).astype(np.float32)

        # calculate loss
        mse_loss_unknown_ = compute_mse_loss(pred, label, trimap)
        sad_loss_unknown_ = compute_sad_loss(pred, label, trimap)[0]
        grad_loss_unknown_ = compute_gradient_loss(pred, label, trimap)
        conn_loss_unknown_ = compute_connectivity_error(pred, label, trimap, 0.1)
        print('Unknown Region: MSE:', mse_loss_unknown_, ' SAD:', sad_loss_unknown_, ' GRAD:', grad_loss_unknown_,
              ' CONN:', conn_loss_unknown_)

        # save for average
        img_names.append(image_name)

        mse_loss_unknown.append(mse_loss_unknown_)  # mean l2 loss per unknown pixel
        sad_loss_unknown.append(sad_loss_unknown_)  # l1 loss on unknown area
        grad_loss_unknown.append(grad_loss_unknown_)
        conn_loss_unknown.append(conn_loss_unknown_)
        print('[{}/{}] "{}" processed'.format(i, len(os.listdir(args.label_dir)), image_name))

    print('* Unknown Region: MSE:', np.array(mse_loss_unknown).mean(), ' SAD:', np.array(sad_loss_unknown).mean(),
          ' GRAD:', np.array(grad_loss_unknown).mean(), ' CONN:', np.array(conn_loss_unknown).mean())
    print('* if you want to report scores in your paper, please use the official matlab codes for evaluation.')


if __name__ == '__main__':
    a = 6//4
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred-dir', type=str,default='predAlpha/Transparent-460', help="output dir")

    parser.add_argument('--label-dir', type=str, default='./dataset/Transparent-460/alpha_copy', help="GT alpha dir")
    parser.add_argument('--trimap-dir', type=str, default='./dataset/Transparent-460/trimap_copy', help="trimap dir")
    
    # parser.add_argument('--label-dir', type=str, default='./dataset/UHR-395/test/alpha_copy', help="GT alpha dir")
    # parser.add_argument('--trimap-dir', type=str, default='./dataset/UHR-395/test/trimaps', help="trimap dir")

    args = parser.parse_args()
    log = Logger(args.pred_dir + 'A-log' + '.txt')
    sys.stdout = log
    evaluate(args)
