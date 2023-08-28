import argparse
import os
import shutil

import h5py
import nibabel as nib
import numpy as np
import SimpleITK as sitk
import torch
from medpy import metric
from scipy.ndimage import zoom
from scipy.ndimage.interpolation import zoom
from tqdm import tqdm
from networks.net_factory import net_factory

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='/data/dataset/LA2018_Seg/Test', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='LA_Seg/E-Net', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='E-Net', help='model_name')
parser.add_argument('--num_classes', type=int,  default=1,
                    help='output channel of network')
parser.add_argument('--labeled_num', type=int, default=8,
                    help='labeled data')
args = parser.parse_args()

def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    dice = metric.binary.dc(pred, gt)
    asd = metric.binary.asd(pred, gt)
    hd95 = metric.binary.hd95(pred, gt)
    IoU = metric.binary.jc(pred, gt)
    return dice, hd95, asd, IoU


def test_single_volume(case, net, test_save_path, FLAGS):
    h5f = h5py.File(FLAGS.root_path + "/{}.h5".format(case), 'r')
    image = h5f['image'][:]
    label = h5f['label'][:]
    label[label == 255] = 1
    image = image.transpose(2, 0, 1)
    label = label.transpose(2, 0, 1)

    prediction = np.zeros_like(label)
    for ind in range(image.shape[0]):
        slice = image[ind, :, :]
        x, y = slice.shape[0], slice.shape[1]
        slice = zoom(slice, (224 / x, 224 / y), order=0)
        input = torch.from_numpy(slice).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            if FLAGS.model == "unet_urds":
                out_main, _, _, _ = net(input)
            else:
                out_main = net(input)[0]
            out = torch.argmax(torch.softmax(
                out_main, dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
            pred = zoom(out, (x / 224, y / 224), order=0)
            prediction[ind] = pred

    first_metric = calculate_metric_percase(prediction == 1, label == 1)


    img_itk = sitk.GetImageFromArray(image.astype(np.float32))
    # img_itk.SetSpacing((1, 1, 10))
    prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
    # prd_itk.SetSpacing((1, 1, 10))
    lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
    # lab_itk.SetSpacing((1, 1, 10))
    sitk.WriteImage(prd_itk, test_save_path + case + "_pred.nii.gz")
    sitk.WriteImage(img_itk, test_save_path + case + "_img.nii.gz")
    sitk.WriteImage(lab_itk, test_save_path + case + "_gt.nii.gz")
    return first_metric


def Inference(FLAGS):
    with open(FLAGS.root_path + '/val.list', 'r') as f:
        image_list = f.readlines()
    image_list = sorted([item.replace('\n', '').split(".")[0]
                         for item in image_list])
     snapshot_path = "../LA2018/{}_{}".format(
         FLAGS.exp, FLAGS.labeled_num)
    
     test_save_path = "../LA2018/{}_{}/predictions/".format(
         FLAGS.exp, FLAGS.labeled_num)
    
    if os.path.exists(test_save_path):
        shutil.rmtree(test_save_path)
    os.makedirs(test_save_path)
    net = net_factory(net_type=FLAGS.model, in_chns=1,
                      class_num=FLAGS.num_classes)
   
    save_mode_path = os.path.join(

        snapshot_path, 'model_LA.pth')
    net.load_state_dict(torch.load(save_mode_path), strict=False)
    print("init weight from {}".format(save_mode_path))
    net.eval()

    first_total = 0.0

    for case in tqdm(image_list):
        first_metric = test_single_volume(
            case, net, test_save_path, FLAGS)
        first_total += np.asarray(first_metric)
    avg_metric = [first_total / len(image_list)]
    return avg_metric


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    FLAGS = parser.parse_args()
    metric = Inference(FLAGS)
    print(metric)
