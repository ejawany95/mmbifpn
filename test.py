from mm_bifpn import MM_BiFPN
from utils import *

from data_loader_brats18 import BraTS18DataLoader

from torch.utils.data import DataLoader
from torch.autograd import Variable

import torch
import torch.nn as nn
import numpy as np

import time
import resource

#hyper parameter
data_dir = '/data/MICCAI_BraTS_2018_Data_Training' #path to dataset directory
conf_test = '/data/MICCAI_BraTS_2018_Data_Training/newtest_18.txt' #path to valid.txt file
saved_model_path = '/data/final_epoch.pth' #path to trained model .pth file
batch_size = 32
# as the experiment is done is 5-fold cross validation manner, plese change the conf_test valid.txt file accordingly for each fold
#as well as the trained model .pth file


#multi-GPU
cuda_available = torch.cuda.is_available()
if cuda_available:
    device_ids=[0]
    torch.cuda.set_device(device_ids[0])

def to_var(tensor):
    return Variable(tensor.cuda() if cuda_available else tensor)

def evaluation(net, test_dataset, criterion, save_dir=saved_model_path):
    test_loss = []
    iou_5_class_all = []
    fp_5_class_all = []
    dice_whole_tumor = []
    dice_core_tumor = []
    dice_enhancing_tumor = []
    sen_whole_tumor = []
    sen_core_tumor = []
    sen_enhancing_tumor = []
    ppv_whole_tumor = []
    ppv_core_tumor = []
    ppv_enhancing_tumor = []
    jac_whole_tumor = []
    jac_core_tumor = []
    jac_enhancing_tumor = []
    fp_whole_tumor = []
    fp_core_tumor = []
    fp_enhance_tumor = []

    with torch.no_grad():
        time_start = time.perf_counter()
        net.eval()

        for step, (images_vol, label_vol, subject) in enumerate(test_dataset):
            # images_vol    5D tensor       (bz, 155, 4, 240, 240)
            #label_vol      4D tensor       (bz, 155, 240, 240)
            subj_target = label_vol.long().squeeze() # 3D tensor  155 * 240 * 240
            subj_predict = torch.zeros(label_vol.squeeze().shape) # 3D tensor 155 * 240 * 240
            # confusion_matrix(subj_target, subj_predict)

            for t in range(155):
                image = to_var(images_vol[:, t, ...]) # 4D bz(1) * 4 * 240 * 240
                label = to_var(label_vol[:, t, ...]) # 4D tensor    bz(1) * 240 * 240
                predicts = net(image) # 4D tensor   bz(1) * 5 * 240 * 250

                loss_valid = criterion(predicts, label.long())
                test_loss.append(float(loss_valid))

                #softmax and reverse
                predicts = one_hot_reverse(predicts) #3D long T bz(1) * 240 * 240 (0-4)
                subj_predict[t, ...] = predicts.squeeze().long().data

                # if t == 75:
                #     # save_test_images(image, predicts, label, subject, t, save_dir=save_dir + 'images/')
                #     save_predict_image(predicts, subject, t, save_dir=save_dir + 'images/')
                #     save_label_image(label, subject, t, save_dir=save_dir + 'images/')
                # colored_segimg(image, predicts, subject, t, save_dir='new_unet18_pth/images_picplus/')

            #calculate iou
            subj_5class_iou = cal_subject_iou_5class(subj_predict, subj_target)
            subj_5_class_fp = cal_subject_fp_5class(subj_predict, subj_target)
            subj_whole_tumor_dice = cal_subject_dice_whole_tumor(subj_predict, subj_target)
            subj_core_tumor_dice = cal_subject_dice_tumor_core(subj_predict, subj_target)
            subj_enhancing_tumor = cal_subject_dice_enhancing_core(subj_predict, subj_target)

            iou_5_class_all.append(subj_5class_iou)
            fp_5_class_all.append(subj_5_class_fp)
            dice_whole_tumor.append(subj_whole_tumor_dice)
            dice_core_tumor.append(subj_core_tumor_dice)
            dice_enhancing_tumor.append(subj_enhancing_tumor)

            #calculate jaccard
            subj_jac_whole_tumor = cal_subject_jaccard_whole_tumor(subj_target, subj_predict)
            jac_whole_tumor.append(subj_jac_whole_tumor)

            subj_jac_core_tumor = cal_subject_jaccard_tumor_core(subj_target,subj_predict)
            jac_core_tumor.append(subj_jac_core_tumor)

            subj_jac_enhancing_tumor = cal_subject_jaccard_enhancing_core(subj_target, subj_predict)
            jac_enhancing_tumor.append(subj_jac_enhancing_tumor)


            #calculate sensitivity
            subj_sens_whole_tumor = cal_sen_whole_tumor(subj_target,subj_predict)
            sen_whole_tumor.append(subj_sens_whole_tumor)

            subj_sen_core_tumor = cal_sen_core_tumor(subj_target, subj_predict)
            sen_core_tumor.append(subj_sen_core_tumor)

            subj_sen_enhancing_tumor = cal_sen_enhancing_tumor(subj_target, subj_predict)
            sen_enhancing_tumor.append(subj_sen_enhancing_tumor)


            #calculate PPV
            subj_ppv_whole_tumor = cal_ppv_whole_tumor(subj_target, subj_predict)
            ppv_whole_tumor.append(subj_ppv_whole_tumor)

            subj_ppv_core_tumor = cal_ppv_core_tumor(subj_target, subj_predict)
            ppv_core_tumor.append(subj_ppv_core_tumor)

            subj_ppv_enhancing_tumor = cal_ppv_enhancing_tumor(subj_target, subj_predict)
            ppv_enhancing_tumor.append(subj_ppv_enhancing_tumor)

            #calculate false positive
            subj_fp_whole_tumor = cal_fp_whole_tumor(subj_target, subj_predict)
            fp_whole_tumor.append(subj_fp_whole_tumor)

            subj_fp_core_tumor = cal_fp_core_tumor(subj_target, subj_predict)
            fp_core_tumor.append(subj_fp_core_tumor)

            subj_fp_enhancing_tumor = cal_fp_enhance_tumor(subj_target, subj_predict)
            fp_enhance_tumor.append(subj_fp_enhancing_tumor)
            # #save Image
            # if save_dir is not None:
            #     hl, name = subject[0].split('/')[-2]
            #     image_save_dir = save_dir + hl + '/' + name + '.nii.gz'
            #     save_array_as_mha(subj_predict, image_save_dir)
            #     save_train_images(image, predicts, label, index, epoch, save_dir=save_dir)

        print('Dice for whole tumor is')
        average_iou_whole_tumor = sum(dice_whole_tumor) / (len(dice_whole_tumor) * 1.0)
        print(average_iou_whole_tumor)

        print('Dice for core tumor is')
        average_iou_core_tumor = sum(dice_core_tumor) / (len(dice_core_tumor) * 1.0)
        print(average_iou_core_tumor)

        print('Dice enhancing tumor is')
        average_iou_enhancing_tumor = sum(dice_enhancing_tumor) / (len(dice_enhancing_tumor) * 1.0)
        print(average_iou_enhancing_tumor)

        print('Sensitivity Whole Tumor = ')
        average_sen_whole_tumor = sum(sen_whole_tumor)/ (len(sen_whole_tumor) * 1.0)
        print(average_sen_whole_tumor)

        print('Sensitivity Core Tumor = ')
        average_sen_core_tumor = sum(sen_core_tumor) / (len(sen_core_tumor) * 1.0)
        print(average_sen_core_tumor)

        print('Sensitivity Enhancing Tumor = ')
        average_sen_enhancing_tumor = sum(sen_enhancing_tumor) / (len(sen_enhancing_tumor) * 1.0)
        print(average_sen_enhancing_tumor)

        print('PPV whole tumor = ')
        average_ppv_whole_tumor = sum(ppv_whole_tumor) / (len(ppv_whole_tumor) * 1.0)
        print(average_ppv_whole_tumor)

        print('PPV core tumor = ')
        average_ppv_core_tumor = sum(ppv_core_tumor) / (len(ppv_core_tumor) * 1.0)
        print(average_ppv_core_tumor)

        print('PPV for enhancing tumor = ')
        average_ppv_enhancing_tumor = sum(ppv_enhancing_tumor) / (len(ppv_enhancing_tumor) * 1.0)
        print(average_ppv_enhancing_tumor)

        print('Jaccard Whole Tumor = ')
        average_jac_whole_tumor = sum(jac_whole_tumor)/ (len(jac_whole_tumor) * 1.0)
        print(average_jac_whole_tumor)

        print('Jaccard Core Tumor = ')
        average_jac_core_tumor = sum(jac_core_tumor) / (len(jac_core_tumor) * 1.0)
        print(average_jac_core_tumor)

        print('Jaccard Enhancing Tumor = ')
        average_jac_enhancing_tumor = sum(jac_enhancing_tumor) / (len(jac_enhancing_tumor) * 1.0)
        print(average_jac_enhancing_tumor)

        print('FP Whole Tumor = ')
        average_fp_whole_tumor = sum(fp_whole_tumor)/ (len(fp_whole_tumor) * 1.0)
        print(average_fp_whole_tumor)

        print('FP Core Tumor = ')
        average_fp_core_tumor = sum(fp_core_tumor) / (len(fp_core_tumor) * 1.0)
        print(average_fp_core_tumor)

        print('FP Enhance Tumor = ')
        average_fp_enhance_tumor = sum(fp_enhance_tumor) / (len(fp_enhance_tumor) * 1.0)
        print(average_fp_enhance_tumor)

        # iou_5i = []
        for i in range(5):
            iou_i = []
            for iou5 in iou_5_class_all:
                iou_i.append(iou5[i])
            average_iou_label_i = sum(iou_i) / (len(iou_i) * 1.0)
            print('IoU for label ' + str(i) + '  is  ' + str(average_iou_label_i))
        avg_iou_5_class_all = np.mean(iou_5_class_all, axis=0)

        for j in range(5):
            fp_i = []
            for fp5 in fp_5_class_all:
                fp_i.append(fp5[j])
            average_fp_label_i = sum(fp_i) / (len(fp_i) * 1.0)
            print('False positive for label ' + str(j) + ' is ' + str(average_fp_label_i))
        avg_fp_5_class_all = np.mean(fp_5_class_all, axis=0)




        time_elapsed = (time.perf_counter() - time_start)
        memMb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0 / 1024.0
        print("%5.1f secs %5.1f MByte" % (time_elapsed, memMb))
        print('done!')

    return average_iou_whole_tumor, average_iou_core_tumor, average_iou_enhancing_tumor, \
           average_sen_whole_tumor, average_sen_core_tumor, average_sen_enhancing_tumor, \
           average_ppv_whole_tumor, average_ppv_core_tumor, average_ppv_enhancing_tumor, \
           average_jac_whole_tumor, average_jac_core_tumor, average_jac_enhancing_tumor, \
           average_fp_whole_tumor, average_fp_core_tumor, average_fp_enhance_tumor, \
           avg_iou_5_class_all, avg_fp_5_class_all, test_loss




def load_model():
    net = MM_BiFPN(1, 5, 32)
    if cuda_available:
        net = net.cuda()
        net = nn.DataParallel(net, device_ids=device_ids)

    state_dict = torch.load(saved_model_path, map_location='cuda:0')
    net.load_state_dict(state_dict)
    return net

if __name__ == "__main__":

    # if not os.path.exists(save_dir):
    #     os.mkdir(save_dir)
    # if not os.path.exists(save_dir + 'HGG'):
    #     os.mkdir(save_dir + 'HGG')
    # if not os.path.exists(save_dir + 'LGG'):
    #     os.mkdir(save_dir + 'LGG')

    net = load_model()

    print('test_data...')
    test_data = BraTS18DataLoader(data_dir=data_dir, conf=conf_test, train=False)
    test_dataset = DataLoader(dataset=test_data, batch_size=1, shuffle=False)

    weight = torch.from_numpy(test_data.weight).float() #weight for all class
    weight = to_var(weight)
    criterion = nn.CrossEntropyLoss(weight=weight)

    evaluation(net, test_dataset, criterion)
