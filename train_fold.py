import time
from math import sqrt

import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

import resource
from sklearn.model_selection import KFold

from utils import *
from mm_fpn import MM_FPN
from test import evaluation
from data_loader_brats18 import BraTS18DataLoader

import time
import resource

#hyper parameter
data_dir = 'BraTS-DMFNet/MICCAI_BraTS_2018_Data_Training'#path to training data directory
data_dir_valid = 'BraTS-DMFNet/MICCAI_BraTS_2018_Data_Training'#path to valid data directory
# conf_train = 'data_list/train_18_0.txt' #path to training list 'train.txt' file
# conf_valid = 'data_list/valid_18_0.txt'#path to valid list 'valid.txt' file
# save_pth_dir = 'mmfpn18_pth/'#path to save trained model '.pth' file
save_dir = 'mmfpn18_pth/try/'

learning_rate = 0.0001
batch_size = 32
epochs = 100
k_folds = 5
# #for fold results
# results = {}

cuda_available = torch.cuda.is_available()
device_ids = [4,5,6,7] #number of gpu available
torch.cuda.set_device(device_ids[0])

#build model
net = MM_FPN(1, 5, 32)

if cuda_available:
    net = net.cuda()
    net = nn.DataParallel(net, device_ids=device_ids)

# if not os.path.exists(save_dir):
#     os.mkdir(save_dir)
root = 'data_list/'

def write(data, fname, root=root):
    fname = os.path.join(root, fname)
    with open(fname, 'w') as f:
        f.write('\n'.join(data))



# #data preparation
# print('train data...')
# train_data = BraTS18DataLoader(data_dir=data_dir, conf=conf_train, train=True)
#
# print('Valid data...')
# valid_data = BraTS18DataLoader(data_dir=data_dir, conf=conf_valid, train=False)
#
# #dataloader
# train_dataset = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
# valid_dataset = DataLoader(dataset=valid_data, batch_size=1, shuffle=True)

def to_var(tensor):
    return Variable(tensor.cuda() if cuda_available else tensor)

def reset_weight(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.reset_parameters()

def main():
    hgg = os.listdir(os.path.join(data_dir, 'HGG'))
    hgg = [os.path.join('HGG', f) for f in hgg]

    lgg = os.listdir(os.path.join(data_dir, 'LGG'))
    lgg = [os.path.join('LGG', f) for f in lgg]

    X = hgg + lgg
    Y = [1] * len(hgg) + [0] * len(lgg)

    write(X, 'all.txt')

    X, Y = np.array(X), np.array(Y)

    # Define the K-fold Cross Validator
    kfold = KFold(n_splits=k_folds, shuffle=True)

    # Start print
    print('--------------------------------')

    allfold_dice_wt = []
    allfold_dice_ct = []
    allfold_dice_et = []
    allfold_sen_wt = []
    allfold_sen_ct = []
    allfold_sen_et = []
    allfold_ppv_wt = []
    allfold_ppv_ct = []
    allfold_ppv_et = []
    allfold_jac_wt = []
    allfold_jac_ct = []
    allfold_jac_et = []
    allfold_iou_label = []

    for k, (train_index, valid_index) in enumerate(kfold.split(Y, Y)):
    # for k in enumerate(kfold):
        dice_whole = []
        dice_core = []
        dice_enhance = []
        sen_whole_tumor = []
        sen_core_tumor = []
        sen_enhance_tumor = []
        ppv_whole_tumor = []
        ppv_core_tumor = []
        ppv_enhance_tumor = []
        jac_whole_tumor = []
        jac_core_tumor = []
        jac_enhance_tumor = []
        iou_label_class = []

        print(f'FOLD {k}')
        print('--------------------------------')

        train_list = list(X[train_index])
        valid_list = list(X[valid_index])

        write(train_list, 'train_18_{}.txt'.format(k))
        write(valid_list, 'valid_18_{}.txt'.format(k))

        if not os.path.exists(os.path.join(save_dir, 'fold{}'.format(k))):
            os.mkdir(os.path.join(save_dir, 'fold{}'.format(k)))

        #define data loaders for training and valid data in this fold

        # data preparation
        print('train data...')
        train_data = BraTS18DataLoader(data_dir=data_dir, conf='data_list/train_18_{}.txt'.format(k), train=True)

        print('Valid data...')
        valid_data = BraTS18DataLoader(data_dir=data_dir, conf='data_list/valid_18_{}.txt'.format(k), train=False)

        # dataloader
        train_dataset = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
        valid_dataset = DataLoader(dataset=valid_data, batch_size=1, shuffle=True)

        #build model
        net = MM_FPN(1, 5, 32)
        net.apply(reset_weight)

        if cuda_available:
            net = net.cuda()
            net = nn.DataParallel(net, device_ids=device_ids)

        score_max = -1.0
        best_epoch = 0
        weight = torch.from_numpy(train_data.weight).float()
        weight = to_var(weight)

        optimizer = optim.Adam(params=net.parameters(), lr=learning_rate, betas=(0.9, 0.999))
        criterion = nn.CrossEntropyLoss(weight=weight)
        time_start = time.perf_counter()

        for epoch in range(1, epochs + 1):
            print('epoch: ' + str(epoch))
            train_loss = []
            net.train()
            for step, (image, label, index) in enumerate(train_dataset):
                image = to_var(image)  # 4D tensor    bz * 4(modal) * 240 * 240
                label = to_var(label)  # 3D tensor   bz * 240 * 240 (value 0-4)

                optimizer.zero_grad()
                predicts = net(image)  # 4D tensor    bz * 5(class) * 240 * 240
                loss_train = criterion(predicts, label.long())
                train_loss.append(float(loss_train))
                loss_train.backward()
                optimizer.step()

                # save sample image for each epoch
                if epoch == epochs:
                    print('.. step ... %d' % step)
                    print('.. loss ... %f' % loss_train)
                    predicts = one_hot_reverse(predicts)  # 3Dlong tensor bz * 240 * 240 ( val 0-4)
                    save_train_images(image, predicts, label, index, epoch, save_dir='mmfpn18_pth/try/fold{}'.format(k))

            # calculate valid loss
            print('valid ...')
            iou_whole, iou_core, iou_enhance, sen_whole, sen_core, sen_enhance, ppv_whole, ppv_core, ppv_enhance, jac_whole, jac_core, jac_enhance, iou_label, valid_loss = evaluation(net, valid_dataset, criterion, save_dir=None)


            # save loss for one batch
            print('train_epoch_loss ' + str(sum(train_loss) / (len(train_loss) * 1.0)))
            print('valid_epoch_loss ' + str(sum(valid_loss) / (len(valid_loss) * 1.0)))

            # save model
            if iou_whole > score_max:
                best_epoch = epoch
                torch.save(net.state_dict(),
                           os.path.join(save_dir, f'fold{k}/best_epoch.pth'))
                score_max = iou_whole

            print('valid_meanIoU_max ' + str(score_max))
            print('Current Best Epoch is %d' % best_epoch)

            if epoch == epochs:
                torch.save(net.state_dict(), os.path.join(save_dir, f'fold{k}/final_epoch.pth'))

            dice_whole.append(iou_whole)
            # for item in dice_whole:
            #     print('dice whole tumor list' + item)
            # print('dice whole tumor list' + dice_whole)
            dice_core.append(iou_core)
            dice_enhance.append(iou_enhance)
            sen_whole_tumor.append(sen_whole)
            sen_core_tumor.append(sen_core)
            sen_enhance_tumor.append(sen_enhance)
            ppv_whole_tumor.append(ppv_whole)
            ppv_core_tumor.append(ppv_core)
            ppv_enhance_tumor.append(ppv_enhance)
            jac_whole_tumor.append(jac_whole)
            jac_core_tumor.append(jac_core)
            jac_enhance_tumor.append(jac_enhance)
            iou_label_class.append(iou_label)

        # print('one_epoch_dice_whole ' + (sum(dice_whole) / (len(dice_whole) * 1.0)))
        singlefold_dice_wt = sum(dice_whole) / (len(dice_whole) * 1.0)
        print('fold{} dice whole'.format(k))
        print(singlefold_dice_wt)

        singlefold_dice_ct = sum(dice_core) / (len(dice_core) * 1.0)
        print('fold{} dice core'.format(k))
        print(singlefold_dice_ct)

        singlefold_dice_et = sum(dice_enhance) / (len(dice_enhance) * 1.0)
        print('fold{} dice enhance'.format(k))
        print(singlefold_dice_et)

        singlefold_sen_wt = sum(sen_whole_tumor) / (len(sen_whole_tumor) * 1.0)
        print('fold{} sen whole'.format(k))
        print(singlefold_sen_wt)

        singlefold_sen_ct = sum(sen_core_tumor) / (len(sen_core_tumor) * 1.0)
        print('fold{} sen core'.format(k))
        print(singlefold_sen_ct)

        singlefold_sen_et = sum(sen_enhance_tumor) / (len(sen_enhance_tumor) * 1.0)
        print('fold{} sen enhance'.format(k))
        print(singlefold_sen_et)

        singlefold_ppv_wt = sum(ppv_whole_tumor) / (len(ppv_whole_tumor) * 1.0)
        print('fold{} ppv whole'.format(k))
        print(singlefold_ppv_wt)

        singlefold_ppv_ct = sum(ppv_core_tumor) / (len(ppv_core_tumor) * 1.0)
        print('fold{} ppv core'.format(k))
        print(singlefold_ppv_ct)

        singlefold_ppv_et = sum(ppv_enhance_tumor) / (len(ppv_enhance_tumor) * 1.0)
        print('fold{} ppv enhance'.format(k))
        print(singlefold_ppv_et)

        singlefold_jac_wt = sum(jac_whole_tumor) / (len(jac_whole_tumor) * 1.0)
        print('fold{} jac whole'.format(k))
        print(singlefold_jac_wt)

        singlefold_jac_ct = sum(jac_core_tumor) / (len(jac_core_tumor) * 1.0)
        print('fold{} jac core'.format(k))
        print(singlefold_jac_ct)

        singlefold_jac_et = sum(jac_enhance_tumor) / (len(jac_enhance_tumor) * 1.0)
        print('fold{} jac enhance'.format(k))
        print(singlefold_jac_et)

        singlefold_iou_class = sum(iou_label_class) / (len(iou_label_class) * 1.0)
        print('fold{} iou class'. format(k))
        print(singlefold_iou_class)

        time_elapsed = (time.perf_counter() - time_start)
        memMb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0 / 1024.0

        allfold_dice_wt.append(singlefold_dice_wt)
        allfold_dice_ct.append(singlefold_dice_ct)
        allfold_dice_et.append(singlefold_dice_et)
        allfold_sen_wt.append(singlefold_sen_wt)
        allfold_sen_ct.append(singlefold_sen_ct)
        allfold_sen_et.append(singlefold_sen_et)
        allfold_ppv_wt.append(singlefold_ppv_wt)
        allfold_ppv_ct.append(singlefold_ppv_ct)
        allfold_ppv_et.append(singlefold_ppv_et)
        allfold_jac_wt.append(singlefold_jac_wt)
        allfold_jac_ct.append(singlefold_jac_ct)
        allfold_jac_et.append(singlefold_jac_et)
        allfold_iou_label.append(singlefold_iou_class)

        print("%5.1f secs %5.1f MByte" % (time_elapsed, memMb))
        print('Best epoch is %d' % best_epoch)
        print('done!')



    avg_dice_whole = sum(allfold_dice_wt) / (len(allfold_dice_wt) * 1.0)
    std_dice_whole = np.std(allfold_dice_wt)
    print('dice_WT :')
    print(*allfold_dice_wt, sep=",")
    print('mean_dice_WT : ' + (str(avg_dice_whole)))
    print('std_dice_WT : ' + (str(std_dice_whole)))

    avg_dice_core = sum(allfold_dice_ct) / (len(allfold_dice_ct) * 1.0)
    std_dice_core = np.std(allfold_dice_ct)
    print('dice_CT :')
    print(*allfold_dice_ct, sep=",")
    print('mean_dice_CT : ' + (str(avg_dice_core)))
    print('std_dice_CT : ' + (str(std_dice_core)))

    avg_dice_enhance = sum(allfold_dice_et) / (len(allfold_dice_et) * 1.0)
    std_dice_enhance = np.std(allfold_dice_et)
    print('dice_ET :')
    print(*allfold_dice_et, sep=",")
    print('mean_dice_ET : ' + (str(avg_dice_enhance)))
    print('std_dice_ET : ' + (str(std_dice_enhance)))

    avg_sen_whole = sum(allfold_sen_wt) / (len(allfold_sen_wt) * 1.0)
    std_sen_whole = np.std(allfold_sen_wt)
    print('sen_WT :')
    print(*allfold_sen_wt, sep=",")
    print('mean_sen_WT : ' + (str(avg_sen_whole)))
    print('std_sen_WT : ' + (str(std_sen_whole)))

    avg_sen_core = sum(allfold_sen_ct) / (len(allfold_sen_ct) * 1.0)
    std_sen_core = np.std(allfold_sen_ct)
    print('sen_CT :')
    print(*allfold_sen_ct, sep=",")
    print('mean_sen_CT : ' + (str(avg_sen_core)))
    print('std_sen_CT : ' + (str(std_sen_core)))

    avg_sen_enhance = sum(allfold_sen_et) / (len(allfold_sen_et) * 1.0)
    std_sen_enhance = np.std(allfold_sen_et)
    print('sen_ET :')
    print(*allfold_sen_et, sep=",")
    print('mean_sen_ET : ' + (str(avg_sen_enhance)))
    print('std_sen_ET : ' + (str(std_sen_enhance)))

    avg_ppv_whole = sum(allfold_ppv_wt) / (len(allfold_ppv_wt) * 1.0)
    std_ppv_whole = np.std(allfold_ppv_wt)
    print('ppv_WT :')
    print(*allfold_ppv_wt, sep=",")
    print('mean_ppv_WT : ' + (str(avg_ppv_whole)))
    print('std_ppv_WT : ' + (str(std_ppv_whole)))

    avg_ppv_core = sum(allfold_ppv_ct) / (len(allfold_ppv_ct) * 1.0)
    std_ppv_core = np.std(allfold_ppv_ct)
    print('ppv_CT :')
    print(*allfold_ppv_ct, sep=",")
    print('mean_ppv_CT : ' + (str(avg_ppv_core)))
    print('std_ppv_CT : ' + (str(std_ppv_core)))

    avg_ppv_enhance = sum(allfold_ppv_et) / (len(allfold_ppv_et) * 1.0)
    std_ppv_enhance = np.std(allfold_ppv_et)
    print('ppv_ET :')
    print(*allfold_ppv_et, sep=",")
    print('mean_ppv_ET : ' + (str(avg_ppv_enhance)))
    print('std_ppv_ET : ' + (str(std_ppv_enhance)))

    avg_jac_whole = sum(allfold_jac_wt) / (len(allfold_jac_wt) * 1.0)
    std_jac_whole = np.std(allfold_jac_wt)
    print('jaccard_WT :')
    print(*allfold_jac_wt, sep=",")
    print('mean_jac_CT : ' + (str(avg_jac_whole)))
    print('std_jac_CT : ' + (str(std_jac_whole)))

    avg_jac_core = sum(allfold_jac_ct) / (len(allfold_jac_ct) * 1.0)
    std_jac_core = np.std(allfold_jac_ct)
    print('jaccard_CT :')
    print(*allfold_jac_ct, sep=",")
    print('mean_jac_CT : ' + (str(avg_jac_core)))
    print('std_jac_CT : ' + (str(std_jac_core)))

    avg_jac_enhance = sum(allfold_jac_et) / (len(allfold_jac_et) * 1.0)
    std_jac_enhance = np.std(allfold_jac_et)
    print('jaccard_ET :')
    print(*allfold_jac_et, sep=",")
    print('mean_jac_CT :' + (str(avg_jac_enhance)))
    print('std_jac_CT :' + (str(std_jac_enhance)))

    avg_ioulabel_core = sum(allfold_iou_label) / (len(allfold_iou_label) * 1.0)
    std_ioulabel_core = np.std(allfold_iou_label)
    print('mean_jac_CT : ' + (str(avg_ioulabel_core)))
    print('std_jac_CT : ' + (str(std_ioulabel_core)))

if __name__ == '__main__':
    main()









