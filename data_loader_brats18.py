from torch.utils.data import Dataset

from src.utils import *

modals = ['flair', 't1', 't1ce', 't2']


class BraTS18DataLoader(Dataset):
    def __init__(self, data_dir, conf='/media/data1/suriza/MICCAI_BraTS2020_TrainingData_2/all.txt', train=True):
        img_lists = []
        train_config = open(conf).readlines()
        for data in train_config:
            img_lists.append(os.path.join(data_dir, data.strip('\n')))

        print('\n' + '~' * 50)
        print('******** Loading data from disk ********')
        self.data = []
        self.freq = np.zeros(5) #for label 0,1,2,3,4
        self.zero_vol = np.zeros((4, 240, 240))  #
        count = 0
        for subject in img_lists:
            count += 1
            if count % 10 == 0:
                print('loading subject %d' %count)
            volume, label = BraTS18DataLoader.get_subject(subject) # 4 * 155 * 240 * 240,  155 * 240 * 240
            print(volume.shape)
            volume = norm_vol(volume)

            self.freq += self.get_freq(label)
            if train is True:
                length = volume.shape[1]
                for i in range(length):
                    name = subject + '=slice' + str(i)
                    if (volume[:, i, :, :] == self.zero_vol).all():  # when training, ignore zero data
                        continue
                    else:
                        self.data.append([volume[:, i, :, :], label[i, :, :], name])
            else:
                volume = np.transpose(volume, (1, 0, 2, 3))
                self.data.append([volume, label, subject])

        self.freq = self.freq / np.sum(self.freq)
        self.weight = np.median(self.freq) / self.freq
        print('********  Finish loading data  ********')
        print('********  Weight for all classes  ********')
        print(self.weight)
        if train is True:
            print('********  Total number of 2D images is ' + str(len(self.data)) + ' **********')
        else:
            print('********  Total number of subject is ' + str(len(self.data)) + ' **********')

        print('~' * 50)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # ********** get file dir **********
        [image, label, name] = self.data[index]  # get whole data for one subject

        # print(image.dtype)
        # print(label.dtype)
        # print(np.unique(image))
        # print(np.unique(label))

        # ********** change data type from numpy to torch.Tensor **********
        image = torch.from_numpy(image).float()  # Float Tensor 4, 240, 240
        label = np.asarray(label, dtype='uint8')
        label = torch.from_numpy(label).float()
        # print(label.dtype)# Float Tensor 240, 240
        return image, label, name

    @staticmethod
    def get_subject(subject):
        """
        :param_subject: absolute dir
        :return:
        volume      4D numpy    4 * 155 *240 * 240
        label       4D numpy 155 * 240 * 240 * 240
        """
        #get file
        files = os.listdir(subject)
        multi_mode_dir = []
        label_dir = ""
        for f in files:
            if f == '.nii':
                continue
            if 'flair' in f or 't1' in f or 't2' in f: #if is data
                multi_mode_dir.append(f)
            elif 'seg' in f:
                label_dir = f

        #load 4 mode images
        multi_mode_imgs = []
        for mod_dir in multi_mode_dir:
            path = os.path.join(subject, mod_dir)
            img = load_mha_as_array(path)
            multi_mode_imgs.append(img)
        print(multi_mode_dir)

        #get label
        label_dir = os.path.join(subject, label_dir)
        label = load_mha_as_array(label_dir)

        volume = np.asarray(multi_mode_imgs)
        return volume, label

    def get_freq(self,label):
        """
        :param label: numpy 155 * 240 * 240     val: 0,1,2,3,4
        :return:
        """
        class_count = np.zeros((5))
        for i in range(5):
            a = (label == i) + 0
            class_count[i] = np.sum(a)
        return class_count


#test case
if __name__ == "__main__":
    vol_num = 4
    data_dir = '/media/data1/suriza/MICCAI_BraTS2020_TrainingData_2'
    conf = '/media/data1/suriza/MICCAI_BraTS2020_TrainingData_2/all.txt'
    #test data loader for training data
    brats18 = BraTS18DataLoader(data_dir, conf=conf, train = True)
    image2d, label2d, im_name = brats18[60]

    print('image size...')
    print(image2d.shape)

    print('label size...')
    print(label2d.shape)
    print(im_name)
    name = im_name.split('/')[1]
    save_one_image_label(image2d, label2d, 'img5/img_label_%s.jpg' %name)

    #test data loader for testing data
    brats18_test = BraTS18DataLoader(data_dir=data_dir, conf=conf, train=False)
    image_volume, label_volume, subject = brats18_test[0]
    print(image_volume.shape)
    print(label_volume.shape)
    print(subject)