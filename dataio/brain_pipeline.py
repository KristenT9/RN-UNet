import matplotlib.pyplot as plt
import nibabel as nib
import tensorflow as tf
from utils.config import *
from sklearn.feature_extraction.image import extract_patches_2d
from skimage.transform import rotate,rescale

class BrainPipeline(object):
    '''
    A class for processing brain scans for one patient
    INPUT:  filepath 'path': path to directory of one patient.
    '''

    def __init__(self, path, c,h,w,augs,mode):
        self.path = path
        self.c=c
        self.h = h
        self.w = w
        self.augs=augs
        self.mode=mode

    def read_scans_2d(self):
        '''
        Read 2D patches (patch_h,patch_w,c,modality)
        :return:(16*c,patch_h,patch_w,6)（16*c,patch_h,patch_w)
        '''
        patches=np.zeros((0,patch_h,patch_w,7))
        null_patches = np.zeros((0, patch_h, patch_w, 7))
        img= self.load_img()  #(h,w,c,7)
        count = np.sum(img[..., 6] == 1)
        print("Patients Path:",self.path,
              "\nThe number of 1 in GT", count)
        if self.mode==mode_list[0]:# unbalanced sample
            if count <= 552.5:
                patch_num = patches2d_per_pat * 4 # 16*4
            elif count > 552.5 and count <= 2101:
                patch_num = patches2d_per_pat * 3
            elif count>2101 and count<=5523.5:
                patch_num = patches2d_per_pat * 2
            else:
                patch_num=patches2d_per_pat
        else:
            patch_num = patches2d_per_pat
        for i in range(self.c):
            patch = extract_patches_2d(img[:,:,i,:], (patch_h, patch_w), max_patches=int(patch_num))
            # (max_patches,patch_h,patch_w,7)
            if self.mode==mode_list[0]:
                if sample_all or sample_prop:
                    for j in range(patch.shape[0]):
                        pi_0 = np.sum(patch[j][..., 6] == 0)
                        pi_1 = np.sum(patch[j][..., 6] == 1)
                        if pi_1==0 :
                            null_patches = np.concatenate((null_patches, patch[j][None, ...]), axis=0)
                        if sample_all and pi_1 != 0:
                            _patch = self._augment(patch[j][None, ...])
                            patches = np.concatenate((patches, _patch), axis=0)
                        elif sample_prop and (pi_0 / pi_1) < 9.0:  # 需要调整的比例；病灶体素>10%
                            _patch = self._augment(patch[j][None, ...])
                            patches = np.concatenate((patches, _patch), axis=0)
            else:
                patches = np.concatenate((patches, patch), axis=0)
        if self.mode==mode_list[0]:
            null_patch = random.sample(list(null_patches), int(patches.shape[0]//10))
            patches = np.concatenate((patches, null_patch), axis=0)
        print(patches.shape)
        return patches

    def img2d_for_predict(self):
        # generate 2D img for prediction(h,w,c,6)
        #return img(c,h,w,6) gt(c,h,w)
        img=self.load_img_pred()
        img=np.transpose(img,[2,0,1,3])
        return img

    def load_img_pred(self):
        print("\nProcessing MRI image...")
        if self.mode == "Testing":
            seq_list = name_list[0:6]
        else:
            seq_list = name_list
        imgpath = glob(self.path + '/*' + seq_list[0] + '*/' + '*Brain*.nii')  # read paths all 7 sequences
        s = nib.load(imgpath[0]).get_fdata()
        _h = s.shape[0]
        _w = s.shape[1]
        _c = s.shape[2]
        img = np.zeros((_h, _w,_c, 0))
        for j in range(len(seq_list)):
            imgpath=glob(self.path + '/*' + seq_list[j] + '*/' + '*Brain*.nii')# read paths all 7 sequences
            s = nib.load(imgpath[0]).get_fdata()
            if j==0:
                s=self.clip_range(s,min=0,max=2600,clip=1)
            if j==4:
                s = self.clip_range(s, min=0, max=20, clip=1)
            if j<6:
                s = self.scale_range(s, scale=0)
                s = self.normalize_img(s, normalize=1)
            img = np.concatenate((img, s[:, :, :, None]), axis=-1)
        print(img.shape)
        return img

    def load_img(self):
        img = np.zeros((self.h, self.w,self.c, 0))
        print("\nProcessing MRI image...")
        if self.mode=="Testing":
            seq_list=name_list[0:6]
        else:
            seq_list=name_list
        for j in range(len(seq_list)):
            imgpath=glob(self.path + '/*' + seq_list[j] + '*/' + '*Brain*.nii')# read paths all 7 sequences
            s = nib.load(imgpath[0]).get_fdata()

            print("Image Type:",name_list[j],
                  "\nThe max value in the image:",np.max(s),
                  "\nThe min value in the image:",np.min(s),
                  "\nThe shape of the image:",s.shape)

            s= self.resize_img(s,resize=1)
            if j==0:
                s=self.clip_range(s,min=0,max=2600,clip=1)
            if j==4:
                s = self.clip_range(s, min=0, max=20, clip=1)
            if j<6:
                s = self.scale_range(s, scale=0)
                s = self.normalize_img(s, normalize=1)
            img = np.concatenate((img, s[:, :, :, None]), axis=-1)
        print(img.shape)
        return img

    def scale_range(self,img,scale):
        if scale==1:
            img = (img - np.min(img)) / (np.max(img) - np.min(img))
        return img

    def normalize_img(self, img,normalize):
        if normalize==1:
            if np.std(img) == 0:
                return img
            else:
                return (img - np.mean(img)) / np.std(img)
        return img

    def flip_img(self,img):
        # 2D_img(num,patch_h,patch_w,7)
        flip_lr_prob = tf.random_uniform([], 0.0, 1.0)
        flip_ud_prob = tf.random_uniform([], 0.0, 1.0)
        img = tf.cond(tf.less(flip_lr_prob, set_prob),
                                    lambda: tf.image.flip_left_right(img),
                                    lambda: img)
        img = tf.cond(tf.less(flip_ud_prob, set_prob),
                      lambda: tf.image.flip_up_down(img),
                      lambda: img)
        return img

    def rotate_img(self,img):
        prob = tf.random_uniform([], 0.0, 1.0)
        if prob<set_prob:
            angle = tf.random_uniform([], 0, 180)
            img=tf.contrib.image.rotate(img,angle)
        return img

    def scale_img(self,img):
        prob = tf.random_uniform([], 0.0, 1.0)
        if prob<set_prob:
            scale = tf.random_uniform([], 0.5, 1.5)
            img=tf.image.resize_nearest_neighbor(img,[scale*patch_h,scale*patch_w])
            img=tf.image.resize_image_with_crop_or_pad(img,patch_h,patch_w)
        return img

    def translate_img(self,img,width_shift_range, height_shift_range):
        # 2D_img(num,patch_h,patch_w,7)
        #This fn will perform the horizontal or vertical shift
        prob = tf.random_uniform([], 0.0, 1.0)
        if prob<set_prob:
            if width_shift_range or height_shift_range:
                if width_shift_range:
                    p = min(4., width_shift_range * patch_w)
                    width_shift_range = tf.random_uniform([], -p, p)
                if height_shift_range:
                    p = min(4., height_shift_range * patch_h)
                    height_shift_range = tf.random_uniform([], -p, p)
                # Translate both
                img = tf.contrib.image.translate(img, [width_shift_range, height_shift_range])
        return img

    def clip_range(self,img,min,max,clip):
        if clip==1:
            img = np.clip(img, min, max)
        return img

    def resize_img(self,img,resize):
        if resize==1:
            img=tf.image.resize_image_with_crop_or_pad(img,h,w)
            remain = self.c - img.shape[2]
            pad_c = int(remain // 2)
            if int(remain % 2) == 1:
                padding = tf.constant([[0, 0], [0, 0], [pad_c, pad_c + 1]])
            else:
                padding = tf.constant([[0, 0], [0, 0], [pad_c, pad_c]])
            img = tf.pad(img, padding)
        return  img

    def _augment(self,img):
        # 2D_img(num,patch_h,patch_w,7)
        if "flip" in self.augs:
            img=self.flip_img(img)
        if "rotate" in self.augs:
            img=self.rotate_img(img)
        if "scale" in self.augs:
            img=self.scale_img(img)
        if "translate" in self.augs:
            img=self.translate_img(img,0.1,0.1)
        return img

    def visual_img_full(self, img, img_type="Original"):
        # 2D (16*c,patch_h,patch_w,6/1)
        # original (h,w,c,7)
        #visualize all slices in patches or images
        print(img.shape, img.dtype)
        if img_type == "2D":
            print("Visualize 2D patches")
            for j in range(img.shape[0]):
                print("The shape :", img[j].shape, "The range of gt:", np.unique(img[j][..., 6]))
                plt.figure(figsize=(20, 10))
                for i in range(6):
                    plt.subplot(1, 7, i + 1)
                    plt.title("Original image")
                    plt.imshow(img[j][:, :, i], cmap="gray")
                plt.subplot(1, 7, 7)
                plt.title("Ground Truth")
                plt.imshow(img[j][:, :, 6], cmap="gray")
                plt.show()
        else:
            print("Visualize Original images")
            dim_c = random.randint(0, int(img.shape[2] - 1))
            print("The shape :", img[dim_c].shape, "The range of gt:", np.unique(img[dim_c][..., 6]))
            plt.figure(figsize=(20, 10))
            for i in range(6):
                plt.subplot(1, 7, i + 1)
                plt.title("Original image")
                plt.imshow(img[:, :, dim_c, i], cmap="gray")
            plt.subplot(1, 7, 7)
            plt.title("Ground Truth")
            plt.imshow(img[:, :, dim_c, 6], cmap="gray")
            plt.show()


if __name__=="__main__":
    imgpath = '/home/huangli/Downloads/PyC-Proj/ISLES2017_Unet_my_result/Training/SMIR.my_result.188873.nii' # read paths all 7 sequences
    s = nib.load(imgpath).get_fdata()
    print("\nThe max value in the image:", np.max(s),
          "\nThe min value in the image:", np.min(s),
          "\nThe shape of the image:", s.shape)


