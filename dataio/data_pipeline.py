import matplotlib.pyplot as plt
import  dataio.brain_pipeline as bp
import h5py
from utils.config import *
import os


def visual_img_full(img, img_type="Original"):
    # 2D (16*c,patch_h,patch_w,6/1)
    # original (h,w,c,7)
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

def load_beta(path):
    f_name = os.path.basename(path)
    TICI = clinical_info[clinical_info["Case SMIR ID 1"] == f_name]["TICIScaleGrade"].values[0]
    if TICI in ["0", "1"]:
        beta = 2
    elif TICI in ["2", "2a", "2b"]:
        beta = 1
    else:
        beta = 3
    return beta

def data_pipeline_pred(mode,path,augs,index=0):
    for i in range(len(path)):
        img = bp.BrainPipeline(path[i], c, h, w, augs, mode).img2d_for_predict()
        with h5py.File(test_img_path + mode + "_2D_data_" + str(index) + str(i) + ".h5", "w") as f:
            f.create_dataset("slices_2d", data=img)
    return

def data_pipeline(mode,path,augs,index=0):
    batch_2d_patches = np.zeros((0, patch_h, patch_w, 7))
    batch_beta=np.zeros((0))
    for i in range(len(path)):
        img_gt = bp.BrainPipeline(path[i], c, h, w, augs, mode).read_scans_2d()
        beta = load_beta(path[i])
        beta = beta + np.zeros((img_gt.shape[0]))
        batch_beta = np.concatenate((batch_beta, beta), axis=0)
        batch_2d_patches = np.concatenate((batch_2d_patches, img_gt), axis=0)
    print(batch_2d_patches.shape)  # (num,patch_h,patch_w,7)
    with h5py.File(hdf5_path + mode + "_2D_data_" + str(index) + ".h5", "w") as f:
        f.create_dataset("batch_patches", data=batch_2d_patches)
        f.create_dataset("batch_beta", data=batch_beta)
    return


if __name__=="__main__":
    read_2d=True
    path = np.array(random.sample(testpath, 2))
    print(path)
    index=0
    data_pipeline_pred(mode=mode_list[2],path=path,augs=[])
    with h5py.File(hdf5_path + mode_list[2] + "_2D_data_"+str(index)+".h5", "r") as f:
        img = f["slices_2d"].value[..., 0:6]
    for j in range(img.shape[0]):
        print("The shape :", img[j].shape)
        plt.figure(figsize=(20, 10))
        for i in range(6):
            plt.subplot(1, 7, i + 1)
            plt.title("Original image")
            plt.imshow(img[j][:, :, i], cmap="gray")
        plt.show()



