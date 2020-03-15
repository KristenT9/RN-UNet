import tensorflow as tf
tf.enable_eager_execution()
from matplotlib import pyplot as plt
import models.networks.U_Net as unet
import models.networks.Unet_nonlocal as unet_nonlocal
import dataio.brain_pipeline as bp
from utils.config import *
from tensorflow.python.keras import callbacks
from sklearn.model_selection import KFold
import dataio.data_pipeline as dp
import h5py
import math
import nibabel as nib
from models.networks.utils import *
from functools import reduce


def visualize_loss(history):
    print(history.history.keys())
    plt.figure(figsize=(40,40))
    #num_epochs = range(epochs)
    keys=list(history.history.keys())
    for i in range(len(keys)//2):
        metric = history.history[keys[i]]
        val_metric = history.history["val_"+keys[i]]
        plt.subplot(math.ceil(len(keys)//4), 2, i+1)
        plt.plot(range(len(metric)), metric, label="Training "+keys[i])
        plt.plot(range(len(val_metric)), val_metric, label="Validation "+keys[i])
        plt.legend(loc="upper right",fontsize="x-small")
        plt.title("Training and Validation "+keys[i])
        plt.tight_layout()
        plt.subplots_adjust(left=0.07, bottom=0.07, top=0.93, right=0.93, hspace=0.4, wspace=0.3)
    plt.show()

def train_model(model_name,weights_save_path):
    with h5py.File(hdf5_path + mode_list[0] + "_2D_data_" + str(index) + ".h5", "r") as f:
        train_img = f["batch_patches"].value[..., 0:6]  # (num,patch_h,patch_w,6)
        train_gt = f["batch_patches"].value[..., 6]  # (num,patch_h,patch_w)
        train_beta = f["batch_beta"].value
    with h5py.File(hdf5_path + mode_list[1] + "_2D_data_" + str(index) + ".h5", "r") as f:
        val_img = f["batch_patches"].value[..., 0:6]  # (num,patch_h,patch_w,6)
        val_gt = f["batch_patches"].value[..., 6]  # (num,patch_h,patch_w)
        val_beta = f["batch_beta"].value
    print(train_img.shape)
    if model_name == "Unet":
        model = unet.unet(input_height=patch_h, input_width=patch_w)
    elif model_name == "Nonlocal dsv Unet":
        model = unet_nonlocal.unet_nonlocal(input_height=patch_h, input_width=patch_w)
    else:
        model = unet_nonlocal.xunet_nonlocal(input_height=patch_h, input_width=patch_w)
    cp = [callbacks.EarlyStopping(monitor='val_dice',
                                  patience=10,
                                  mode='max'),
          callbacks.ModelCheckpoint(filepath=weights_save_path,
                                    monitor='val_dice',
                                    save_best_only=True,
                                    save_weights_only=True,
                                    mode='max',
                                    verbose=1)]
    print("Training " + model_name + " Model")
    if load_weights == True:
        print("Loading " + model_name + " Model Weights")
        model.load_weights(weights_save_path)
    if load_beta == True:
        inputs = [train_img, train_gt[..., None], train_beta]
        history = model.fit(inputs, [], batch_size=batch_size, epochs=epochs,
                            validation_data=([val_img, val_gt[..., None], val_beta], []),
                            shuffle=True, callbacks=cp)
        eval_metrics = model.evaluate([val_img, val_gt[..., None], val_beta], [])
    else:
        history = model.fit(train_img, train_gt[..., None], batch_size=batch_size, epochs=epochs,
                            validation_data=(val_img, val_gt[..., None]),
                            shuffle=True, callbacks=cp)
        eval_metrics = model.evaluate(val_img, val_gt[..., None])
    visualize_loss(history)
    return eval_metrics

def predict_model(model_name,id,mode,path,weights_path,cv_index):
    with h5py.File(test_img_path + mode + "_2D_data_" + str(cv_index)+ str(id) + ".h5", "r") as f:
        if mode==mode_list[0] or mode_list[1]:
            test_img = f["slices_2d"].value[..., 0:6]  # (32 ,h,w,6)
            gt_img = f["slices_2d"].value[..., 6:]
        else:
            test_img = f["slices_2d"].value
    c = test_img.shape[0]
    h = test_img.shape[1]
    w = test_img.shape[2]
    print(path,c,h,w)
    if model_name == "Unet":
        model = unet.unet(input_height=h, input_width=w)
    elif model_name == "Nonlocal dsv Unet":
        model = unet_nonlocal.unet_nonlocal(input_height=h, input_width=w)
    else:
        model = unet_nonlocal.xunet_nonlocal(input_height=h, input_width=w)
    print("Loading " + model_name + " Model Weights",weights_path.split('/')[-1])
    model.load_weights(weights_path)
    eval_res = model.evaluate(test_img, gt_img, batch_size=batch_size)


    pred = model.predict(test_img,batch_size=batch_size)# (32, 256, 256, 1)
    pred[pred > 0.5] = 1.
    pred[pred <= 0.5] = 0.
    pred = np.transpose(pred,[1,2,0,3])
    test_img = np.transpose(test_img,[1,2,0,3]) # (256,256,32,6)
    adc_path = glob(path + '/*' + "ADC" + '*/' + '*Brain*.nii')[0]
    mtt_ID = glob(path + '/*' + "MTT" + '*' )[0].split("/")[-1].split(".")[-1]
    adc_img = nib.load(adc_path)
    inter_img = adc_img.get_data()
    inter_img[:] = pred[...,0]
    adc_img.set_data_dtype(np.uint16)
    nib.save(adc_img,str(pred_saved_path+mode+str(cv_index)+'/SMIR.my_result.'+mtt_ID+'.nii'))
    return eval_res

def predict_model_ensemble(model_name,id,mode,path,weights_path_list):
    with h5py.File(test_img_path + mode + "_2D_data_" + str(id) + ".h5", "r") as f:
        if mode==mode_list[0] or mode_list[1]:
            test_img = f["slices_2d"].value[..., 0:6]  # (32 ,h,w,6)
        else:
            test_img = f["slices_2d"].value
    c = test_img.shape[0]
    h = test_img.shape[1]
    w = test_img.shape[2]
    print(path,c,h,w)
    if model_name == "Unet":
        model = unet.unet(input_height=h, input_width=w)
    elif model_name == "Nonlocal dsv Unet":
        model = unet_nonlocal.unet_nonlocal(input_height=h, input_width=w)
    else:
        model = unet_nonlocal.xunet_nonlocal(input_height=h, input_width=w)
    pred_list=[]
    for we_index in range(len(weights_path_list)):
        print("Loading " + model_name + " Model Weights", weights_path_list[we_index].split('/')[-1])
        model.load_weights(weights_path_list[we_index])
        pred = model.predict(test_img,batch_size=4)  # (32, 256, 256, 1)
        pred = np.transpose(pred, [1, 2, 0, 3])  # (256,256,32,1)
        pred = pred[...,0]
        pred_list.append(pred) # (256,256,32)
    pred_list=np.array(pred_list) # (7,256,256,32)
    pred=np.mean(pred_list,axis=0)# (256,256,32)
    pred[pred > 0.5] = 1.
    pred[pred <= 0.5] = 0.
    adc_path = glob(path + '/*' + "ADC" + '*/' + '*Brain*.nii')[0]
    mtt_ID = glob(path + '/*' + "MTT" + '*' )[0].split("/")[-1].split(".")[-1]
    adc_img = nib.load(adc_path)
    inter_img = adc_img.get_data()
    inter_img[:] = pred
    adc_img.set_data_dtype(np.uint16)
    nib.save(adc_img,str(pred_saved_path+mode+'/SMIR.my_result.'+mtt_ID+'.nii'))
    return

def visualize_res(img,pred,gt,img_2d):
    # (32,256,256,6) img_2D (32,256,256,1) pred_2D,gt
    print("Shape of Original Image and Ground Truth: ",img.shape,"/", gt.shape)
    plt.figure(figsize=(50, 20))
    if img_2d==False:
        dim_1 = random.randint(0, img.shape[3] - 1)
        for i in range(6):
            plt.subplot(1, 8, i+1)
            plt.title("Original image")
            plt.imshow(img[0][:, :,dim_1, i], cmap="gray")
        plt.subplot(1,8,7)
        plt.title("Predicted Mask")
        plt.imshow(pred[0][:,:,dim_1,0],cmap="gray")
        plt.subplot(1,8,8)
        plt.title("Ground Truth")
        plt.imshow(gt[0][:, :, dim_1,0], cmap="gray")
    else:
        dim_1 = random.randint(0, img.shape[0] - 1)
        for i in range(6):
            plt.subplot(1, 8, i+1)
            plt.title("Original image")
            plt.imshow(img[dim_1][:, :, i], cmap="gray")
        plt.subplot(1,8,7)
        plt.title("Predicted Mask")
        plt.imshow(pred[dim_1][:,: ,0],cmap="gray")
        plt.subplot(1,8,8)
        plt.title("Ground Truth")
        plt.imshow(gt[dim_1][:, :, 0], cmap="gray")
    plt.tight_layout()
    plt.show()
    return

def print_metrics(mat_metric,path_list,cv_id):
    mat_metric = np.array(mat_metric)
    print(mat_metric, type(mat_metric), mat_metric.shape,type(path_list))

    data_df_1 = pd.DataFrame(mat_metric)
    data_df_2 = pd.DataFrame(path_list)
    # create and writer pd.DataFrame to excel
    writer = pd.ExcelWriter(model_name+'_Save_Excel_'+str(cv_id)+'.xlsx')
    data_df_1.to_excel(writer, 'page_1')  # , float_format='%.5f')  # float_format 控制精度
    data_df_2.to_excel(writer, 'page_2')
    writer.save()
    avg_metric = np.mean(mat_metric, axis=0)
    std_metric = np.std(mat_metric, axis=0)
    print("\n", "K-flod Cross Validation:", "\n")
    for i in range(avg_metric.shape[0]):
        print("Average ", metrics[i], ": ", avg_metric[i])
        print("Std. ", metrics[i], ": ", std_metric[i], "\n")


if __name__=="__main__":
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=100)
    load_weights = False
    if model_name == "TI-Attention dsv Nonlocal Unet":
        load_beta = True
    else:
        load_beta = False
    model_path=[]
    save_data = False
    mat_metric = []

    val_path_list=[['/home/huangli/Downloads/PyC-Proj/ISLES2017/training_46',
                    '/home/huangli/Downloads/PyC-Proj/ISLES2017/training_11',
                    '/home/huangli/Downloads/PyC-Proj/ISLES2017/training_23',
                    '/home/huangli/Downloads/PyC-Proj/ISLES2017/training_31',
                    '/home/huangli/Downloads/PyC-Proj/ISLES2017/training_12',
                    '/home/huangli/Downloads/PyC-Proj/ISLES2017/training_1',
                    '/home/huangli/Downloads/PyC-Proj/ISLES2017/training_44'],
                   ['/home/huangli/Downloads/PyC-Proj/ISLES2017/training_2',
                    '/home/huangli/Downloads/PyC-Proj/ISLES2017/training_26',
                    '/home/huangli/Downloads/PyC-Proj/ISLES2017/training_39',
                    '/home/huangli/Downloads/PyC-Proj/ISLES2017/training_40',
                    '/home/huangli/Downloads/PyC-Proj/ISLES2017/training_33',
                    '/home/huangli/Downloads/PyC-Proj/ISLES2017/training_6'],
                   ['/home/huangli/Downloads/PyC-Proj/ISLES2017/training_9',
                    '/home/huangli/Downloads/PyC-Proj/ISLES2017/training_47',
                    '/home/huangli/Downloads/PyC-Proj/ISLES2017/training_43',
                    '/home/huangli/Downloads/PyC-Proj/ISLES2017/training_45',
                    '/home/huangli/Downloads/PyC-Proj/ISLES2017/training_14',
                    '/home/huangli/Downloads/PyC-Proj/ISLES2017/training_48'],
                   ['/home/huangli/Downloads/PyC-Proj/ISLES2017/training_20',
                    '/home/huangli/Downloads/PyC-Proj/ISLES2017/training_21',
                    '/home/huangli/Downloads/PyC-Proj/ISLES2017/training_32',
                    '/home/huangli/Downloads/PyC-Proj/ISLES2017/training_35',
                    '/home/huangli/Downloads/PyC-Proj/ISLES2017/training_42',
                    '/home/huangli/Downloads/PyC-Proj/ISLES2017/training_36'],
                   ['/home/huangli/Downloads/PyC-Proj/ISLES2017/training_41',
                    '/home/huangli/Downloads/PyC-Proj/ISLES2017/training_30',
                    '/home/huangli/Downloads/PyC-Proj/ISLES2017/training_7',
                    '/home/huangli/Downloads/PyC-Proj/ISLES2017/training_16',
                    '/home/huangli/Downloads/PyC-Proj/ISLES2017/training_10',
                    '/home/huangli/Downloads/PyC-Proj/ISLES2017/training_13'],
                   ['/home/huangli/Downloads/PyC-Proj/ISLES2017/training_8',
                    '/home/huangli/Downloads/PyC-Proj/ISLES2017/training_28',
                    '/home/huangli/Downloads/PyC-Proj/ISLES2017/training_22',
                    '/home/huangli/Downloads/PyC-Proj/ISLES2017/training_27',
                    '/home/huangli/Downloads/PyC-Proj/ISLES2017/training_18',
                    '/home/huangli/Downloads/PyC-Proj/ISLES2017/training_15'],
                   ['/home/huangli/Downloads/PyC-Proj/ISLES2017/training_38',
                    '/home/huangli/Downloads/PyC-Proj/ISLES2017/training_4',
                    '/home/huangli/Downloads/PyC-Proj/ISLES2017/training_19',
                    '/home/huangli/Downloads/PyC-Proj/ISLES2017/training_37',
                    '/home/huangli/Downloads/PyC-Proj/ISLES2017/training_24',
                    '/home/huangli/Downloads/PyC-Proj/ISLES2017/training_5']]

    index = 0

    for train_path_index,val_path_index in kfold.split(filepath):
        index=index+1
        model_path_index=model_save_path + '/' + model_name + str(index) + '.hdf5'
        model_path.append(model_path_index)
        filepath=np.array(filepath)
        val_path = val_path_list[index-1]
        train_path = list(set(filepath)-set(val_path))
        print("Validation patient's path:\n",val_path,
              "\nmodel_path_index",model_path_index.split('/')[-1])
        if save_data == True:
            dp.data_pipeline(mode=mode_list[0], path=train_path, augs=augs, index=index)
            dp.data_pipeline(mode=mode_list[1], path=val_path,  augs=[], index=index)
            dp.data_pipeline_pred(mode=mode_list[2], path=testpath, augs=[], index=index)

        print("The experiment of", model_name)
        train_model(model_name, model_path_index)

        for id in range(len(val_path)):
            eval_metrics = predict_model(model_name, id, mode_list[1], val_path[id], model_path_index, cv_index=index)#
            mat_metric.append(eval_metrics)

        print_metrics(np.array(mat_metric).reshape(-1, 6), np.array(val_path).reshape(-1), index)
    print_metrics(np.array(mat_metric).reshape(-1,6), np.array(val_path_list).reshape(-1), 0)
    for i in range(test_num):
        predict_model_ensemble(model_name, i, mode_list[2], testpath[i], model_path)




