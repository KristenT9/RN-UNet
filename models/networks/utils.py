from tensorflow.python.keras import layers,models,losses,optimizers,backend
import tensorflow as tf
from scipy.ndimage import _ni_support
from scipy.ndimage.morphology import distance_transform_edt, binary_erosion,\
    generate_binary_structure
from utils.config import *
from skimage.morphology import opening,closing,remove_small_objects

def confusion(gt,pred):
    pred_pos = backend.clip(pred, 0, 1)
    pred_neg = 1 - pred_pos
    gt_pos = backend.clip(gt, 0, 1)
    gt_neg = 1 - gt_pos
    tp = backend.sum(gt_pos * pred_pos)
    fp = backend.sum(gt_neg * pred_pos)
    fn = backend.sum(gt_pos * pred_neg)
    return tp,fp,fn

def prec(gt,pred):
    smooth = 1.
    tp,fp,fn=confusion(gt,pred)
    prec = (tp + smooth) / (tp + fp + smooth)
    return prec

def recall(gt,pred):
    smooth = 1.
    tp,fp,fn=confusion(gt,pred)
    recall = (tp + smooth) / (tp + fn + smooth)
    return recall

def tp(gt,pred):
    smooth = 1.
    pred_pos = backend.round(backend.clip(pred, 0, 1))
    gt_pos = backend.round(backend.clip(gt, 0, 1))
    tp = (backend.sum(gt_pos * pred_pos) + smooth)/ (backend.sum(gt_pos) + smooth)
    return tp

def tn(gt,pred):
    smooth = 1.
    pred_pos = backend.round(backend.clip(pred, 0, 1))#round（逐元素四舍五入）
    pred_neg = 1 - pred_pos
    gt_pos = backend.round(backend.clip(gt, 0, 1))
    gt_neg = 1 - gt_pos
    tn = (backend.sum(gt_neg * pred_neg) + smooth) / (backend.sum(gt_neg) + smooth )
    return tn

def hd(gt,pred):
    gt_f = gt[..., 0]
    pred_f = pred[..., 0]
    hd = tf.py_func(_hd, [pred_f,gt_f], backend.floatx())
    return hd

def _hd(result, reference):
    """
    Hausdorff Distance.

    Computes the (symmetric) Hausdorff Distance (HD) between the binary objects in two
    images. It is defined as the maximum surface distance between the objects.

    Parameters
    ----------
    result : array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    reference : array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    voxelspacing : float or sequence of floats, optional
        The voxelspacing in a distance unit i.e. spacing of elements
        along each dimension. If a sequence, must be of length equal to
        the input rank; if a single number, this is used for all axes. If
        not specified, a grid spacing of unity is implied.
    connectivity : int
        The neighbourhood/connectivity considered when determining the surface
        of the binary objects. This value is passed to
        `scipy.ndimage.morphology.generate_binary_structure` and should usually be :math:`> 1`.
        Note that the connectivity influences the result in the case of the Hausdorff distance.

    Returns
    -------
    hd : float
        The symmetric Hausdorff Distance between the object(s) in ```result``` and the
        object(s) in ```reference```. The distance unit is the same as for the spacing of
        elements along each dimension, which is usually given in mm.

    See also
    --------
    :func:`assd`
    :func:`asd`

    Notes
    -----
    This is a real metric. The binary images can therefore be supplied in any order.
    """
    voxelspacing = None
    connectivity = 1
    tshape = result.shape
    total_hd = 0.
    result[result >= 0.5] = 1
    result[result < 0.5] = 0
    for i in range(tshape[0]):
        if 0 == np.count_nonzero(result[i]) or 0 == np.count_nonzero(reference[i]):
            hd=0
        else:
            hd1 = _surface_distances(result[i], reference[i], voxelspacing, connectivity).max()
            hd2 = _surface_distances(reference[i], result[i], voxelspacing, connectivity).max()
            hd = max(hd1, hd2)
        total_hd = total_hd + hd
    return tf.cast((total_hd / tshape[0]),tf.float32)#.astype(np.float32)

def _surface_distances(result, reference, voxelspacing=None, connectivity=1):
    """
    The distances between the surface voxel of binary objects in result and their
    nearest partner surface voxel of a binary object in reference.
    """
    result = np.atleast_1d(result.astype(np.bool))
    reference = np.atleast_1d(reference.astype(np.bool))
    if voxelspacing is not None:
        voxelspacing = _ni_support._normalize_sequence(voxelspacing, result.ndim)
        voxelspacing = np.asarray(voxelspacing, dtype=np.float64)
        if not voxelspacing.flags.contiguous:
            voxelspacing = voxelspacing.copy()

    # binary structure
    footprint = generate_binary_structure(result.ndim, connectivity)

    # test for emptiness
    if 0 == np.count_nonzero(result):
        raise RuntimeError('The first supplied array does not contain any binary object.')
    if 0 == np.count_nonzero(reference):
        raise RuntimeError('The second supplied array does not contain any binary object.')

    # extract only 1-pixel border line of objects
    result_border = result ^ binary_erosion(result, structure=footprint, iterations=1)
    reference_border = reference ^ binary_erosion(reference, structure=footprint, iterations=1)

    # compute average surface distance
    # Note: scipys distance transform is calculated only inside the borders of the
    #       foreground objects, therefore the input has to be reversed
    dt = distance_transform_edt(~reference_border, sampling=voxelspacing)
    sds = dt[result_border]

    return sds

def assd(gt,pred):
    gt_f = gt[..., 0]
    pred_f = pred[..., 0]
    assd = tf.py_func(_assd, [pred_f,gt_f], backend.floatx())
    return assd

def _assd(result, reference):
    """
    Average symmetric surface distance.

    Computes the average symmetric surface distance (ASD) between the binary objects in
    two images.

    Parameters
    ----------
    result : array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    reference : array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    voxelspacing : float or sequence of floats, optional
        The voxelspacing in a distance unit i.e. spacing of elements
        along each dimension. If a sequence, must be of length equal to
        the input rank; if a single number, this is used for all axes. If
        not specified, a grid spacing of unity is implied.
    connectivity : int
        The neighbourhood/connectivity considered when determining the surface
        of the binary objects. This value is passed to
        `scipy.ndimage.morphology.generate_binary_structure` and should usually be :math:`> 1`.
        The decision on the connectivity is important, as it can influence the results
        strongly. If in doubt, leave it as it is.

    Returns
    -------
    assd : float
        The average symmetric surface distance between the object(s) in ``result`` and the
        object(s) in ``reference``. The distance unit is the same as for the spacing of
        elements along each dimension, which is usually given in mm.

    See also
    --------
    :func:`asd`
    :func:`hd`

    Notes
    -----
    This is a real metric, obtained by calling and averaging



    and



    The binary images can therefore be supplied in any order.
    """
    voxelspacing = None
    connectivity = 1
    tshape = result.shape
    total_assd = 0.
    result[result >= 0.5] = 1
    result[result < 0.5] = 0
    for i in range(tshape[0]):
        if 0 == np.count_nonzero(result[i]) or 0 == np.count_nonzero(reference[i]):
            assd=0
        else:
            assd = np.mean(
                (asd(result[i], reference[i], voxelspacing, connectivity),
                 asd(reference[i], result[i], voxelspacing, connectivity)))
        total_assd=total_assd+assd
    return tf.cast((total_assd/tshape[0]),tf.float32)#.astype(np.float32)

def asd(result, reference,voxelspacing = None,connectivity = 1):
    """
    Average surface distance metric.

    Computes the average surface distance (ASD) between the binary objects in two images.

    Parameters
    ----------
    result : array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    reference : array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    voxelspacing : float or sequence of floats, optional
        The voxelspacing in a distance unit i.e. spacing of elements
        along each dimension. If a sequence, must be of length equal to
        the input rank; if a single number, this is used for all axes. If
        not specified, a grid spacing of unity is implied.
    connectivity : int
        The neighbourhood/connectivity considered when determining the surface
        of the binary objects. This value is passed to
        `scipy.ndimage.morphology.generate_binary_structure` and should usually be :math:`> 1`.
        The decision on the connectivity is important, as it can influence the results
        strongly. If in doubt, leave it as it is.

    Returns
    -------
    asd : float
        The average surface distance between the object(s) in ``result`` and the
        object(s) in ``reference``. The distance unit is the same as for the spacing
        of elements along each dimension, which is usually given in mm.

    See also
    --------
    :func:`assd`
    :func:`hd`


    Notes
    -----
    This is not a real metric, as it is directed. See `assd` for a real metric of this.

    The method is implemented making use of distance images and simple binary morphology
    to achieve high computational speed.

    Examples
    --------
    The `connectivity` determines what pixels/voxels are considered the surface of a
    binary object. Take the following binary image showing a cross

    >>> from scipy.ndimage.morphology import generate_binary_structure
    >>> cross = generate_binary_structure(2, 1)
    array([[0, 1, 0],
           [1, 1, 1],
           [0, 1, 0]])

    With `connectivity` set to `1` a 4-neighbourhood is considered when determining the
    object surface, resulting in the surface

    .. code-block:: python

        array([[0, 1, 0],
               [1, 0, 1],
               [0, 1, 0]])

    Changing `connectivity` to `2`, a 8-neighbourhood is considered and we get:

    .. code-block:: python

        array([[0, 1, 0],
               [1, 1, 1],
               [0, 1, 0]])

    , as a diagonal connection does no longer qualifies as valid object surface.

    This influences the  results `asd` returns. Imagine we want to compute the surface
    distance of our cross to a cube-like object:

    >>> cube = generate_binary_structure(2, 1)
    array([[1, 1, 1],
           [1, 1, 1],
           [1, 1, 1]])

    , which surface is, independent of the `connectivity` value set, always

    .. code-block:: python

        array([[1, 1, 1],
               [1, 0, 1],
               [1, 1, 1]])

    Using a `connectivity` of `1` we get

    0.0

    while a value of `2` returns us

    0.20000000000000001

    due to the center of the cross being considered surface as well.

    """
    sds = _surface_distances(result, reference, voxelspacing, connectivity)
    asd = sds.mean()
    return asd

def dice(gt,pred):
    smooth=1.
    pred_mat=tf.reshape(pred,[-1])
    gt_mat=tf.reshape(gt,[-1])
    intersection=tf.reduce_sum(pred_mat*gt_mat)
    dice=(2.*intersection+smooth)/(tf.reduce_sum(pred_mat)+tf.reduce_sum(gt_mat)+smooth)
    return dice

def dice_loss(gt,pred):
    loss=1-dice(gt,pred)
    return loss

def res_conv_block(input_layers, num_filters):
    res_encoder = layers.Conv2D(num_filters, (1, 1), kernel_initializer=initializer, padding="same")(input_layers)
    res_encoder = layers.BatchNormalization()(res_encoder)
    res_encoder = layers.Activation("relu")(res_encoder)
    #encoder = layers.SeparableConv2D(num_filters, (3, 3), depthwise_initializer=initializer,
    #                                 pointwise_initializer=initializer, padding="same")(input_layers)
    encoder = layers.Conv2D(num_filters, (3, 3), kernel_initializer=initializer, padding="same")(input_layers)
    encoder = layers.BatchNormalization()(encoder)
    encoder = layers.Activation("relu")(encoder)
    encoder = layers.Conv2D(num_filters, (3, 3), kernel_initializer=initializer, padding="same")(encoder)
    encoder = layers.BatchNormalization()(encoder)
    encoder = layers.Activation("relu")(encoder)
    return encoder + res_encoder

def res_encoder_block(input_layers, num_filters):
    encoder=res_conv_block(input_layers, num_filters)
    encoder_pool=layers.MaxPool2D((2,2),strides=(2,2))(encoder)
    return encoder_pool,encoder

def res_decoder_block(input_layers, concat_tensor, num_filters):
    decoder = layers.Conv2DTranspose(num_filters, (2, 2), strides=(2, 2), kernel_initializer=initializer,
                                     padding="same")(input_layers)
    decoder = layers.concatenate([decoder, concat_tensor], axis=-1)
    decoder = layers.BatchNormalization()(decoder)
    decoder = layers.Activation("relu")(decoder)
    decoder = res_conv_block(decoder, num_filters)
    return decoder

def conv_block(input_layers,num_filters):
  encoder=layers.Conv2D(num_filters,(3,3),kernel_initializer=initializer,padding="same")(input_layers)
  encoder=layers.BatchNormalization()(encoder)
  encoder=layers.Activation("relu")(encoder)
  encoder=layers.Conv2D(num_filters,(3,3), kernel_initializer=initializer,padding="same")(encoder)
  encoder=layers.BatchNormalization()(encoder)
  encoder=layers.Activation("relu")(encoder)
  return encoder

def encoder_block(input_layers,num_filters):
  encoder=conv_block(input_layers,num_filters)
  encoder_pool=layers.MaxPool2D((2,2),strides=(2,2))(encoder)
  return encoder_pool,encoder

def decoder_block(input_layers,concat_tensor,num_filters):
  decoder=layers.Conv2DTranspose(num_filters,(2,2),strides=(2,2),kernel_initializer=initializer,padding="same")(input_layers)
  decoder=layers.concatenate([decoder,concat_tensor],axis=-1)
  decoder=layers.BatchNormalization()(decoder)
  decoder=layers.Activation("relu")(decoder)
  decoder=layers.Conv2D(num_filters,(3,3),kernel_initializer=initializer,padding="same")(decoder)
  decoder=layers.BatchNormalization()(decoder)
  decoder=layers.Activation("relu")(decoder)
  decoder=layers.Conv2D(num_filters,(3,3), kernel_initializer=initializer,padding="same")(decoder)
  decoder=layers.BatchNormalization()(decoder)
  decoder=layers.Activation("relu")(decoder)
  return decoder

def dsv_block(input_layers,num_filters,scale_factor):
    dsv=layers.Conv2D(num_filters,kernel_size=1,strides=1,kernel_initializer=initializer,padding="same")(input_layers)
    dsv=layers.UpSampling2D(size=scale_factor,interpolation='bilinear')(dsv)
    return dsv


if __name__=="__main__":
    gt=np.array([[0,1,0,1,0],[1,1,0,0,0]])
    pred=np.array([[0,0.4,0.2,0.6,0.3],[0.3,0.4,0.2,0.6,0.3]])
    pred[pred > 0.5] = 1.
    pred[pred <= 0.5] = 0.
    hd=hd(gt,pred)
    assd=assd(gt,pred)