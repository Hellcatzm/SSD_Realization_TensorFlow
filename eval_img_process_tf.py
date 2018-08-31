# Author : hellcat
# Time   : 18-8-30

"""
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
 
import numpy as np
np.set_printoptions(threshold=np.inf)
 
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
"""
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import variables
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import math_ops
import util_tf


from enum import Enum, IntEnum
# Resizing strategies.
Resize = IntEnum('Resize', ('NONE',                # Nothing!
                            'CENTRAL_CROP',        # Crop (and pad if necessary).
                            'PAD_AND_RESIZE',      # Pad, and resize to output shape.
                            'WARP_RESIZE'))        # Warp resize.
# VGG mean parameters.
_R_MEAN = 123.
_G_MEAN = 117.
_B_MEAN = 104.
EVAL_SIZE = (300, 300)


def _is_tensor(x):
    """Returns `True` if `x` is a symbolic tensor-like object.
    Args:
      x: A python object to check.
    Returns:
      `True` if `x` is a `tf.Tensor` or `tf.Variable`, otherwise `False`.
    """
    return isinstance(x, (ops.Tensor, variables.Variable))


def _assert(cond, ex_type, msg):
    """A polymorphic assert, works with tensors and boolean expressions.
    If `cond` is not a tensor, behave like an ordinary assert statement, except
    that a empty list is returned. If `cond` is a tensor, return a list
    containing a single TensorFlow assert op.
    Args:
      cond: Something evaluates to a boolean value. May be a tensor.
      ex_type: The exception class to use.
      msg: The error message.
    Returns:
      A list, containing at most one assert op.
    """
    if _is_tensor(cond):
        return [control_flow_ops.Assert(cond, [msg])]
    else:
        if not cond:
            raise ex_type(msg)
        else:
            return []


def _Check3DImage(image, require_static=True):
    """Assert that we are working with properly shaped image.
    Args:
      image: 3-D Tensor of shape [height, width, channels]
        require_static: If `True`, requires that all dimensions of `image` are
        known and non-zero.
    Raises:
      ValueError: if `image.shape` is not a 3-vector.
    Returns:
      An empty list, if `image` has fully defined dimensions. Otherwise, a list
        containing an assert op is returned.
    """
    try:
        image_shape = image.get_shape().with_rank(3)
    except ValueError:
        raise ValueError("'image' must be three-dimensional.")
    if require_static and not image_shape.is_fully_defined():
        raise ValueError("'image' must be fully defined.")
    if any(x == 0 for x in image_shape):
        raise ValueError("all dims of 'image.shape' must be > 0: %s" %
                         image_shape)
    if not image_shape.is_fully_defined():
        return [check_ops.assert_positive(array_ops.shape(image),
                                          ["all dims of 'image.shape' "
                                           "must be > 0."])]
    else:
        return []


def bboxes_crop_or_pad(bboxes,
                       height, width,
                       offset_y, offset_x,
                       target_height, target_width):
    """Adapt bounding boxes to crop or pad operations.
    Coordinates are always supposed to be relative to the image.

    Arguments:
      bboxes: Tensor Nx4 with bboxes coordinates [y_min, x_min, y_max, x_max];
      height, width: Original image dimension;
      offset_y, offset_x: Offset to apply,
        negative if cropping, positive if padding;
      target_height, target_width: Target dimension after cropping / padding.
    """
    with tf.name_scope('bboxes_crop_or_pad'):
        # Rescale bounding boxes in pixels.
        scale = tf.cast(tf.stack([height, width, height, width]), bboxes.dtype)
        bboxes = bboxes * scale
        # Add offset.
        offset = tf.cast(tf.stack([offset_y, offset_x, offset_y, offset_x]), bboxes.dtype)
        bboxes = bboxes + offset
        # Rescale to target dimension.
        scale = tf.cast(tf.stack([target_height, target_width,
                                  target_height, target_width]), bboxes.dtype)
        bboxes = bboxes / scale
        return bboxes


def _ImageDimensions(image):
    """Returns the dimensions of an image tensor.
    Args:
      image: A 3-D Tensor of shape `[height, width, channels]`.
    Returns:
      A list of `[height, width, channels]` corresponding to the dimensions of the
        input image.  Dimensions that are statically known are python integers,
        otherwise they are integer scalar tensors.
    """
    if image.get_shape().is_fully_defined():
        return image.get_shape().as_list()
    else:
        static_shape = image.get_shape().with_rank(3).as_list()
        dynamic_shape = array_ops.unstack(array_ops.shape(image), 3)
        return [s if s is not None else d
                for s, d in zip(static_shape, dynamic_shape)]


def resize_image_bboxes_with_crop_or_pad(image, bboxes,
                                         target_height, target_width):
    """Crops and/or pads an image to a target width and height.
    Resizes an image to a target width and height by either centrally
    cropping the image or padding it evenly with zeros.

    If `width` or `height` is greater than the specified `target_width` or
    `target_height` respectively, this op centrally crops along that dimension.
    If `width` or `height` is smaller than the specified `target_width` or
    `target_height` respectively, this op centrally pads with 0 along that
    dimension.
    Args:
      image: 3-D tensor of shape `[height, width, channels]`
      target_height: Target height.
      target_width: Target width.
    Raises:
      ValueError: if `target_height` or `target_width` are zero or negative.
    Returns:
      Cropped and/or padded image of shape
        `[target_height, target_width, channels]`
    """
    with tf.name_scope('resize_with_crop_or_pad'):
        image = ops.convert_to_tensor(image, name='image')

        assert_ops = []
        assert_ops += _Check3DImage(image, require_static=False)
        assert_ops += _assert(target_width > 0, ValueError,
                              'target_width must be > 0.')
        assert_ops += _assert(target_height > 0, ValueError,
                              'target_height must be > 0.')

        image = control_flow_ops.with_dependencies(assert_ops, image)
        # `crop_to_bounding_box` and `pad_to_bounding_box` have their own checks.
        # Make sure our checks come first, so that error messages are clearer.
        if _is_tensor(target_height):
            target_height = control_flow_ops.with_dependencies(
                assert_ops, target_height)
        if _is_tensor(target_width):
            target_width = control_flow_ops.with_dependencies(assert_ops, target_width)

        def max_(x, y):
            if _is_tensor(x) or _is_tensor(y):
                return math_ops.maximum(x, y)
            else:
                return max(x, y)

        def min_(x, y):
            if _is_tensor(x) or _is_tensor(y):
                return math_ops.minimum(x, y)
            else:
                return min(x, y)

        def equal_(x, y):
            if _is_tensor(x) or _is_tensor(y):
                return math_ops.equal(x, y)
            else:
                return x == y

        height, width, _ = _ImageDimensions(image)
        width_diff = target_width - width
        offset_crop_width = max_(-width_diff // 2, 0)
        offset_pad_width = max_(width_diff // 2, 0)

        height_diff = target_height - height
        offset_crop_height = max_(-height_diff // 2, 0)
        offset_pad_height = max_(height_diff // 2, 0)

        # Maybe crop if needed.
        height_crop = min_(target_height, height)
        width_crop = min_(target_width, width)
        cropped = tf.image.crop_to_bounding_box(image, offset_crop_height, offset_crop_width,
                                                height_crop, width_crop)
        bboxes = bboxes_crop_or_pad(bboxes,
                                    height, width,
                                    -offset_crop_height, -offset_crop_width,
                                    height_crop, width_crop)
        # Maybe pad if needed.
        resized = tf.image.pad_to_bounding_box(cropped, offset_pad_height, offset_pad_width,
                                               target_height, target_width)
        bboxes = bboxes_crop_or_pad(bboxes,
                                    height_crop, width_crop,
                                    offset_pad_height, offset_pad_width,
                                    target_height, target_width)

        # In theory all the checks below are redundant.
        if resized.get_shape().ndims is None:
            raise ValueError('resized contains no shape.')

        resized_height, resized_width, _ = _ImageDimensions(resized)

        assert_ops = []
        assert_ops += _assert(equal_(resized_height, target_height), ValueError,
                              'resized height is not correct.')
        assert_ops += _assert(equal_(resized_width, target_width), ValueError,
                              'resized width is not correct.')

        resized = control_flow_ops.with_dependencies(assert_ops, resized)
        return resized, bboxes


def tf_image_whitened(image, means=(_R_MEAN, _G_MEAN, _B_MEAN)):
    """Subtracts the given means from each image channel.

    Returns:
        the centered image.
    """
    if image.get_shape().ndims != 3:
        raise ValueError('Input must be of size [height, width, C>0]')
    num_channels = image.get_shape().as_list()[-1]
    if len(means) != num_channels:
        raise ValueError('len(means) must match the number of channels')

    mean = tf.constant(means, dtype=image.dtype)
    image = image - mean
    return image


def preprocess_for_eval(image, labels, bboxes,
                        out_shape=EVAL_SIZE, data_format='NHWC',
                        difficults=None, resize=Resize.WARP_RESIZE,
                        scope='ssd_preprocessing_train'):
    """Preprocess an image for evaluation.
    Returns:
        A preprocessed image.
    """
    with tf.name_scope(scope):
        if image.get_shape().ndims != 3:
            raise ValueError('Input must be of size [height, width, C>0]')

        image = tf.to_float(image)
        image = tf_image_whitened(image, [_R_MEAN, _G_MEAN, _B_MEAN])

        # Add image rectangle to bboxes.
        bbox_img = tf.constant([[0., 0., 1., 1.]])
        if bboxes is None:
            bboxes = bbox_img
        else:
            bboxes = tf.concat([bbox_img, bboxes], axis=0)

        if resize == Resize.NONE:
            # No resizing...
            pass
        elif resize == Resize.CENTRAL_CROP:
            # Central cropping of the image.
            image, bboxes = resize_image_bboxes_with_crop_or_pad(
                image, bboxes, out_shape[0], out_shape[1])
        elif resize == Resize.PAD_AND_RESIZE:
            # Resize image first: find the correct factor...
            shape = tf.shape(image)
            factor = tf.minimum(tf.to_double(1.0),
                                tf.minimum(tf.to_double(out_shape[0] / shape[0]),
                                           tf.to_double(out_shape[1] / shape[1])))
            resize_shape = factor * tf.to_double(shape[0:2])
            resize_shape = tf.cast(tf.floor(resize_shape), tf.int32)

            image = util_tf.resize_image(image, resize_shape,
                                         method=tf.image.ResizeMethod.BILINEAR,
                                         align_corners=False)
            # Pad to expected size.
            image, bboxes = resize_image_bboxes_with_crop_or_pad(
                image, bboxes, out_shape[0], out_shape[1])
        elif resize == Resize.WARP_RESIZE:
            # Warp resize of the image.
            image = util_tf.resize_image(image, out_shape,
                                         method=tf.image.ResizeMethod.BILINEAR,
                                         align_corners=False)

        # Split back bounding boxes.
        bbox_img = bboxes[0]
        bboxes = bboxes[1:]
        # Remove difficult boxes.
        if difficults is not None:
            mask = tf.logical_not(tf.cast(difficults, tf.bool))
            labels = tf.boolean_mask(labels, mask)
            bboxes = tf.boolean_mask(bboxes, mask)
        # Image data format.
        if data_format == 'NCHW':
            image = tf.transpose(image, perm=(2, 0, 1))
        return image, labels, bboxes, bbox_img