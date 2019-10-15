integer_ranges={
 "<dtype: 'float16'>": (-1, 1),
 "<dtype: 'float32'>": (-3.4028235e+38, 3.4028235e+38),
 "<dtype: 'float64'>": (-1.7976931348623157e+308, 1.7976931348623157e+308),
 "<dtype: 'int16'>": (-32768, 32767),
 "<dtype: 'int32'>": (-2147483648, 2147483647),
 "<dtype: 'int64'>": (-9223372036854775808, 9223372036854775807),
 "<dtype: 'int8'>": (-128, 127),
 "<dtype: 'uint16'>": (0, 65535),
 "<dtype: 'uint32'>": (0, 4294967295),
 "<dtype: 'uint64'>": (0, 18446744073709551615),
 "<dtype: 'uint8'>": (0, 255)}

DTYPE_RANGE = integer_ranges.copy()

import tensorflow as tf

def rescale_tf(input_image,in_range='image', out_range='dtype'):
  dtype = input_image.dtype
  imin, imax = intensity_range(input_image, in_range)
  omin, omax = intensity_range(input_image,out_range,clip_negative=(imin >= 0))
  print("Calculated wit new omax, omin")
  image=tf.clip_by_value(
    input_image,
    imin,
    imax,
    name=None
)
  print("Clip is done")
  print(tf.math.subtract(imax ,imin))
  image = tf.math.divide(tf.math.subtract(image ,imin), tf.math.subtract(imax ,imin))
  return (image * (omax - omin) + omin)


def intensity_range(image, range_values='image', clip_negative=False):
    if range_values == 'dtype':
        range_values = image.dtype
    if str(range_values) == 'image':
        i_min = tf.reduce_min(image)
        i_max = tf.reduce_max(image)

    elif str(range_values).strip("") in DTYPE_RANGE:
        i_min, i_max = DTYPE_RANGE[str(range_values).strip("")]
        if clip_negative==True:
            i_min = 0
    else:
        i_min, i_max = range_values
    return i_min, i_max



