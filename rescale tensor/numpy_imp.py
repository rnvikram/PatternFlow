import tensorflow as tf
import numpy as np



#Building the dictionary of supported data types and the range they supported by each type
_integer_types = (np.byte, np.ubyte,
                  np.short, np.ushort,
                  np.intc, np.uintc,
                  np.int_, np.uint,
                  np.longlong, np.ulonglong)
_integer_ranges = {t: (np.iinfo(t).min, np.iinfo(t).max)
                   for t in _integer_types}
dtype_range = {np.bool_: (False, True),
               np.bool8: (False, True),
               np.float16: (-1, 1),
               np.float32: (-1, 1),
               np.float64: (-1, 1)}
dtype_range.update(_integer_ranges)

DTYPE_RANGE = dtype_range.copy()
DTYPE_RANGE.update((d.__name__, limits) for d, limits in dtype_range.items())
DTYPE_RANGE.update({'uint10': (0, 2 ** 10 - 1),
                    'uint12': (0, 2 ** 12 - 1),
                    'uint14': (0, 2 ** 14 - 1),
                    'bool': dtype_range[np.bool_],
                    'float': dtype_range[np.float64]})


def intensity_range(image,dtype, range_values='image', clip_negative=False):
    """

    :param image(np array): Input array which is used to pull the min and max value if needed
    :param dtype(str or dtype): Data type of the input array
    :param range_values: Range value identifier to select the min and max value
                         "image"-> The min and max of the image is chosen
                         "dtype"-> The min and max supported by dtype is chosen
                         tuple-> Used as the range
    :param clip_negative(bool): identifier to check if the vlue have to cliped to 0

    :return:
    i_min and i_max which is the range based on the range_values given

    example:
    intensity_range([0,5,10],float,range_values='image',clip_negative=False)  -> 0,10
    intensity_range([0,5,10],float,range_values='dtype',clip_negative=False)  -> 0,1
    intensity_range([-5,5,10],float,range_values='image',clip_negative=False) -> -5,10
    intensity_range([-5,5,10],float,range_values='image',clip_negative=True)  -> 0,10

    """


    if range_values == 'dtype':
        range_values = dtype
    if str(range_values) == 'image':
        i_min = tf.reduce_min(image).eval()
        i_max = tf.reduce_max(image).eval()
    elif  str(range_values) in DTYPE_RANGE:
        i_min, i_max = DTYPE_RANGE[str(range_values)]
        if clip_negative==True:
            i_min = 0
    elif  range_values in DTYPE_RANGE:
        i_min, i_max = DTYPE_RANGE[range_values]
        if clip_negative==True:
            i_min = 0
    else:
        i_min, i_max = range_values
    return i_min, i_max



def rescale_tf(input_image,in_range='image', out_range='dtype'):
  """

  :param input_image(np_array): Input array to be rescaled
  :param in_range:  "image"-> If the range is to be chosen from the image
                    "dtype"-> If the range of the dtype needs to be used
                    tuple-> User defined range
  :param out_range: "image"-> If the range is to be chosen from the image
                    "dtype"-> If the range of the dtype needs to be used
                    tuple-> User defined range
  :return:
  numpy array with values rescaled

  Examples:
    rescale_intensity([1,2,3,4],out_range=(0,1)) --> [0., 0.33333334, 0.66666669, 1.]
    rescale_intensity(np.array([51, 102, 153], dtype=np.uint8)) --> [0, 127, 255]
    rescale_intensity(np.array([51, 102, 153], dtype=np.float16)) --> [0, 0.5, 1]
    rescale_intensity(np.array([51, 102, 153], dtype=np.float16),in_range=(0,102)) --> [0.5, 1., 1.]
    rescale_intensity(np.array([51, 102, 153], dtype=np.unit8),in_range=(0,102)) --> [127, 255, 255]
    rescale_intensity(np.array([51, 102, 153], dtype=np.unit8),out_range=(0,102)) --> [0, 51, 102]
  """
  dtype = input_image.dtype
  if in_range == "image" or in_range == 'dtype' or (len(in_range) == 2 and type(in_range) == tuple):
      pass
  elif in_range in DTYPE_RANGE:
      pass
  else:
      raise ValueError('Unsupported input to in_range', in_range)

  if out_range == "image" or out_range == 'dtype' or (len(out_range) == 2 and type(out_range) == tuple):
      pass
  elif out_range in DTYPE_RANGE:
      pass
  else:
      raise ValueError('Unsupported input to out_range', out_range)
  input_image = tf.constant(input_image)
  sess = tf.InteractiveSession()
  tf.global_variables_initializer().run()
  imin, imax = intensity_range(input_image,dtype, in_range)
  omin, omax = intensity_range(input_image,dtype,out_range,clip_negative=(imin >= 0))
  input_image = tf.dtypes.cast(input_image,"float",name=None)
  image=tf.clip_by_value(input_image,imin,imax,name=None)
  if imin!=imax:
    image=(image-imin)/float(imax-imin)
  output=(image * (omax - omin) + omin).eval()
  output=np.array(output,dtype=dtype)
  InteractiveSession.close()
  sess.close()
  return output



