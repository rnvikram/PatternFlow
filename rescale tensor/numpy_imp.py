import tensorflow as tf
import numpy as np



#Building the dictionary of 
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




def rescale_tf(input_image,in_range='image', out_range='dtype'):
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



def intensity_range(image,dtype, range_values='image', clip_negative=False):
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