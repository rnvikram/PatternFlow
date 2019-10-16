## Rescale Intensity

The function takes an array as input and rescales the values in it based on the parameters passed to it. The options avaiable for the input range and out range are the same.

Tuple with min and max: It rescales the values in the array using that range
"image": Uses the input's max and min to rescale the values in the array
"dtype": Uses the input's dtyoe to fix max and min values which is then used to rescale the values in the array
dtype-name: Uses the min and max of the dtype specified. 


* numpy.bool_: (False, True),
* numpy.float16: (-1, 1),
* numpy.float32: (-1, 1),
* numpy.float64: (-1, 1),
* numpy.int16: (-32768, 32767),
* numpy.int32: (-2147483648, 2147483647),
* numpy.int64: (-9223372036854775808, 9223372036854775807),
* numpy.int64: (-9223372036854775808, 9223372036854775807),
* numpy.int8: (-128, 127),
* numpy.uint16: (0, 65535),
* numpy.uint32: (0, 4294967295),
* numpy.uint64: (0, 18446744073709551615),
* numpy.uint64: (0, 18446744073709551615),
* numpy.uint8: (0, 255),
* 'bool': (False, True),
* 'bool_': (False, True),
* 'float': (-1, 1),
* 'float16': (-1, 1),
* 'float32': (-1, 1),
* 'float64': (-1, 1),
* 'int16': (-32768, 32767),
* 'int32': (-2147483648, 2147483647),
* 'int64': (-9223372036854775808, 9223372036854775807),
* 'int8': (-128, 127),
* 'uint10': (0, 1023),
* 'uint12': (0, 4095),
* 'uint14': (0, 16383),
* 'uint16': (0, 65535),
* 'uint32': (0, 4294967295),
* 'uint64': (0, 18446744073709551615),
* 'uint8': (0, 255)
