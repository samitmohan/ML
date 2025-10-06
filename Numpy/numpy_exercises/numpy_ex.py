# https://github.com/rougier/numpy-100/blob/master/100_Numpy_exercises.md
# numpy 100 exercises
#### 1. Import the numpy package under the name `np` (★☆☆)
import numpy as np

#### 2. Print the numpy version and the configuration (★☆☆)
print(np.__version__)
print(np.show_config())

#### 3. Create a null vector of size 10 (★☆☆)
null_vector = np.zeros(10)
print(null_vector)

#### 4. How to find the memory size of any array (★☆☆)
memory_size = null_vector.nbytes
# print(memory_size) 
size = null_vector.size
size_per_element = null_vector.itemsize
print(size * size_per_element) # 80


#### 5. How to get the documentation of the numpy add function from the command line? (★☆☆)
print(np.info(np.add))

#### 6. Create a null vector of size 10 but the fifth value which is 1 (★☆☆)
null_vector[4] = 1
print(null_vector)
#### 7. Create a vector with values ranging from 10 to 49 (★☆☆)

vector_bw_10_50 = np.arange(10, 50)
print(vector_bw_10_50)

#### 8. Reverse a vector (first element becomes last) (★☆☆)
print(vector_bw_10_50[::-1])
print(np.flip(vector_bw_10_50))


#### 9. Create a 3x3 matrix with values ranging from 0 to 8 (★☆☆)
matrix_3x3 = np.arange(0, 9).reshape(3, 3)
print(f"3x3 matrix:\n {matrix_3x3}")

#### 10. Find indices of non-zero elements from [1,2,0,0,4,0] (★☆☆)
arr = np.array([1,2,0,0,4,0])
non_zero_indices = np.nonzero(arr)
print(non_zero_indices)

#### 11. Create a 3x3 identity matrix (★☆☆)
id_matrix = np.eye(3)
print(id_matrix)

#### 12. Create a 3x3x3 array with random values (★☆☆)
random_array = np.random.rand(3, 3, 3)
print(random_array)

#### 13. Create a 10x10 array with random values and find the minimum and maximum values (★☆☆)
random_arr_10x10 = np.random.rand(10, 10)
print(random_arr_10x10.min())
print(random_arr_10x10.max())

#### 14. Create a random vector of size 30 and find the mean value (★☆☆)
random_vector = np.random.rand(30)
print(random_vector.mean())

#### 15. Create a 2d array with 1 on the border and 0 inside (★☆☆)
arr_2d = np.ones((5, 5))
arr_2d[1:-1, 1: -1] = 0
print(arr_2d)

#### 16. How to add a border (filled with 0's) around an existing array? (★☆☆)
arr_2d = np.pad(arr_2d, pad_width=1, mode='constant', constant_values=0)
print(arr_2d)

#### 17. What is the result of the following expression? (★☆☆)   
0 * np.nan
print(np.nan == np.nan) # false
print(np.inf > np.nan) # false
print(np.nan - np.nan) # nan?
print(np.nan in set([np.nan])) # true
print(0.3 == 3 * 0.1) # false

#### 18. Create a 5x5 matrix with values 1,2,3,4 just below the diagonal (★☆☆)
arr_5x5 = np.diag(np.arange(1,5), k=-1) # k = where the elements will be placed in the diagonal
print(arr_5x5)

#### 19. Create a 8x8 matrix and fill it with a checkerboard pattern (★☆☆)
arr_8x8 = np.zeros((8, 8))
arr_8x8[1::2, ::2] = 1
arr_8x8[::2, 1::2] = 1
print(arr_8x8)

#### 20. Consider a (6,7,8) shape array, what is the index (x,y,z) of the 100th element? (★☆☆)

print("\n Unravelling Indexes")
print(np.unravel_index(100,(6,7,8))) # 1, 5, 4

#### 21. Create a checkerboard 8x8 matrix using the tile function (★☆☆)
arr_8x8_tile = np.tile(np.array([[0, 1], [1, 0]]), (4, 4)) # cool
print(arr_8x8_tile)

#### 22. Normalize a 5x5 random matrix (★☆☆)
z = np.random.rand(5, 5)
normalised = z - np.mean(z) / np.std(z)
print(normalised)

#### 23. Create a custom dtype that describes a color as four unsigned bytes (RGBA) (★☆☆)
color = np.dtype([("r", np.ubyte, 1), ("g", np.ubyte, 1), ("b", np.ubyte,1 ), ("a", np.ubyte, 1)])
print(color)

#### 24. Multiply a 5x3 matrix by a 3x2 matrix (real matrix product) (★☆☆)
z = np.dot(np.ones((5,3)), np.ones((3,2))) # 5 * 2 matrix full of 3s
print(z)

#### 25. Given a 1D array, negate all elements which are between 3 and 8, in place. (★☆☆)
z = np.arange(11)
z[(3 < z) & (z < 8)] *= -1
print(z)


#### 26. What is the output of the following script? (★☆☆)
print(sum(range(5),-1))
print(sum(range(5),-1)) 



#### 27. Consider an integer vector Z, which of these expressions are legal? (★☆☆)

# Z**Z
# 2 << Z >> 2
# Z <- Z
# 1j*Z
# Z/1/1
# Z<Z>Z


#### 28. What are the result of the following expressions? (★☆☆)
print(np.array(0) / np.array(0)) # nan
print(np.array(0) // np.array(0)) # 0
print(np.array([np.nan]).astype(int).astype(float)) # [0.]


#### 29. How to round away from zero a float array ? (★☆☆)
z = np.random.uniform(-10,+10,10)
print(np.copysign(np.ceil(np.abs(z)), z))


##### 30. How to find common values between two arrays? (★☆☆)
z1 = np.random.randint(0,10,10)
z2 = np.random.randint(0,10,10)
print(np.intersect1d(z1,z2))


#### 31. How to ignore all numpy warnings (not recommended)? (★☆☆)
defaults = np.seterr(all="ignore")
z = np.ones(1) / 0
_ = np.seterr(**defaults)


#### 32. Is the following expressions true? (★☆☆)
np.sqrt(-1) == np.emath.sqrt(-1) # invalid


#### 33. How to get the dates of yesterday, today and tomorrow? (★☆☆)
yesterday = np.datetime64('today', 'D') - np.timedelta64(1, 'D')
today     = np.datetime64('today', 'D')
tomorrow  = np.datetime64('today', 'D') + np.timedelta64(1, 'D')
print(yesterday, today, tomorrow)


#### 34. How to get all the dates corresponding to the month of July 2016? (★★☆)

july_2016 = np.arange('2016-07', '2016-08', dtype='datetime64[D]')
print(july_2016)

#### 35. How to compute ((A+B)*(-A/2)) in place (without copy)? (★★☆)

A = np.ones(3) * 1
B = np.ones(3) * 2
C = np.ones(3) * 3
np.add(A, B, out=B)
np.divide(A, 2, out=A)
np.negative(A, out=A)
np.multiply(A, B, out=A)
print(A)


#### 36. Extract the integer part of a random array using 5 different methods (★★☆)¶
z = np.random.uniform(0, 10, 10)
print(np.floor(z))
print(np.ceil(z) - 1)
print(z.astype(int))
print(np.trunc(z))
print(z - z % 1)


#### 37. Create a 5x5 matrix with row values ranging from 0 to 4 (★★☆)¶
z = np.zeros((5, 5))
z += np.arange(5) # so cool
print(z)


#### 38. Consider a generator function that generates 10 integers and use it to build an array (★☆☆)¶
def generate():
    for x in range(10):
        yield x
z = np.fromiter(generate(), dtype=float, count=-1)
print(z)

#### 39. Create a vector of size 10 with values ranging from 0 to 1, both excluded (★★☆)¶
z = np.linspace(0, 1, 11, endpoint=False)[1:]
print(z)

#### 40. Create a random vector of size 10 and sort it (★★☆)¶
z = np.random.random(10)
z.sort()
print(z)

#### 41. How to sum a small array faster than np.sum? (★★☆)¶
z = np.arange(10)
np.add.reduce(z)

#### 42. Consider two random array A and B, check if they are equal (★★☆)¶
A = np.random.randint(0, 2, 5)
B = np.random.randint(0, 2, 5)
equal = np.allclose(A, B) # values
equal_both_shape_and_value = np.array_equal(A, B) # shape and values


#### 43. Make an array immutable (read-only) (★★☆)¶
immutable_arr = np.zeros(10)
immutable_arr[0] = 1 
print(immutable_arr)

immutable_arr.flags.writeable = False
# immutable_arr[0] = 2 # fails 


#### 44. Consider a random 10x2 matrix representing cartesian coordinates, convert them to polar coordinates (★★☆)¶
z = np.random.random((10, 2))
x, y = z[:, 0], z[:, 1]
r = np.sqrt(x**2 + y**2)
t = np.arctan2(y, x)
print(f"Polar coordinates, \n r : {r}, \n t: {t}")

#### 45. Create random vector of size 10 and replace the maximum value by 0 (★★☆)¶
z = np.random.random(10)
z[z.argmax()] = 0
print(z)

#### 46. Create a structured array with `x` and `y` coordinates covering the [0,1]x[0,1] area (★★☆)

str_arr = np.zeros((5,5), [('x', float), ('y', float)])
str_arr['x'], str_arr['y'] = np.meshgrid(np.linspace(0, 1, 5), np.linspace(0, 1, 5))
print(str_arr)

#### 47. Given two arrays, X and Y, construct the Cauchy matrix C (Cij =1/(xi - yj))
X = np.arange(8)
Y = X + 0.5
C = 1.0 / np.subtract.outer(X, Y)


print(f"Cauchy Matrix:  {np.linalg.det(C)}")


#### 48. Print the minimum and maximum representable value for each numpy scalar type (★★☆)¶
for dtype in [np.int8, np.int32, np.int64]:
   print(np.iinfo(dtype).min)
   print(np.iinfo(dtype).max)   

for dtype in [np.float32, np.float64]:
    print(np.finfo(dtype).min)
    print(np.finfo(dtype).max)


#### 49. How to print all the values of an array? (★★☆)¶
# np.set_printoptions(threshold=np.nan)
z = np.zeros((16, 16))
print(z)



#### 50. How to find the closest value (to a given scalar) in a vector? (★★☆)¶
z = np.arange(100) # 1-100 arr
v = np.random.uniform(0, 100) # random val
print(v)
index = (np.abs(z - v)).argmin()
print(z[index])

#### 51. Create a structured array representing a position (x,y) and a color (r,g,b) (★★☆)
arr = np.zeros(10, [('position', [('x', float, 1), ('y', float, 1)]), ('color', [('r', float, 1), ('g', float, 1), ('b', float, 1)])])
# print(arr)


#### 52. Consider a random vector with shape (100,2) representing coordinates, find point by point distances (★★☆)¶
random_vec = np.random.random((100, 2))
x, y = np.atleast_2d(random_vec[:, 0], random_vec[:, 1])

dist = np.sqrt((x-x.T)**2 + (y-y.T)**2)
print(f"Distance = {dist}")

# much faster with scipy
import scipy
import scipy.spatial

Z = np.random.random((10,2))
D = scipy.spatial.distance.cdist(Z,Z)



#### 53. How to convert a float (32 bits) array into an integer (32 bits) in place?
z = np.arange(10, dtype=np.float32)
z = z.astype(np.int32, copy=False)
print(z)


#### 54. How to read the following file? (★★☆)¶


from io import StringIO

# Fake file 
s = StringIO("""1, 2, 3, 4, 5\n
                6,  ,  , 7, 8\n
                 ,  , 9,10,11\n""")
Z = np.genfromtxt(s, delimiter=",", dtype=np.int16) # empty spaces get -1
print(Z)


#### 55. What is the equivalent of enumerate for numpy arrays? (★★☆)
z = np.arange(9).reshape(3, 3)
for index, value in np.ndenumerate(z):
    print(index, value)

for index in np.ndindex(z.shape):
    print(index, z[index])

#### 56. Generate a generic 2D Gaussian-like array (★★☆)¶




















































