import numpy as np

one_d = np.array(([1, 5, 6, 8, 9, 10]))
ten_zeroed = np.zeros(10)
fifteen_ones = np.ones(15)
sequence = np.arange(16)
stepper = np.arange(0, 51, 5)
spaced = np.linspace(0, 15, num=5)

shaped = one_d.reshape(3, 2)
changed = np.flip(shaped)

add = shaped + changed

back_to_one = add.flatten()

diff_type = back_to_one.astype('f')

print(diff_type)

# ndm, size, shape, dtype, copy, sort