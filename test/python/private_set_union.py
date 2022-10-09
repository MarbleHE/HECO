import random
from heco.frontend import *

sf64 = Secret[f64]
sbool = Secret[bool]

# p = FrontendProgram()
# with CodeContext(p):
#     def encryptedPSU(a_id: Tensor[128, 8, sbool], a_data: Tensor[128, sf64],
#                      b_id: Tensor[128, 8, sbool], b_data: Tensor[128, sf64]) -> sf64:
#         sum: sf64 = 0
#         for i in range(0, 128):
#             sum = sum + a_data[i]
#
#         for i in range(0, 128):
#             unique: sbool = 1
#             for j in range(0, 128):
#                 # compute a_id[i]== b_id[i]
#                 equal: sbool = 1
#                 for k in range(0, 8):
#                     equal = equal * (not (a_id[i][k] ^ b_id[j][k]))
#                 unique = unique and (not equal)
#
#         sum=sum + unique * a_data[i]
#
#         return sum


p = FrontendProgram()
with CodeContext(p):
    def encryptedPSU(a_id: Tensor[128, 8, sf64], a_data: Tensor[128, sf64],
                     b_id: Tensor[128, 8, sf64], b_data: Tensor[128, sf64]) -> sf64:
        sum: sf64 = 0
        for i in range(0, 128):
            sum = sum + a_data[i]

        for i in range(0, 128):
            unique: sf64 = 1
            for j in range(0, 128):
                # compute a_id[i]== b_id[j]
                equal: sf64 = 1
                for k in range(0, 8):
                    # a xor b == (a-b)^2
                    x = (a_id[i][k] - b_id[j][k])**2
                    # not x == 1 - x
                    nx = 1 - x
                    equal = equal * nx
                nequal = 1 - equal
                unique = unique * nequal

        sum = sum + unique * a_data[i]

        return sum

# Compiling FHE code
context = SEAL.BGV.new(poly_mod_degree=2048)
f = p.compile(context=context)

# Running FHE code
x = [random.randrange(100) for _ in range(128)]
d = [[random.randrange(100) for _ in range(128)] for _ in range(128)]
x_enc = context.encrypt(x)
d_enc = context.encrypt(d)
r_enc = f(x_enc, d_enc)

# Decrypting Result
s = context.decrypt(r_enc)
