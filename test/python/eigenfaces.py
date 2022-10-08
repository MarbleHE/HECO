import random
from heco.frontend import *

sf64 = Secret[f64]

p = FrontendProgram()
with CodeContext(p):
    def encryptedEigenfaces(x: Tensor[128, sf64],
                            d: Tensor[128, 128, sf64]) -> sf64:
        r: Tensor[128, sf64] = [0] * 128
        for i in range(0, 128):
            sum: sf64 = 0
            for j in range(0, 128):
                sum = sum + x[j]*x[j] - d[i][j] * d[i][j]
            r[i] = sum
        return r

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
