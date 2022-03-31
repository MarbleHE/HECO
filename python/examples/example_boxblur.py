from pyabc import *
import logging

p = ABCProgram(logging.DEBUG, backend=ABCBackend.MLIRText)
# p = ABCProgram(logging.DEBUG)

with ABCContext(p, logging.DEBUG):
    # You can annotate a type as secret by two ways:
    def boxblur(img: SecretIntVector) -> SecretIntVector:
        img2 = img  # Technically this is not correct in python (since it would only copy the reference)
        for x in range(8):
            for y in range(8):
                value: Secret[int] = 0
                for j in range(-1, 2):
                    for i in range(-1, 2):
                        value += img[((x+i)*8 + (y+j)) % 64]
                img2[8*x + y] = value
        return img2

if __name__ == "__main__":
    # TODO: Printing MLIR for the moment, remove when we actually execute it.
    with open("example_boxblur.mlir", "w") as f:
        p.dump()
        p.dump(f)
