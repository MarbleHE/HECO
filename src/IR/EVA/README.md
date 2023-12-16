# EVA: An Encrypted Vector Arithmetic Language

This is an implementation of EVA, a language for FHE proposed by Bathathri et al. ([EVA: An Encrypted Vector Arithmetic Language and Compiler for Efficient Homomorphic Computation](https://arxiv.org/abs/1912.11951)). Here are the operations included in the language:

* NEGATE
* ADD
* SUB
* MULTIPLY
* ROTATE_LEFT
* ROTATE_RIGHT

All of the above operations operate on vectors, which need to have identical length across the whole program.
An important limitation is that there are no operations which
directly allow for combining values into a vector or retrieving scalars from the vectors.
This, however, can be achirved using rotations, multiplications by a constant and additions.

Furthermore, it supports a few ciphertext maintenance operations.
Ciphertexts under consideration are polynomials modulo $Q$, which is a product of primes ($Q = q_1 * q_2 * ... * q_l$).
The encrypted value is kept as in a fixed point format: as an integer that has to be divided by a certain
factor (called scale) to achieve the actual value.

Because the range fpr integers in bounded, the scale needs to be kept under a certain limit.
Also, there are certain requirements on the modulo and the scale given by the operations:

* ADD and SUB -- both operands need to have the same scale and modulo.
* MULTIPLY -- both operands need to have the same modulo.
The scale of the result is the product of the operands' scales.

These are the supported ciphertext maintenance operations:
* RELINEARIZE -- decreases the degree of the ciphertexts (which are polynomials) to 1.
* MODSWITCH -- shortens the modulo chain by one (removes the last prime from the product $q_1 * q_2 * ... q_{l'}$).
* RESCALE -- decreases the scale of the ciphertext while also shortening the modulo chain by one.
* CONSTANT -- takes an value known at compile-time and ecrypts it using a given modulo and scale.

## Passes

We also implement passes proposed by Bathathri et al. that insert ciphertext maintenance operations.
This produces a program satisfying the aforementioned requirements on ciphertexts give by the operations
while keeping the scales below a certain (configurable) level and without sactificing too much performance.

Here follows the list of the passes in the recommended order of application:

### Lowering FHE to EVA (fhe2eva)


### Waterline Rescale (evawaterline)


### Match Scale (evamatchscale)


### Lazy Modswitch Insertion (evalazymodswitch)


### Relinearize (evarelinearize)


### Mark Metadata (evametadata)

