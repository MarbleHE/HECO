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

```
heco --fhe2eva --cannonicalize < <path to the input file>
```

This pass translates a program in FHE dialect to the EVA dialect.
Unfortunetately, so far not all valid FHE programs are supported, yet.
Here is the list of limitations:
* All ciphertext values in the program have to be vectors of the same length equal to a power of two. As a result, combine, insert and extract operations are not supported.
* Multiplication and addition must have exactly two operands.

### Waterline Rescale (evawaterline)
```
heco --evawaterline="source_scale=<source scale> scale_drop=<scale drop> waterline=<waterline>" < <path to the input file>
```

This pass inserts rescale operations to limit the maximal scale of the program.
In the resulting program, a rescale operation will be applied to each partial result whose scale is above the waterline.
This way we guarantee that "meaningful", i.e. non rescale, operations only operate on values with the scale at most the configured waterline.

This pass takes three optional arguments (when the argument is missing, pass takes a default value):
* source scale -- the scale of the source nodes (encrypted data taken by program as arguments).
* scale dtop -- the number by which the scale is divided by the rescale operation.
* waterline -- the threshold above which the rescale operations are inserted.

For convenience, we assume all of the above and the scale to be powers of two and we represent them as the exponents.
Thus, `source_scale=20` means that all arguments have the scale of $2^{20}$ and `scale_drop=20` means that the scale becomes $2^{20}$ times smaller after a rescale operation.

### Match Scale (evamatchscale)

```
heco --evamatchscale="source_scale=<source scale> scale_drop=<scale drop> waterline=<waterline>" < <path to the input file>
```

This pass ensures that operands of addition and subtraction operations have equal scales.
It's achieved by multiplying the one with smaller scale by a vector of ones whose encryption has the apprioriate scale.

This pass takes the same arguments as evawaterline (source scale, scale drop, waterline), which are used to calculate the
scales of partial results in the program.


### Lazy Modswitch Insertion (evalazymodswitch)
```
heco --evamodswitch="source_modulo=<source modulo>" < <path to the input file>
```

This pass ensures that operands of multiplication, addition and subtraction operations have the same moduli.
This is achieved by applying a modswitch operation to the operand with larger modulo.

Evalazymodswitch and one optional argument (if it's not given, the pass assumes a default value):
* source modulo -- the number $l$ of primes forming $Q = q_1 * q_2 * ... * q_l$ that it used to encrypt the arguments of the program.

### Relinearize (evarelinearize)
```
heco --evarelinearize < <path to the input file>
```
This pass inserts a relinearize operation after every multiplication.
This is to make sure that the program only operates on polynomials with degree one in order to avoid overhead from processing higher degree polynomials.

### Mark Metadata (evametadata)
```
heco --evametadata="source_modulo=<source modulo> source_scale=<source scale> scale_drop=<scale_drop> waterline=<waterline>" < <path to the input file>
```
This pass extends all EVA operations with attributes informing about the scale and modulo of the result.
The information is given in the same format as the arguments (scale is given as the exponent of the power of two and modulo as the number of primes in the product).
This pass takes four optional arguments with the same semantics as those have in the above passes:
* source modulo
* source scale
* scale drop
* waterline
