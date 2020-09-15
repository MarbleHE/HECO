/* supported operators

 * binary:
 * - arithmetic addition
 * - arithmetic subtraction
 * - arithmetic multiplication
 * - arithmetic division
 * - arithmetic modulo?
 *
 * - logical AND
 * - logical OR
 * - logical XOR
 * - logical NOT
 *
 * - left-shift
 * - right-shift

 * comparison:
 * - equals
 * - not equals
 * - greater-than
 * - less-than
 * - greater-equal
 * - less-equal
 *
 * unary:
 * - negation
 * - increment (++)
 * - decrement (--)
*/

/* supported types:
 * - reserved tokens  (e.g., true, false)
 * - identifiers      (e.g., int identifier;)
 * - numbers          (integer, floating-point numbers)
 * - strings
 */

/* supported keywords:
 * - if
 * - else
 * - elif       (so we do not need support for single statement-blocks as "else if" is: else { if (..) {...}})
 * - for        (only the traditional form with 3 args?)
 * - while, do  (would suggest to not support them as can also be expressed using "if")
 * - break      (cannot be expressed yet in our AST but makes sense to add)
 * - continue   (cannot be expressed yet in our AST but makes sense to add)
 * - return
 * - void
 * - switch/case/default (would suggest to not support them as can also be expressed using "if")
 */

int computePrivate(int inputA, int inputB, int inputC) {
  // NOTE: Operator precedence must be handled properly!
  int prod = inputA + inputB*inputC;
  return prod/3;
}

float computeDiscountOnServer(secret_bool qualifiesForSpecialDiscount) {
  secret_float discountRate = 0;
  { // Block does not make any sense but should be valid
    discountRate = qualifiesForSpecialDiscount*0.90 + (1 - qualifiesForSpecialDiscount)*0.98;
  }
  return discountRate;
}

int sumNTimes2(int inputA) {
  int sum = 0;
  int base = 2;
  for (int i = 0; i <= inputA; i = i + 1) {
    sum = sum + (base*i);
  }
  return sum;
}

void compute(encrypted_bool encryptedA) {
  plaintext_bool alpha = (true ^ false) || encryptedA;
}

int extractArbitraryMatrixElements() {
  int M[1][3] = {{14, 27, 32}};
  int N[1][3] = {{19, 21, 38}};
  return M[0][1] + N[0][1];
}

int[] extractArbitraryMatrixElements() {
  // providing size for 1-dimensional arrays is optional
  int M[] = {14, 27, 32};   // we should be able to easily infer the size, or will we use a dynamically-sized container?
  int B[] = {19, 21, 38};
  int result[] = {M[0], B[0], M[1], B[1]};
  return result;  // return array here
}

int[] computeCrossProduct() {
   int M[1][3] = {{14, 27, 32}};
   int B[1][3] = {{19, 21, 38}};
   int result[3] = { M[0][1]*B[0][2] - M[0][2]*B[0][1],
            M[0][2]*B[0][0] - M[0][0]*B[0][2],
            M[0][0]*B[0][1] - M[0][1]*B[0][0] };
   return result;
 }

int simpleIfConditionalAssignment(encrypted_int cond) {
  int a = 1;
  if (cond > 11) {
    a = 83;
  } else {
    a = 11;
  }
  return a;
}

 int sumVectorElements(int numIterations) {
    int M[] = {54, 32, 63, 38, 13, 20};
    int sum = 0;
    for (int i = 0; i < numIterations; i++) {
      sum = sum + M[i];
    }
    return sum;
 }

int[] runLaplacianSharpeningAlgorithm(int[] img, int imgSize, int x, int y) {
   int weightMatrix[3][3] = {{1, 1, 1}, {1, -8, 1}, {1, 1, 1}};
   int img2[];
   int value = 0;
   for (int j = -1; j < 2; ++j) {
      for (int i = -1; i < 2; ++i) {
         value = value + weightMatrix[i+1][j+1] * img[imgSize*(x+i)+y+j];
      }
   }
   img2[imgSize*x+y] = img[imgSize*x+y] - (value >> 1);
   return img2;
}

 int[] permuteMatrixElements() {
    int M[] = {14, 27, 32 };
    M[0] = 11;
    return M;
 }

 int[][] extendMatrixAddingElements() {
   int m[][];   // size: 0x0
   for (int i = 0; i < 3; ++i) {
     int t[];
     for (int j = 0; j < 3; ++j) {
       t[0][t.dimSize(1)] = i*j;
     }
     m[m.dimSize(0)] = t;
   }
   return m;  // m = [0*0 0*1 0*2; 1*0 1*1 1*2; 2*0 2*1 2*2] = [0 0 0; 0 1 2; 0 2 4], size: 3x3
 }

int[] runLaplacianSharpeningAlgorithm(int[] img, int imgSize) {
   int img2[imgSize*imgSize];
   int weightMatrix[3][3] = {{1, 1, 1}, {1, -8, 1}, {1, 1, 1}};
   for (int x = 1; x < imgSize - 1; ++x) {
       for (int y = 1; y < imgSize - 1; ++y) {
           int value = 0;
           for (int j = -1; j < 2; ++j) {
               for (int i = -1; i < 2; ++i) {
                   value = value + weightMatrix[i+1][j+1] * img[imgSize*(x+i)+y+j];
               }
           }
           img2[imgSize*x+y] = img[imgSize*x+y] - (value/2);

       }
   }
   return img2;
}
