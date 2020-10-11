#include "ast_opt/ast/AbstractNode.h"
#include "ast_opt/visitor/Vectorizer.h"
#include "ast_opt/parser/Parser.h"
#include "../ASTComparison.h"
#include "gtest/gtest.h"

TEST(VectorizerTest, trivialVectors) {

  const char *inputChars = R""""(
    x[0] = y[0];
    x[1] = y[1];
    x[2] = y[2];
    x[3] = y[3];
    x[4] = y[4];
    x[5] = y[5];
    x[6] = y[6];
    x[7] = y[7];
    x[8] = y[8];
    x[9] = y[9];
    )"""";
  auto inputCode = std::string(inputChars);
  auto inputAST = Parser::parse(inputCode);

  Vectorizer v;
  v.setRootScope(std::make_unique<Scope>(*inputAST));
  v.getRootScope().addIdentifier("x");
  v.getRootScope().addIdentifier("y");
  inputAST->accept(v);

  const char *expectedChars = R""""(
    x = y;
    )"""";
  auto expectedCode = std::string(expectedChars);
  auto expectedAST = Parser::parse(expectedCode);

  EXPECT_TRUE(compareAST(*inputAST, *expectedAST));
}

////
// CV (x[0]):
/// Computation Plan: __x__ = y;
/// target_slot = 0
// CV (x[1]):
/// Computation Plan: __x__ = y;
/// target_slot = 1
/// ditto  for a...
/// Ok, emit x:
// iterate through all computation plans for x[0] to x[n]. If there is none, assume x[i] = x_original[i];
// Check if they're compatible, if yes, output only a single one.
// Downside: Linear in number of statements
// Better: Have CV's for entire vectors and work on them as you go

// CV (x):
// slots: 0
/// Computation Plan: x = y;
// slots: 1,2,3....
// Computation Plan: x = old_x (or rather, ref/ptr to the old expression);
// CV (x):
// slots: 0,1
/// Computation Plan: x = y;
// slots: 2,3....
// Computation Plan: x = old_x (or rather, ref/ptr to the old expression);

TEST(VectorizerTest, trivialInterleavedVectors) {

  const char *inputChars = R""""(
    x[0] = y[0];
    a[0] = b[0];
    x[1] = y[1];
    a[1] = b[1];
    x[2] = y[2];
    a[2] = b[2];
    x[3] = y[3];
    a[3] = b[3];
    )"""";
  auto inputCode = std::string(inputChars);
  auto inputAST = Parser::parse(inputCode);

  Vectorizer v;
  v.setRootScope(std::make_unique<Scope>(*inputAST));
  v.getRootScope().addIdentifier("x");
  v.getRootScope().addIdentifier("y");
  v.getRootScope().addIdentifier("a");
  v.getRootScope().addIdentifier("b");
  inputAST->accept(v);

  const char *expectedChars = R""""(
    x = y;
    )"""";
  auto expectedCode = std::string(expectedChars);
  auto expectedAST = Parser::parse(expectedCode);

  EXPECT_TRUE(compareAST(*inputAST, *expectedAST));
}

TEST(VectorizerTest, singleOutlierVector) {
  const char *inputChars = R""""(
    x[0] = y[0];
    x[1] = y[1];
    x[2] = y[2];
    x[3] = y[3];
    x[4] = y[4];
    x[5] = y[5];
    x[6] = y[6];
    x[7] = y[7];
    x[8] = y[8];
    x[9] = 5;
    )"""";
  auto inputCode = std::string(inputChars);
  auto inputAST = Parser::parse(inputCode);

  Vectorizer v;
  v.setRootScope(std::make_unique<Scope>(*inputAST));
  v.getRootScope().addIdentifier("x");
  v.getRootScope().addIdentifier("y");
  inputAST->accept(v);

  const char *expectedChars = R""""(
    x = y;
    x = x *** {1,1,1,1,1,1,1,1,0};
    x = x +++ {0,0,0,0,0,0,0,0,5};
    )"""";
  auto expectedCode = std::string(expectedChars);
  auto expectedAST = Parser::parse(expectedCode);

  EXPECT_TRUE(compareAST(*inputAST, *expectedAST));
}


// CV (sum):
// Computation Plan: old-sum + x[0]
//..next statement
// assuming we recognize in-place updates:
// Computation Plan: old-sum + x[0] + x[1]
//...
// CV (sum):
// Computation Plan: old-sum + x[0] + ... + x[7]; (Operator Expression)
// If trying to update with non-transparent operation (for now, any different one) causes emit of sum!
// At the end: Emit OperatorExpression, which introduces rotations
TEST(VectorizerTest, sumStatementsPowerOfTwo) {

  //If sum is vector valued, this would mean something very different
  // Specifically, it would mean that in each step, x[i] is added to each slot.
  // TODO: Therefore, we need to pass in some additional information to the Vectorizer
  // i.e. that sum has a scalar datatype.
  //TODO: We *do* need to differentiate between vector and scalar types in our input language and during compilation!
  const char *inputChars = R""""(
    sum = sum + x[0];
    sum = sum + x[1];
    sum = sum + x[2];
    sum = sum + x[3];
    sum = sum + x[4];
    sum = sum + x[5];
    sum = sum + x[6];
    sum = sum + x[7];
    )"""";
  auto inputCode = std::string(inputChars);
  auto inputAST = Parser::parse(inputCode);

  Vectorizer v;
  v.setRootScope(std::make_unique<Scope>(*inputAST));
  v.getRootScope().addIdentifier("sum");
  v.getRootScope().addIdentifier("x");
  inputAST->accept(v);

  // We communicate the slot of the result to the runtime system using the auxiliary information file
  // I.e. we define which slot of which variable to output by just supplying a line like
  // __output__ = sum[0];
  const char *expectedChars = R""""(
    sum = x + rotate(x, 4);
    sum = sum + rotate(sum, 2);
    sum = sum + rotate(sum, 1);
    )"""";
  auto expectedCode = std::string(expectedChars);
  auto expectedAST = Parser::parse(expectedCode);

  EXPECT_TRUE(compareAST(*inputAST, *expectedAST));
}

TEST(VectorizerTest, sumStatementsGeneral) {

  const char *inputChars = R""""(
    sum = sum + x[0];
    sum = sum + x[1];
    sum = sum + x[2];
    sum = sum + x[3];
    sum = sum + x[4];
    sum = sum + x[5];
    sum = sum + x[6];
    sum = sum + x[7];
    sum = sum + x[8];
    sum = sum + x[9];
    )"""";
  auto inputCode = std::string(inputChars);
  auto inputAST = Parser::parse(inputCode);

  Vectorizer v;
  v.setRootScope(std::make_unique<Scope>(*inputAST));
  v.getRootScope().addIdentifier("sum");
  v.getRootScope().addIdentifier("x");
  inputAST->accept(v);

  // First extend the vector to the next power of two (and mask away any potential garbage)
  // TODO: Is there a away to avoid this by keeping some additional info or having invariants/guarantees on inputs?
  // TODO: The compiler should all of this internally, everything output to the runtime system is just executed blindly
  // i.e. if masking is needed, compiler outputs masking statement, if not it's simply ommitted
  // TODO: Internally in the computation, the compiler can keep track of target slot and runtime system does not need to know
  // TODO: However, when returning and decrypting the client needs to know => i.e. generate one auxillary file that defines input encoding and output decoding!
  const char *expectedChars = R""""(
    sum = sum + rotate(sum, 8);
    sum = sum + rotate(sum, 4);
    sum = sum + rotate(sum, 2);
    sum = sum + rotate(sum, 1);
    )"""";
  auto expectedCode = std::string(expectedChars);
  auto expectedAST = Parser::parse(expectedCode);

  EXPECT_TRUE(compareAST(*inputAST, *expectedAST));
}

// Idea for algorithm: Go through operand-expr (lik sum) style and instead of only checking for exact match,
// do full batchability of expression logic, comparing all in set + current candidate and potentially transforming current or in set
// Main challenge: need to later output batching "map" from this. easiest if all variables are "free", i.e. not constrained.
// More difficult in general, lots of option, could also encode things twice, but now optimality no longer obvious.
TEST(VectorizerTest, cardioTest) {

  //TODO: With variable substition, this would look very different!
  //TODO: After running the If-Rewriter, we would have to run CTES across the Block again (this is an example why)
  const char *inputChars = R""""(
    risk = risk +++ (man && (age > 50));
    risk = risk +++ (woman && (age > 40));
    risk = risk +++ smoking;
    risk = risk +++ diabetic;
    risk = risk +++ high_blood_pressure;
    risk = risk +++ (cholesterol < 40);
    risk = risk +++ (weight > (height - 90));
    risk = risk +++ (daily_physical_activity < 30);
    risk = risk +++ (man && (alcohol > 3));
    risk = risk +++ (woman && (alcohol > 2));
    )"""";
  auto inputCode = std::string(inputChars);
  auto inputAST = Parser::parse(inputCode);

  Vectorizer v;
  v.setRootScope(std::make_unique<Scope>(*inputAST));
  v.getRootScope().addIdentifier("risk");
  v.getRootScope().addIdentifier("man");
  v.getRootScope().addIdentifier("age");
  v.getRootScope().addIdentifier("woman");
  v.getRootScope().addIdentifier("smoking");
  v.getRootScope().addIdentifier("diabetic");
  v.getRootScope().addIdentifier("high_blood_pressure");
  v.getRootScope().addIdentifier("cholesterol");
  v.getRootScope().addIdentifier("weight");
  v.getRootScope().addIdentifier("height");
  v.getRootScope().addIdentifier("daily_physical_activity");
  v.getRootScope().addIdentifier("alcohol");
  inputAST->accept(v);

  // First extend the vector to the next power of two (and mask away any potential garbage)
  // TODO: Is there a away to avoid this by keeping some additional info or having invariants/guarantees on inputs?
  // TODO: The compiler should all of this internally, everything output to the runtime system is just executed blindly
  // i.e. if masking is needed, compiler outputs masking statement, if not it's simply ommitted
  // TODO: Internally in the computation, the compiler can keep track of target slot and runtime system does not need to know
  // TODO: However, when returning and decrypting the client needs to know => i.e. generate one auxillary file that defines input encoding and output decoding!

  // TODO: Express input encoding properly

  // Encoding an input value int multiple slots is allowed (obviously needed for "standard" scalar encoding of {x,x,x,x,...}
  // However, in order to prevent trivial "just compute on client", only input variables and literals are allowed, no expressions
  // TODO: How is order determined?
  const char *expectedAuxillaryChars = R""""(
    __input0__ = {man, woman, smoking, diabetic, high_blood_pressure, 1,           1,             1,                       man,     woman  };
    __input1__ = {age, age,   1,       1,        1,                   cholesterol, 0,             daily_physical_activity, 3,       2      };
    __input2__ = {0,   0,     0,       0,        0,                   0,           height,        0,                       0,       0      };
    __input3__ = {50,  40,    0,       0,        0,                   40,          weight,        30,                      alcohol, alcohol};
    )"""";

  // Ideally, we'd need
  // __flags__ = {man, woman, smoking, diabetic, high_blood_pressure, 1,           1,             1,                       man,     woman  };
  // __lhs__ =   {age, age,   1,       1,        1,                   cholesterol, (height - 90), daily_physical_activity, 3,       2      };
  // __rhs__ =   {50,  40,    0 ,      0 ,       0,                   40,          weight,        30,                      alcohol, alcohol};

  // First we need to compute
  // __lhs__ =   {age, age,   1,       1,        1,                   cholesterol, (height - 90), daily_physical_activity, 3,       2      };
  // From __input1__ and __input2__:
  // __input2__ = __input2__ - {0, 0, 0, 0, 0, 0, 90, 0, 0, 0};
  // __input3__ = __input2__ + __input3__;


  // We need to differentiate between +,-,* in the plaintext space and + and * in the FHE scheme!!
  // Otherwise, the addition between input2 and input3 might be unnecessarily expensive b/c of binary emulation!
  // Alternatively, the compiler could already output the binary logic, but that feels messy
  const char *expectedChars = R""""(
    __input2__ = __input2__ - {0, 0, 0, 0, 0, 0, 90, 0, 0, 0};
    __input2__ = __input2__ +++ __input3__;
    risk = __input0__ *** (__input2__ > __input1__);
    )"""";
  auto expectedCode = std::string(expectedChars);
  auto expectedAST = Parser::parse(expectedCode);

  EXPECT_TRUE(compareAST(*inputAST, *expectedAST));
  EXPECT_EQ(v.getAuxiliaryInformation(), expectedAuxillaryChars);
}

// Simplified test case without "-90" and with same comparison operators in all conditions
TEST(VectorizerTest, cardioTestSimplified) {

  // Before Variable Substitution:
  //  risk = risk +++ (man && (age > 50));
  //  risk = risk +++ (woman && (age > 40));
  //  risk = risk +++ smoking;
  //  risk = risk +++ diabetic;
  //  risk = risk +++ high_blood_pressure;
  //  risk = risk +++ (40 > cholesterol);
  //  risk = risk +++ (weight > height);
  //  risk = risk +++ (30 > daily_physical_activity);
  //  risk = risk +++ (man && (alcohol > 3));
  //  risk = risk +++ (woman && (alcohol > 2));

  // With Variable Substitution
  // TODO: Here, variable substitution is somewhat less helpful.
  //   We need to implement expression-batching for this!
  // assuming risk = 0;
  //  risk =(man && (age > 50)) +++ (woman && (age > 40)) +++ smoking +++ diabetic
  //  +++ high_blood_pressure +++ (40 > cholesterol) +++ (weight > height) +++ (30 > daily_physical_activity)
  //  +++ (man && (alcohol > 3)) +++ (woman && (alcohol > 2));

  const char *inputChars = R""""(
   risk =(man && (age > 50)) +++ (woman && (age > 40)) +++ smoking +++ diabetic
   +++ high_blood_pressure +++ (40 > cholesterol) +++ (weight > height) +++ (30 > daily_physical_activity)
   +++ (man && (alcohol > 3)) +++ (woman && (alcohol > 2));
    )"""";
  auto inputCode = std::string(inputChars);
  auto inputAST = Parser::parse(inputCode);

  Vectorizer v;
  v.setRootScope(std::make_unique<Scope>(*inputAST));
  v.getRootScope().addIdentifier("risk");
  v.getRootScope().addIdentifier("man");
  v.getRootScope().addIdentifier("age");
  v.getRootScope().addIdentifier("woman");
  v.getRootScope().addIdentifier("smoking");
  v.getRootScope().addIdentifier("diabetic");
  v.getRootScope().addIdentifier("high_blood_pressure");
  v.getRootScope().addIdentifier("cholesterol");
  v.getRootScope().addIdentifier("weight");
  v.getRootScope().addIdentifier("height");
  v.getRootScope().addIdentifier("daily_physical_activity");
  v.getRootScope().addIdentifier("alcohol");
  inputAST->accept(v);

  const char *expectedAuxillaryChars = R""""(
    __input0__ = {man, woman, smoking, diabetic, high_blood_pressure, 1,                    1,             1,                       man,     woman  };
    __input1__ = {age, age,   1,       1,        1,                   40,                   weight,        30,                      alcohol, alcohol};
    __input2__ = {50,  40,    0,       0,        0,                   cholesterol,          height,        daily_physical_activity, 3,       2};
    )"""";

  const char *expectedChars = R""""(
    risk = __input0__ *** (__input1__ > __input2__);
    )"""";
  auto expectedCode = std::string(expectedChars);
  auto expectedAST = Parser::parse(expectedCode);

  EXPECT_TRUE(compareAST(*inputAST, *expectedAST));
  EXPECT_EQ(v.getAuxiliaryInformation(), expectedAuxillaryChars);
}

//TODO: Add a test case for matrix-vector-product
TEST(VectorizerTest, matrixVectorTest) {

  // Pre-CTES Program:
  // for (int i = 0; i < 3; i++) {
  //  for (int j = 0; j < 3; j++) {
  //    c[i] = c[i] + (a[i][j]*b[j]);
  // }

  // Without any variable substitution:
  // i = 0
  // c[0] = c[0] + a[0][0]*b[0];
  // c[0] = c[0] + a[0][1]*b[1];
  // c[0] = c[0] + a[0][2]*b[2];
  // c[1] = c[1] + a[1][0]*b[0];
  // c[1] = c[1] + a[1][1]*b[1];
  // c[1] = c[1] + a[1][2]*b[2];
  // c[2] = c[2] + a[2][0]*b[0];
  // c[2] = c[2] + a[2][1]*b[1];
  // c[2] = c[2] + a[2][2]*b[2];

  // With variable substitution (In reality, should be OperatorExpressions)
  // Assuming c = {0,0,0}
  // c[0] = a[0][0]*b[0] + a[0][1]*b[1] + a[0][2]*b[2];
  // c[1] = a[1][0]*b[0] + a[1][1]*b[1] + a[1][2]*b[2];
  // c[2] = a[2][0]*b[0] + a[2][1]*b[1] + a[2][2]*b[2];

  // With "merged" vectors [new] = [3*i + j]
  // c[0] = a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
  // c[1] = a[3]*b[0] + a[4]*b[1] + a[5]*b[2];
  // c[2] = a[6]*b[0] + a[7]*b[1] + a[8]*b[2];

  const char *inputChars = R""""(
   c[0] = a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
   c[1] = a[3]*b[0] + a[4]*b[1] + a[5]*b[2];
   c[2] = a[6]*b[0] + a[7]*b[1] + a[8]*b[2];
    )"""";
  auto inputCode = std::string(inputChars);
  auto inputAST = Parser::parse(inputCode);

  Vectorizer v;
  inputAST->accept(v);

  // Here, we assume that b ={b[0], b[1], b[2], 0, 0, 0, 0, 0, 0, ...}
  const char *expectedChars = R""""(
    c = a * b;
    c = c + a * rotate(b,-3);
    c = c + a * rotate(b,-6);
    )"""";
  auto expectedCode = std::string(expectedChars);
  auto expectedAST = Parser::parse(expectedCode);

  EXPECT_TRUE(compareAST(*inputAST, *expectedAST));
}

//TODO: Add a large test case for the entire kernel program

//TODO: Add test cases where the input vector is predefined and we have to rotate!

//TODO: Create CTES testcases for each manually "CTESed" program and its original.

//TODO: Write of lots of tests for "find best rotation" <- try to extend to general situations and free/constrained encodings of variables

//TODO: Write lots of tests for batchability detection logic and think about algorithm shortcuts for "boring case" like sum.

TEST(VectorizerTest, batchableExpression) {

  const char *inputChars = R""""(
    x = (a*b) + (c*d);
    )"""";
  auto inputCode = std::string(inputChars);
  auto inputAST = Parser::parse(inputCode);

  Vectorizer v;
  v.setRootScope(std::make_unique<Scope>(*inputAST));
  v.getRootScope().addIdentifier("x");
  v.getRootScope().addIdentifier("a");
  v.getRootScope().addIdentifier("b");
  v.getRootScope().addIdentifier("c");
  v.getRootScope().addIdentifier("d");
  inputAST->accept(v);

  const char *expectedAuxillaryChars = R""""(
    __input0__ = {a,c};
    __input1__ = {b,d}
    x = __input0__[0];
    )"""";

  const char *expectedChars = R""""(
    __input0__ = __input0__ * __input1__
    __input0__ = __input0__ + rotate(__input0__,1);
    )"""";
  auto expectedCode = std::string(expectedChars);
  auto expectedAST = Parser::parse(expectedCode);

  EXPECT_TRUE(compareAST(*inputAST, *expectedAST));
  EXPECT_EQ(v.getAuxiliaryInformation(), expectedAuxillaryChars);
}

// CV(x)
// TODO: If expressions are this small, maybe actually use our heuristic to skip batching them!
// slots: 0
// Execution Plan: [in Constraints: __input0__ = {a,c}, __input1__ = {c,d}]
//                   __input0__ = __input0__ * __input1__
//                   __input0__ = __input0__ + rotate(__input0__,1);

// CV(x)
// slots: 0
// Execution Plan: [in Constraints: __input0__ = {a,c}, __input1__ = {c,d}]
//                   __input0__ = __input0__ * __input1__
//                   __input0__ = __input0__ + rotate(__input0__,1);
// slots: 1
// Execution Plan: [in Constraints: __input2__ = {e,g}, __input3__ = {f,h}]
//                   __input2__ = __input2__ * __input2__
//                   __input2__ = __input2__ + rotate(__input2__,1);
TEST(VectorizerTest, batchableExpressionVectorizable) {

  const char *inputChars = R""""(
    x[0] = (a*b) + (c*d);
    x[1] = (e*f) + (g*h);
    x[2] = (i*j) + (k*l);
    x[3] = (m*n) + (o*p);
    )"""";
  auto inputCode = std::string(inputChars);
  auto inputAST = Parser::parse(inputCode);

  Vectorizer v;
  v.setRootScope(std::make_unique<Scope>(*inputAST));
  v.getRootScope().addIdentifier("x");
  v.getRootScope().addIdentifier("a");
  v.getRootScope().addIdentifier("b");
  v.getRootScope().addIdentifier("c");
  v.getRootScope().addIdentifier("d");
  inputAST->accept(v);


  // NOTE: This is what we would IDEALLY like to see.
  // However, with our current greedy system, this is not what would happen
  // Instead, it would first batch {a,c,b,d} and do the things as in the previous example?
  const char *expectedAuxillaryChars = R""""(
    __input0__ = {a,e,i,m,c,g,k,o,b,f,j,n,d,h,l,p};
    x = __input0__[0];
    )"""";

  const char *expectedChars = R""""(
    __input0__ = __input0__ * rotate(__input0__,4);
    __input0__ = __input0__ * rotate(__input0__,2);
    __input0__ = __input0__ + rotate(__input0__,1);
    )"""";
  auto expectedCode = std::string(expectedChars);
  auto expectedAST = Parser::parse(expectedCode);

  EXPECT_TRUE(compareAST(*inputAST, *expectedAST));
  EXPECT_EQ(v.getAuxiliaryInformation(), expectedAuxillaryChars);
}

//TODO: We batch vectors as continuous elements in a ctxt. But vectors-of-vectors need special logic.
// Ideally, generalize to 3-dimensional case, but probably not necessary. In that case, just treat each as individual variable.
// For now: Assume this "vector merging" happens in CTES?