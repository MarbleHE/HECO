#include <gtest/gtest.h>
#include <include/ast_opt/mockup_classes/Ciphertext.h>

TEST(CiphertextTest, testAddCtxtCtxt) { /* NOLINT */
  Ciphertext ctxtA({10, 20, 30, 40});
  Ciphertext ctxtB({9, 5, 7, 13});
  EXPECT_EQ(ctxtA + ctxtB, Ciphertext({19, 25, 37, 53}));
}

TEST(CiphertextTest, testMulCtxtCtxt) { /* NOLINT */
  Ciphertext ctxtA({10, 20, 30, 40});
  Ciphertext ctxtB({9, 5, 7, 13});
  EXPECT_EQ(ctxtA*ctxtB, Ciphertext({90, 100, 210, 520}));
}

TEST(CiphertextTest, testAddCtxtScalar) { /* NOLINT */
  Ciphertext ctxtA({10, 20, 30, 40});
  EXPECT_EQ(ctxtA + 3.0, Ciphertext({13, 23, 33, 43}));
}

TEST(CiphertextTest, testMulCtxtScalar) { /* NOLINT */
  Ciphertext ctxtA({10, 20, 30, 40});
  EXPECT_EQ(ctxtA*3.0, Ciphertext({30, 60, 90, 120}));
}

TEST(CiphertextTest, copyCtor) { /* NOLINT */
  Ciphertext ctxtA({10, 20, 30, 40});
  EXPECT_EQ(ctxtA, Ciphertext({10, 20, 30, 40}));
  EXPECT_EQ(ctxtA.getNumCiphertextSlots(), 8192);
  EXPECT_EQ(ctxtA.getNumCiphertextElements(), 4);
  EXPECT_EQ(ctxtA.getOffsetOfFirstElement(), 0);

  Ciphertext ctxtB = ctxtA;  // same as: ctxtB(ctxtA)
  EXPECT_EQ(ctxtB.getNumCiphertextSlots(), 8192);
  EXPECT_EQ(ctxtB.getNumCiphertextElements(), 4);
  EXPECT_EQ(ctxtB.getOffsetOfFirstElement(), 0);

  ctxtA = ctxtA*2;
  EXPECT_EQ(ctxtA, Ciphertext({20, 40, 60, 80}));
  EXPECT_EQ(ctxtB, Ciphertext({10, 20, 30, 40}));
}

TEST(CiphertextTest, rotatePositive) { /* NOLINT */
  Ciphertext ctxtA({10, 20, 30, 40});
  auto rotatedA = ctxtA.rotate(14);
  EXPECT_EQ(rotatedA.getOffsetOfFirstElement(), 14);
  for (int i = 0; i < 14; ++i) EXPECT_EQ(rotatedA.getElementAt(i), 0);
  EXPECT_EQ(rotatedA.getElementAt(14), 10);
  EXPECT_EQ(rotatedA.getElementAt(15), 20);
  EXPECT_EQ(rotatedA.getElementAt(16), 30);
  EXPECT_EQ(rotatedA.getElementAt(17), 40);
  for (int i = 18; i < rotatedA.getNumCiphertextSlots(); ++i) EXPECT_EQ(rotatedA.getElementAt(i), 0);
}

TEST(CiphertextTest, rotateNegative) { /* NOLINT */
  Ciphertext ctxtA({10, 20, 30, 40});
  auto rotatedA = ctxtA.rotate(-3);
  EXPECT_EQ(rotatedA.getOffsetOfFirstElement(), -3);
  for (int i = 1; i < 8188; ++i) EXPECT_EQ(rotatedA.getElementAt(i), 0);
  EXPECT_EQ(rotatedA.getElementAt(8189), 10);
  EXPECT_EQ(rotatedA.getElementAt(8190), 20);
  EXPECT_EQ(rotatedA.getElementAt(8191), 30);
  EXPECT_EQ(rotatedA.getElementAt(0), 40);
}

TEST(CiphertextTest, rotatePositiveJumpOver) { /* NOLINT */
  auto numCiphertextSlots = 8192;
  std::vector<double> seqNums(numCiphertextSlots);
  std::iota(seqNums.begin(), seqNums.end(), 0);
  Ciphertext ctxt(seqNums, numCiphertextSlots);
  auto rotatedCtxt = ctxt.rotate(256);
  EXPECT_EQ(rotatedCtxt.getOffsetOfFirstElement(), 256);
  // numCiphertextSlots - rotationFactor = 8192 - 256 = 7936 => ctxt slot 0 has value 7936
  for (int i = 0; i < ctxt.getNumCiphertextSlots(); ++i)
    EXPECT_EQ(rotatedCtxt.getElementAt(i), (7936 + i)%numCiphertextSlots);
}

