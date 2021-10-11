// Expected Output for "batched" BoxBlurTest

// should follow this approach:
// Ctxt encryptedBoxBlur(Ctxt& img) {
//  int img_size = ...
//
//  // Rotate pixels in outer 3x3 kernel to center pixel's slot
//  Ctxt img_rot_1 = rotate(img, 1);
//  Ctxt img_rot_2 = rotate(img, -1);
//  Ctxt img_rot_3 = rotate(img, img_size);
//  Ctxt img_rot_4 = rotate(img, - img_size);
//  Ctxt img_rot_5 = rotate(img, img_size + 1);.
//  Ctxt img_rot_6 = rotate(img, img_size - 1);
//  Ctxt img_rot_7 = rotate(img, -img_size + 1);
//  Ctxt img_rot_8 = rotate(img, -img_size - 1);
//
//  // Add up all pixels
//  Ctxt img2 = img + img_rot_1 + img_rot_2 + img_rot_3 + img_rot_4 + img_rot_5 + img_rot_6 + img_rot_7 + img_rot_8;
//  return img2;
// }


builtin.module  {
  builtin.func private @encryptedBoxBlur(%img: tensor<16xi64>) -> tensor<16xi64> {
    %one = constant 1 : i64
    %m_one = constant -1 : i64
    %four = constant 4 : i64
    %m_four = constant -4 : i64
    %five = constant 5 : i64
    %six = constant 3 : i64
    %m_three = constant -3 : i64
    %m_five = constant -5 : i64
    %img_rot_1 = fhe.rotate(img, %one) : tensor<16xi64>
    %img_rot_2 = fhe.rotate(img, %m_one) : tensor<16xi64>
    %img_rot_3 = fhe.rotate(img, %four) : tensor<16xi64>
    %img_rot_4 = fhe.rotate(img, %m_four) : tensor<16xi64>
    %img_rot_5 = fhe.rotate(img, %five) : tensor<16xi64>
    %img_rot_6 = fhe.rotate(img, %six) : tensor<16xi64>
    %img_rot_7 = fhe.rotate(img, %m_three) : tensor<16xi64>
    %img_rot_8 = fhe.rotate(img, %m_five) : tensor<16xi64>
    %sum_0 = fhe.add %img, %img_rot_1 : tensor<16xi64>
    %sum_1 = fhe.add %sum_0, %img_rot_2 : tensor<16xi64>
    %sum_2 = fhe.add %sum_1, %img_rot_3 : tensor<16xi64>
    %sum_3 = fhe.add %sum_2, %img_rot_4 : tensor<16xi64>
    %sum_4 = fhe.add %sum_3, %img_rot_5 : tensor<16xi64>
    %sum_5 = fhe.add %sum_4, %img_rot_6 : tensor<16xi64>
    %sum_6 = fhe.add %sum_5, %img_rot_7 : tensor<16xi64>
    %sum_7 = fhe.add %sum_6, %img_rot_8 : tensor<16xi64>
    return %sum_7 : tensor<16xi64>
  }
}

