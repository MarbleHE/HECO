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
  builtin.func private @encryptedBoxBlur(%img: tensor<16xindex>) -> tensor<16xindex> {
    %one = constant 1 : index
    %m_one = constant -1 : index
    %four = constant 4 : index
    %m_four = constant -4 : index
    %five = constant 5 : index
    %six = constant 3 : index
    %m_three = constant -3 : index
    %m_five = constant -5 : index
    %img_rot_1 = abc.fhe_rotate %img, %one : tensor<16xindex>, index, tensor<16xindex>
    %img_rot_2 = abc.fhe_rotate %img, %m_one :  tensor<16xindex>, index, tensor<16xindex>
    %img_rot_3 = abc.fhe_rotate %img, %four :   tensor<16xindex>, index, tensor<16xindex>
    %img_rot_4 = abc.fhe_rotate %img, %m_four : tensor<16xindex>, index, tensor<16xindex>
    %img_rot_5 = abc.fhe_rotate %img, %five :   tensor<16xindex>, index, tensor<16xindex>
    %img_rot_6 = abc.fhe_rotate %img, %six :    tensor<16xindex>, index, tensor<16xindex>
    %img_rot_7 = abc.fhe_rotate %img, %m_three : tensor<16xindex>, index, tensor<16xindex>
    %img_rot_8 = abc.fhe_rotate %img, %m_five : tensor<16xindex>, index, tensor<16xindex>
    %sum = abc.fhe_add (%img, %img_rot_1, %img_rot_2, %img_rot_3,  %img_rot_4, %img_rot_5, %img_rot_6, %img_rot_7, %img_rot_8) : (tensor<16xindex>, tensor<16xindex>, tensor<16xindex>, tensor<16xindex>, tensor<16xindex>, tensor<16xindex>, tensor<16xindex>, tensor<16xindex>, tensor<16xindex>) -> tensor<16xindex>
    return %sum : tensor<16xindex>
  }
}

