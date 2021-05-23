tain_class.py is a main function

StageNet.py is the proposed network which is based on Res2Net.

UNet++.py is natwork for segmenting the lung and lesion region.


StageNet-100epoches.pth is the trained model after 100 epoches. Download at https://1drv.ms/u/s!Aqsffm-C9o6dk2ReF1sXkuTy97z6?e=R9Mn4L


Each row of the train.txt is the input of our model: column[1]= save address of current CT scan; 
column[2]= save address of previous CT scan; column[3]= sex; column[2]= age; column[4]= progress;
column[5]=minus-progress; column[6]= label.
