#! /bin/bash

# download model
wget --no-check-certificate -O model_dict_p1_download/vgg16_bn78.pkl https://www.dropbox.com/s/qz9q9ryvim64grl/vgg16_bn78.pkl?dl=0
wget --no-check-certificate -O model_dict_p1_download/vgg19_bn77.pkl https://www.dropbox.com/s/9ax7enanxgj4tn6/vgg19_bn77.pkl?dl=0

# predict
python3 predict_p1.py -i $1 -o $2