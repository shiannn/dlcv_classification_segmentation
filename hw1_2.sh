#! /bin/bash

# download model
wget --no-check-certificate -O model_dict_p2_download/vgg-fcn16_64.pkl https://www.dropbox.com/s/tn433y09bkxrk18/vgg-fcn16_64.pkl?dl=0
wget --no-check-certificate -O model_dict_p2_download/vgg-fcn32_65.pkl https://www.dropbox.com/s/74ddbfomvy4m0ce/vgg-fcn32_65.pkl?dl=0
wget --no-check-certificate -O model_dict_p2_download/vgg16bn-fcn32_65.pkl https://www.dropbox.com/s/a5mx7zx8n6yq2a3/vgg16bn-fcn32_65.pkl?dl=0

# predict
python3 predict_p2.py -i $1 -o $2