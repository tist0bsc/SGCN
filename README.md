# SGCN
Depth-wise Split Separable Graph Convolution Network for Road Extraction in Complex Environment from High-resolution Remote Sensing Imagery

## quick start
### requirements
python 3.6 CUDA10.1 GPU:2080Ti*1

pytorch==1.1.0
tqdm==4.49.0
Pillow==6.1.0
opencv-python==4.1.0.25
### parameters
You can changge epoch，batch_size，lr and decay in train_config.json
### train
1.Download the files mentioned in "dataset/gansu/readme"  
2.python3 main.py  
It will create a folder, named 'logs', and a log file. This log file will record the training process. 

And the trained model, with maximum OA in validation set, will be saved in a folder, named 'saved', and record the epoch num in 'best_epoch.txt' when saving the model. 
### eval
1.python3 eval.py  
It will evalute this model in test dataset, and print the metrics, including OA, IOU, precision, recall, F1, then save the confusion matrix in 'saved' folder.
### perdict
1.python3 predict.py  
You can find the visual results in 'predict/gansu/'，grays value of roads in 'vis' is 255 ，while 1 in 'mask'.

