import os
import cv2
import json
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from models import SGCNNet
from metrics import eval_metrics
from train import toString

dataset_Rootdir='dataset/'

def eval(config):
    device = torch.device('cuda:0')

    selected = config['train_model']['model'][config['train_model']['select']]
    if selected == 'SGCNNet':
        model = SGCNNet.SGCN_res50(num_classes=config['num_classes'])
    check_point = os.path.join(config['save_model']['save_path'], selected+'_roads.pth')
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.2304,0.3295,0.4405],std=[0.1389,0.1316,0.1278])
        ]
    )
    model.load_state_dict(torch.load(check_point), False)
    model.cuda()
    model.eval()
    #confuse_matrix
    conf_matrix_test = np.zeros((config['num_classes'],config['num_classes']))

    correct_sum = 0.0
    labeled_sum = 0.0
    inter_sum = 0.0
    unoin_sum = 0.0
    pixelAcc = 0.0
    mIoU = 0.0
        
    class_precision=np.zeros(config['num_classes'])
    class_recall=np.zeros(config['num_classes'])
    class_f1=np.zeros(config['num_classes'])
    with open(config['img_txt'], 'r', encoding='utf-8') as f:
        for line in f.readlines():
            image_name, label_name = line.strip().split()
            root_path=dataset_Rootdir
            image_name=os.path.join(root_path,image_name)
            label_name=os.path.join(root_path,label_name)
            label = torch.from_numpy(np.asarray(Image.open(label_name), dtype=np.int32)).long().cuda()

            image = Image.open(image_name)
            image = transform(image).float().cuda()
            #batch_size=1
            image = image.unsqueeze(0)            

            output = model(image)
            correct, labeled, inter, unoin, conf_matrix_test = eval_metrics(output, label, config['num_classes'],conf_matrix_test)
            correct_sum += correct
            labeled_sum += labeled
            inter_sum += inter
            unoin_sum += unoin
            pixelAcc = 1.0 * correct_sum / (np.spacing(1) + labeled_sum)
            mIoU = 1.0 * inter_sum / (np.spacing(1) + unoin_sum)
                
            for i in range(config['num_classes']):
                #precision of each class
                class_precision[i]=1.0*conf_matrix_test[i,i]/conf_matrix_test[:,i].sum()
                #recall of each class
                class_recall[i]=1.0*conf_matrix_test[i,i]/conf_matrix_test[i].sum()
                #f1 of each class
                class_f1[i]=(2.0*class_precision[i]*class_recall[i])/(class_precision[i]+class_recall[i])
    print( 'OA {:.5f} |IOU {} |mIoU {:.5f} |class_precision {}| class_recall {} | class_f1 {}|'.format(          
            pixelAcc, toString(mIoU), mIoU.mean(),toString(class_precision),toString(class_recall),toString(class_f1)))
    np.savetxt(os.path.join(config['save_model']['save_path'], selected+'_conf_matrix_test.txt'),conf_matrix_test,fmt="%d")

if __name__ == "__main__":
    with open(r'eval_config.json', encoding='utf-8') as f:
        config = json.load(f)
    eval(config)