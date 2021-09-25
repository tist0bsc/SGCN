import os
import cv2
import json
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from models import SGCNNet

dataset_Rootdir='dataset/'

def predict(config,num_classes):
    device = torch.device('cuda:0')
    selected = config['predict_model']['model'][config['predict_model']['select']]
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



    pre_base_path = os.path.join(config['pre_dir'], 'predict_' + selected+'_test')
    if os.path.exists(pre_base_path) is False:
        os.makedirs(pre_base_path)
    pre_mask_path = os.path.join(pre_base_path, 'mask')
    if os.path.exists(pre_mask_path) is False:
        os.makedirs(pre_mask_path)
    pre_vis_path = os.path.join(pre_base_path, 'vis')
    if os.path.exists(pre_vis_path) is False:
        os.makedirs(pre_vis_path)
    
    with open(config['img_txt'], 'r', encoding='utf-8') as f:
        for line in f.readlines():
            image_name, _ = line.strip().split()
            root_path=dataset_Rootdir
            image_name=os.path.join(root_path,image_name)
            
            image = Image.open(image_name)
            image = transform(image).float().cuda()
            #batch_size=1
            image = image.unsqueeze(0)            

            output = model(image)
            _, pred = output.max(1)
            pred = pred.view(config['img_width'], config['img_height'])
            mask_im = pred.cpu().numpy().astype(np.uint8)

            file_name = image_name.split('/')[-1]
            save_label = os.path.join(pre_mask_path, file_name)
            cv2.imwrite(save_label, mask_im)
            print("写入{}成功".format(save_label))
            save_visual = os.path.join(pre_vis_path, file_name)
            print("开始写入{}".format(save_visual))
            translabeltovisual(save_label, save_visual,num_classes)
            print("写入{}成功".format(save_visual))

def translabeltovisual(save_label, path,num_classes):
    im = cv2.imread(save_label)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            pred_class=im[i][j][0]
            im[i][j] = num_classes[pred_class]
    im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, im)


if __name__ == "__main__":
    with open(r'predict_config.json', encoding='utf-8') as f:
        config = json.load(f)
    num=int(config['num_classes'])
    num_classes=[[0,0,0], [255,255,255]]
    predict(config,num_classes)


