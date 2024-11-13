import os 
import numpy as np
import shutil

def catch_case_class(opt):
    picture_full_path_list=list()
    picture_name_list = list()
    for hospital_name in opt['hospital_list']:
        for fold_name in opt['fold_list']:
            one_feature_data_name_list = os.listdir(opt['image_root_path']+'/'+hospital_name+'/'+fold_name)
            for data_name in one_feature_data_name_list:
                preson_name = data_name.split('_')[1]+'_'+data_name.split('_')[2]+'_'+data_name.split('_')[3].split('.')[0]
                picture_full_path_list.append(hospital_name+'/'+fold_name+'/'+data_name)
                picture_name_list.append(preson_name)
    return picture_full_path_list,picture_name_list

if __name__ == '__main__':

    opt = {
            'fold_list':['feature1','feature2','feature3'],
            'hospital_list':['Bei-Hu20221022'],
            'save_root':'D:/remake_MiniGPT/Rliang_VLM/data/minigpt_fatty-and-fibrosis/total_img/',
            'image_root_path':# image root
            }

    picture_full_path_list, picture_name_list = catch_case_class(opt)
    os.makedirs(opt['save_root'], exist_ok=True)
    for p in picture_full_path_list:
        if not os.path.isfile(opt['save_root']+p):
            print(p.split('/')[-1])
            shutil.copyfile(opt['image_root_path']+p, opt['save_root']+p.split('/')[-1])
