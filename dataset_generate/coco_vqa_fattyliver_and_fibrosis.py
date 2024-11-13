import numpy as np
import os
import shutil
import pandas as pd
import matplotlib.pyplot as plt
import json
import skimage
from skimage.io import imread, imshow
from skimage import transform

from sklearn.model_selection import train_test_split

def create_dataset_file(opt):
    ##建立train、val、test資料夾
    for p in opt['Dataset_need']:

        if not os.path.isdir(opt['save_dataset_root']+'/'+p):
            os.mkdir(opt['save_dataset_root']+'/'+p)

def catch_case_class(root_path):

    data_name_list = os.listdir(root_path)
    picture_full_path_list = list()
    picture_name_list = list()

    for data_name in data_name_list:
        preson_name = data_name.split('_')[1]+'_'+data_name.split('_')[2]+'_'+data_name.split('_')[3].split('.')[0]
        picture_full_path_list.append(root_path+data_name)
        picture_name_list.append(preson_name)

    return picture_full_path_list,picture_name_list

def create_person_match_label(opt):

    df = pd.read_csv(opt['Tissue_label_csv_path'])
    person_ID_list = df.iloc[:,0].tolist()
    fatty_list = df.iloc[:,1].tolist()
    fibrosis_list = df.iloc[:,2].tolist()


    picture_full_path_list, picture_name_list = catch_case_class(opt['image_root_path'])
    fatty_label_macthing_picture_name_list=list()
    fibrosis_label_macthing_picture_name_list=list()

    for image_num, image_full_name in enumerate(picture_name_list):
        ID_num = person_ID_list.index(image_full_name.split('_')[0])
        fatty_label_macthing_picture_name_list.append(fatty_list[ID_num])
        fibrosis_label_macthing_picture_name_list.append(fibrosis_list[ID_num])

    return picture_full_path_list, picture_name_list,fatty_label_macthing_picture_name_list, fibrosis_label_macthing_picture_name_list

def save_pic_to_train_val_test_file(opt, dataset):

    for i in range(len(opt['Dataset_need'])):

        if not os.path.isdir(opt['save_dataset_root']+'/'+opt['Dataset_need'][i]+'/image/'):
            os.mkdir(opt['save_dataset_root']+'/'+opt['Dataset_need'][i]+'/image/')

        if not os.path.isdir(opt['save_dataset_root']+'/'+opt['Dataset_need'][i]+'/vqa_ann/'):
            os.mkdir(opt['save_dataset_root']+'/'+opt['Dataset_need'][i]+'/vqa_ann/')

        for p in dataset[i]:
            img = imread(p)
            data_name = p.split('/')[-1]
            print(data_name)
            res_img = transform.resize(img,output_shape=(512,512),order=3)
            plt.imsave(opt['save_dataset_root']+'/'+opt['Dataset_need'][i]+'/image/'+data_name,res_img)
            del p,img,res_img

def ann_json(data_path, data_list, pic_name_list, fatty_list, fibrosis_list):

    data_json_output = list()
    question_block = {
                    'Q1':'What is the degree of fatty liver?',
                    'Q2':'What is the degree of liver fibrosis?',
                    'Q3':'What are the degrees of fatty liver and liver fibrosis?',
                        }

    answer_block = {
                    'A1':'The fatty liver in the picture is {}',
                    'A2':'The liver fibrosis in the picture is {}',
                    'A3':'In the picture, the fatty liver is {} and the liver fibrosis is {}',
                        }
    c=0
    for image_num, image_full_name in enumerate(data_list):

        image_name = image_full_name.split('.')[0]
        ID_num = pic_name_list.index(image_name.split('_')[1]+'_'+image_name.split('_')[2]+'_'+image_name.split('_')[3])##對齊從dataset的圖片中找到在CSV相應的病人label
        for Q_num, Q_name in enumerate(question_block):
            c+=1
            data_propmt = dict()
            answer_list = []
            data_propmt['question_id'] = image_full_name+'_'+'{:03d}'.format(Q_num+1)##圖片ID加上第幾個問題(int)

            data_propmt['question'] = question_block[Q_name]

            if 'fatty liver' in question_block[Q_name]:

                answer_list.append(opt['fatty_liver'][fatty_list[ID_num]])

            if 'liver fibrosis' in question_block[Q_name]:

                answer_list.append(opt['liver_fibrosis'][fibrosis_list[ID_num]-1])
            print(answer_list)
            if len(answer_list)==1:
                data_propmt['answer']=[answer_block['A'+str(Q_num+1)].format(answer_list[0])]*5##list[str]
            elif len(answer_list)==2:
                data_propmt['answer']=[answer_block['A'+str(Q_num+1)].format(answer_list[0],answer_list[1])]*5##list[str]

            data_propmt['image'] = data_path+data_list[image_num]##圖片路徑(str)

            data_propmt['dataset'] = 'vqa' ##str

            data_json_output.append(data_propmt.copy())
            # print(data_json_output)
    print(c)
    return data_json_output

def save_to_json(opt, picture_name_list, fatty_label_list, fibrosis_label_list):

    for i in range(len(opt['Dataset_need'])):
        image_path = opt['save_dataset_root']+'/'+opt['Dataset_need'][i]+'/image/'
        ann_path = opt['save_dataset_root']+'/'+opt['Dataset_need'][i] +'/vqa_ann/'
        image_name_list = os.listdir(image_path)
        vqa_ann = ann_json(image_path, image_name_list, picture_name_list, fatty_label_list, fibrosis_label_list)

        with open(ann_path+'fatty_and_fibrosis.json', "w") as file:
            json.dump(vqa_ann, file)

if __name__ == '__main__':

    opt={
        'fold_list':['feature1','feature2','feature3'],
        'Dataset_need':['train','valid','test'],
        'hospital_list':['Bei-Hu20221022'],
        'fatty_liver':['normal', 'mild', 'moderate', 'severe'],
        'liver_fibrosis':['mild', 'moderate', 'severe', 'cirrhosis'],
        'image_root_path':'D:/remake_MiniGPT/Rliang_VLM/2024_11_5_minigpt_module/data/minigpt_fatty-and-fibrosis/some_data/',
        'save_dataset_root':'D:/remake_MiniGPT/Rliang_VLM/2024_11_5_minigpt_module/data/minigpt_fatty-and-fibrosis',
        'Tissue_label_csv_path':'D:/remake_MiniGPT/Rliang_VLM/fatty_and_fibrosis.csv',
            }
    create_dataset_file(opt)
    picture_full_path_list, picture_name_list, fatty_label_list, fibrosis_label_list = create_person_match_label(opt)
    X_train, X_val_test, y_train, y_val_test = train_test_split(picture_full_path_list, fatty_label_list, test_size=0.3, random_state=410,stratify=fatty_label_list)
    X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, test_size=0.3, random_state=410,stratify=y_val_test)
    train0 = len(np.where(np.array(y_train)==0)[0])/len(y_train)##確認資料平衡
    train1 = len(np.where(np.array(y_train)==1)[0])/len(y_train)
    train2 = len(np.where(np.array(y_train)==2)[0])/len(y_train)
    train3 = len(np.where(np.array(y_train)==3)[0])/len(y_train)
    val0 = len(np.where(np.array(y_val)==0)[0])/len(y_val)
    val1 = len(np.where(np.array(y_val)==1)[0])/len(y_val)
    val2 = len(np.where(np.array(y_val)==2)[0])/len(y_val)
    val3 = len(np.where(np.array(y_val)==3)[0])/len(y_val)
    test0 = len(np.where(np.array(y_test)==0)[0])/len(y_test)
    test1 = len(np.where(np.array(y_test)==1)[0])/len(y_test)
    test2 = len(np.where(np.array(y_test)==2)[0])/len(y_test)
    test3 = len(np.where(np.array(y_test)==3)[0])/len(y_test)
    print('Train data:%d,Valid data:%d,Test data:%d'%(len(y_train),len(y_val),len(y_test)))
    print('Train data 比例:0:%f,1:%f,2:%f,3:%f'%(train0,train1,train2,train3))
    print('Valid data 比例:0:%f,1:%f,2:%f,3:%f'%(val0,val1,val2,val3))
    print('Test data 比例:0:%f,1:%f,2:%f,3:%f'%(test0,test1,test2,test3))

    save_pic_to_train_val_test_file(opt, dataset = [X_train, X_val, X_test])
    save_to_json(opt, picture_name_list, fatty_label_list, fibrosis_label_list)
