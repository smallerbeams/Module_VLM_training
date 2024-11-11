import pandas as pd
import glob
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
from tqdm import tqdm
from sklearn.metrics import precision_score, f1_score, accuracy_score, recall_score
from sklearn.metrics import classification_report, confusion_matrix
from demo_utils import *

def gradio_ask(user_message, chatbot, chat_state, gr_img, img_list, upload_flag, replace_flag):
    if len(user_message) == 0:
        text_box_show = 'Input should not be empty!'
    else:
        text_box_show = ''

    gr_img = gr_img['image']
    
    if chat_state is None:
        chat_state = CONV_VISION.copy()

    if upload_flag:
        if replace_flag:
            chat_state = CONV_VISION.copy()  # new image, reset everything
            replace_flag = 0
            chatbot = []
        img_list = []
        llm_message = chat.upload_img(gr_img, chat_state, img_list)
        upload_flag = 0

    chat.ask(user_message, chat_state)

    chatbot = chatbot + [[user_message, None]]

    return text_box_show, chatbot, chat_state, img_list, upload_flag, replace_flag

def gradio_answer(chatbot, chat_state, img_list, temperature):
    if len(img_list) > 0:
        if not isinstance(img_list[0], torch.Tensor):
            chat.encode_img(img_list)
    llm_message = chat.answer(conv=chat_state,
                              img_list=img_list,
                              temperature=temperature,
                              max_new_tokens=500,
                              max_length=2000)[0]
    chatbot[-1][1] = llm_message
    return chatbot, chat_state

def i_o(user_message, input_image):
    chatbot,image_list, chat_stage, upload_flag, replace_flag,  = [], [], None, 1, 0
    # model setting
    temperature = 0.6
    user_message = user_message
    input_image = input_image
    gr_img = {"image":input_image, "mask":[]}
    text_box_show, chatbot, chat_state, img_list, upload_flag, replace_flag = gradio_ask(user_message, chatbot, chat_stage, gr_img, image_list, upload_flag, replace_flag)
    ###測試
    # print(chatbot)
    # breakpoint()
    ###
    chatbot, chat_state = gradio_answer(chatbot = chatbot, chat_state = chat_state, img_list = img_list, temperature=temperature)
    Q_ = chatbot[0][0]
    A_ = chatbot[0][1]
    # print("===================================")
    # print("Q: ", Q_)
    # print("A: ", A_)
    # print("===================================")
    # if A_=='Flawed' or A_=='yes':
    #     A_ = 1
    # else:
    #     A_ = 0
    return A_
def metric_calc(y_test, y_pred, ths=0.5):
    if ths!=None:
        y_pred = (y_pred>ths).astype(int)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
    sens = recall_score(y_test, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='macro')
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    # print(tn, fp, fn, tp)
    spec = tn/(tn+fp)
    return [acc*100, prec*100, sens*100, spec*100, f1*100, int(tp), int(fn), int(tn), int(fp)]

def check_format(question_block, pepole_qustion, answer_block, robot_answer):
    ####Ture is 1 Flase is 0
    QB_num = question_block.index(pepole_qustion)
    # print(QB_num)
    # need_answer_sentence = answer_block[QB_num].split('{}')[:-1]
    need_answer_sentence = ['The fatty liver in the picture is ','The liver fibrosis in the picture is ',
                            'the fatty liver in the picture is ','the liver fibrosis in the picture is ',
                        'In the picture, the fatty liver is ', 'and the liver fibrosis is ']
    format_ok = 0
    for p in need_answer_sentence:
        if p in robot_answer:
            format_ok = 1
            break
        else:
            format_ok = 0
            # print(format_ok)

    return format_ok

def check_level(question_block, pepole_qustion, answer_list, answer_block, robot_answer):
    ####Ture is 1 Flase is 0
    QB_num = question_block.index(pepole_qustion)
    # print(QB_num)
    # need_answer_sentence = answer_block[QB_num].split('{}')[:-1]
    need_answer_sentence = ['The fatty liver in the picture is {}','The liver fibrosis in the picture is {} ',
                            'the fatty liver in the picture is {}','the liver fibrosis in the picture is {}',
                        'In the picture, the fatty liver is {} and the liver fibrosis is {}']
    level_ok = 0
    if len(answer_list)==1:
        for p in need_answer_sentence[:-1]:
            ans_des = p.format(answer_list[0])##list[str]
            print(ans_des)
            print(robot_answer)
            if ans_des in robot_answer:
                level_ok = 1
                break
            else:
                level_ok = 0

    elif len(answer_list)==2:
        ans_des = need_answer_sentence[-1].format(answer_list[0],answer_list[1])##list[str]
        print(ans_des)
        print(robot_answer)
        if ans_des in robot_answer:
            level_ok = 1
        else:
            level_ok = 0
    return level_ok


if __name__ == '__main__':

    model_config = {'amp':True,
                    'use_distributed':False,
                    'accum_grad_iters':1,
                    'batch_size':1,
                    'max_epoch':5,
                    'cuda_enabled':True,
                    'chat_template':True,
                    'end_sym':['</s>',"\n <|start_header_id|>assistant<|end_header_id|> {} <|eot_id|>"],
                    'prompt_template':["<s>[INST] {} [/INST]",
                    """<|begin_of_text|><|start_header_id|>user<|end_header_id|> {} <|eot_id|>"""],
                    'max_txt_len': 1024,
                    'max_context_len': 3500,
                    'output_dir':'/mnt/disk4/Rliang/Module_minigpt_traing/ckpt/fatty-and-fibrosis_try',
                    'stage_ckpt':'/mnt/disk4/Rliang/Module_minigpt_traing/ckpt/fatty-and-fibrosis_try/checkpoint_4.pth',
                    # 'stage_ckpt':'',
                    'vis_root_train':'/mnt/disk4/acvlab/dataset/minigpt_fatty-and-fibrosis_train/coco/image/train',
                    'ann_paths_train':['/mnt/disk4/acvlab/dataset/minigpt_fatty-and-fibrosis_train/coco_vqa/fatty_and_fibrosis.json'],
                    'vis_root_valid':'/mnt/disk4/acvlab/dataset/minigpt_fatty-and-fibrosis_test/coco/image/test',
                    'ann_paths_valid':['/mnt/disk4/acvlab/dataset/minigpt_fatty-and-fibrosis_test/coco_vqa/fatty_and_fibrosis.json'],
                    }
    llm_config = {'llm_model_path':'/mnt/disk4/acvlab/demo_code/MiniGPT-4/Llama-2-7b-chat-hf',#必用權重
                    'low_resource':False,
                    'low_res_device':0,
                    'lora_r':64,
                    'lora_target_modules':["q_proj", "v_proj"],
                    'lora_alpha':16,
                    'lora_dropout':0.05
                    }
    vit_config = {
                    # 'model_path':'/mnt/disk4/Rliang/Minigpt_model_stage/models/eva_vit_g.pth',
                    'model_path':None,
                    'image_size': 448,  #bilp2 = 448, clip = 224 or 336
                    'drop_path_rate':0,
                    'use_grad_checkpoint':True,
                    'vit_precision':'fp16',
                    'freeze_vit':True,
                    }
    lr_config = {'init_lr': 1e-5,
                    'beta2':0.999,
                    'min_lr':1e-6,
                    'decay_rate':None,
                    'weight_decay':0.05,
                    'warmup_start_lr':1e-6,
                    'warmup_steps':1000,
                    'iters_per_epoch':1000
                }
    DDP_config = {'distributed_set':True,
                    'rank':None,
                    'world_size':0,
                    'gpu':0,
                    'dist_url':"env://",
                    'dist_backend':"nccl"
                        }
    print('====Code Strat====')
    print('====Model create ====')
    model = base_minigptv2.Minigptv2(vit_config=vit_config, llm_config=llm_config, model_conf= model_config)
    if model_config['stage_ckpt']:
        ####讀取minigptv2_finetune.yaml
        print("Load Minigpt-v2-LLM Checkpoint: {}".format(model_config['stage_ckpt']))
        ckpt = torch.load(model_config['stage_ckpt'], map_location="cpu")
        msg = model.load_state_dict(ckpt['model'], strict=False)
        # print(model)
        # summary(model.to('cpu'),input_size=(3,224,224),device='cpu')
        # breakpoint()
    else:
        print("NO CKPT")
        # print(model.state_dict())

        # model = None
    print('====Model OK====')
    print('====Dataset create ====')
    bounding_box_size = 100
    model = model.eval()
    image_dir = glob.glob(os.path.join("/mnt/disk4/acvlab/dataset/minigpt_fatty-and-fibrosis_test/coco/image/test", "*"))
    df = pd.read_csv('/mnt/disk4/acvlab/dataset/fatty_and_fibrosis.csv')
    person_ID_list = df.iloc[:,0].tolist()
    fatty_list = df.iloc[:,1].tolist()
    fibrosis_list = df.iloc[:,2].tolist()
    # user_message = "Does this photo have any flaws? Defective or Flawless" #描述
    # user_message = "Does this photo have any flaws?" #描述
    question_block = [
                    'What is the degree of fatty liver?',
                    'What is the degree of liver fibrosis?',
                    'What are the degrees of fatty liver and liver fibrosis?',
                        ]
    answer_block = [
                    'The fatty liver in the picture is {}.',
                    'The liver fibrosis in the picture is {}.',
                    'In the picture, the fatty liver is {} and the liver fibrosis is {}.',
                        ]
    GT=[]
    GT_format=[]
    GT_level=[]
    GT_fatty = []
    GT_fibrosis = []
    PD = []
    images_name=[]
    format_Ture_count=[]
    level_Ture_count =[]
    Q_format_score_list = []
    Q_level_score_list = []
    Q_item_list = []
    fatty_liver_answer = ['exclude', 'mild', 'moderate', 'severe']
    liver_fibrosis_answer = ['mild', 'moderate', 'severe', 'cirrhosis']
    for Question_num, Question in enumerate(question_block):
        user_message = Question
        print(Question_num,user_message)
        Q_format_score = 0
        Q_level_score = 0
        for img_path in tqdm(image_dir):
            # breakpoint()
            answer_list = []
            chat = Chat(model, vis_processor, device=device)
            full_img_name = img_path.split('/')[-1]
            img_name_num = full_img_name.split('_')[1]
            img_name = full_img_name.split('.')[0]
            images_name.append(img_name+'_'+'{:03d}'.format(Question_num+1))
            ID_num = person_ID_list.index(img_name_num)
            GT_fatty.append(fatty_list[ID_num])
            GT_fibrosis.append(fibrosis_list[ID_num])

            if 'fatty liver' in Question:

                answer_list.append(fatty_liver_answer[fatty_list[ID_num]])

            if 'liver fibrosis' in Question:

                answer_list.append(liver_fibrosis_answer[fibrosis_list[ID_num]-1])
            # print(answer_block[Question_num].format(answer_list[0]))
            if len(answer_list)==1:
                data_propmt=[answer_block[Question_num].format(answer_list[0])]##list[str]
            elif len(answer_list)==2:
                data_propmt=[answer_block[Question_num].format(answer_list[0],answer_list[1])]##list[str]

            GT.append(data_propmt[0])
            # GT_format.append(answer_block[Question_num].split('{}'))
            # GT_level.append(answer_list)

            input_image = Image.open(img_path).convert("RGB")
            robot_reslut=i_o(user_message, input_image)

            PD.append(robot_reslut)
            format_reslut = check_format(question_block, Question, answer_block, robot_reslut)
            level_reslut = check_level(question_block, Question, answer_list, answer_block, robot_reslut)
            format_Ture_count.append(format_reslut)
            level_Ture_count.append(level_reslut)

            Q_format_score += format_reslut
            Q_level_score += level_reslut

        Q_format_score_list.append(Q_format_score)
        Q_level_score_list.append(Q_level_score)
        Q_item_list.append(len(image_dir))
    [Q_item_list.append(0)for i in range(len(images_name)-len(Q_item_list))]
    [Q_format_score_list.append(0)for i in range(len(images_name)-len(Q_format_score_list))]
    [Q_level_score_list.append(0)for i in range(len(images_name)-len(Q_level_score_list))]
    print(len(images_name), len(GT_fatty),len(GT_fibrosis),len(GT_level),len(GT))
    print(sum(format_Ture_count))
    print(sum(level_Ture_count))
    check_df = pd.DataFrame([])
    check_df['name'] = images_name
    check_df['GT_fatty'] = GT_fatty
    check_df['GT_fibrosis'] = GT_fibrosis
    # check_df['GT_format_ans'] = GT_format
    # check_df['GT_level_ans'] = GT_level
    check_df['GT_ans'] = GT
    check_df['PD_ans'] = PD
    check_df['PD_format_ans'] = format_Ture_count
    check_df['PD_level_ans'] = level_Ture_count
    check_df['Q_Right_format'] = Q_format_score_list
    check_df['Q_Right_level'] = Q_level_score_list
    check_df['Q_item'] = Q_item_list

    print(check_df)
    # check_df.to_csv(f'./demo_ckpt/fatty-and-fibrosis_for_module.csv', index=False, encoding='utf-8-sig')
    breakpoint()
    