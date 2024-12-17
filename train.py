import os, sys
import torch
import signal ## KeyboardInterrupt 程序流程指令
import torch.distributed as dist
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "true"
from torch.nn.parallel import DistributedDataParallel as DDP
from process_block.common.dist_utils import get_rank, init_distributed_mode
import random
import numpy as np
import torch.backends.cudnn as cudnn


####model 資料夾
from process_block.models import base_minigptv2
####dataset 資料夾

from process_block.processors.blip_processors import BlipCaptionProcessor, \
                Blip2ImageTrainProcessor, Blip2ImageEvalProcessor

from process_block.Dataset.coco_vqa_dataset import COCOVQADataset, COCOVQAEvalDataset
from process_block.Dataloader.Mutli_dataloader import Minigptv2_Mutli_Dataloader
from torch.utils.data import DataLoader
####optimizer
from process_block.utils.optimizer_factory import minigptv2_optimizer
from process_block.utils.lr_sched_cls import LinearWarmupCosineLRScheduler
#### train loop
from main import Task

####---------------------------------------------------------------------------------------
def setup_seeds():
    seed = 42 + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True

def signal_handler(signal, frame):##可以在主程式執行時，執行鍵盤指令時可以直接中止主程式內容，不會卡住
    print("\nCaught interrupt signal. Cleaning up CUDA memory...")
    torch.cuda.empty_cache()
    sys.exit(0)

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
                    'stage_ckpt':'/mnt/disk4/acvlab/demo_code/MiniGPT-4/checkpoint_stage2.pth',
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
                    # 'model_path':None, #eva_clip_g, clip_large_336
                    'model_path':'/mnt/disk4/Rliang/Minigpt_model_stage/models/eva_vit_g.pth',
                    # 'model_path':"../../VITModel/clip-vit-base-patch16", #clip-vit-base-patch16, clip-vit-large-patch14-336
                    # 'model_path':"../../VITModel/clip-vit-large-patch14-336", #clip-vit-base-patch16, clip-vit-large-patch14-336
                    'image_size': 448,  #bilp2 = 448, clip = 224 or 336
                    # 'image_size':336,  #bilp2 = 448, clip = 224 or 336
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

    signal.signal(signal.SIGINT, signal_handler)
    try:
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
        # print(model)
        # breakpoint()
        print('====Dataset create ====')
        # train_dataset = COCOVQADataset(
        #                     Blip2ImageTrainProcessor(image_size=vit_config['image_size']),
        #                     BlipCaptionProcessor(),
        #                     vis_root=model_config['vis_root_train'],
        #                     ann_paths=model_config['ann_paths_train']
        #                     )
        # training_loader = DataLoader(
        #                     train_dataset,
        #                     batch_size=model_config['batch_size'],
        #                     num_workers=10,
        #                     shuffle=True,
        #                     pin_memory=True
        #                     )
        Mutli_dataset = {
                    'coco_vqa':{'train' : COCOVQADataset(
                                            Blip2ImageTrainProcessor(image_size=vit_config['image_size']),
                                            BlipCaptionProcessor(),
                                            vis_root=model_config['vis_root_train'],
                                            ann_paths=model_config['ann_paths_train']
                                                ),
                                'batch_size' : 4,
                                'sample_ratio' : 15}
                                }
        training_loader = Minigptv2_Mutli_Dataloader(
                            datasets = Mutli_dataset,
                            num_workers=10,
                            is_trains = True,
                            model_conf = model_config,
                            ).train_loader

        print('====Dataset OK ====')
        print('====Training task create ====')
        tasks = Task(
                    model = model,
                    data_loader = training_loader,
                    optimizer = minigptv2_optimizer(lr_conf=lr_config, model=model),
                    lr_scheduler = LinearWarmupCosineLRScheduler(
                                    optimizer = minigptv2_optimizer(lr_conf=lr_config, model=model),
                                    max_epoch = model_config['max_epoch'],
                                    iters_per_epoch = lr_config['iters_per_epoch'],
                                    min_lr = lr_config['min_lr'],
                                    init_lr = lr_config['init_lr'],
                                    warmup_steps=lr_config['warmup_steps'],
                                    warmup_start_lr=lr_config['warmup_start_lr'],
                                        ),
                    model_conf = model_config
                    )
        # breakpoint()
        print('====Training task OK ====')

        tasks.train()
        # if not hasattr(training_loader, "__next__"):
        #     # convert to iterator if not already
        #     data_loader = iter(training_loader)
        # samples = next(data_loader)
        # print(samples)
        # breakpoint()
        # main_.main_process(max_epoch=20, lr_config=lr_config)
    except KeyboardInterrupt:
        pass
    finally:
        print("\nExiting program.")

