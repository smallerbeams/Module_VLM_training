
import os
import numpy as np
import torch
import torch.nn as nn
import contextlib

from transformers import LlamaTokenizer

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_int8_training,
)
from process_block.models.eva_vit import create_eva_vit_g
from process_block.models.Llama2_Causual import LlamaForCausalLM


class Minigptv2(nn.Module):

    def __init__(self, vit_config=dict(), llm_config=dict(), model_conf=dict()):
        super().__init__()

        self.llama_model, self.llama_tokenizer = self.init_llm(llm_config['llm_model_path'], 
            low_resource=llm_config['low_resource'], low_res_device=llm_config['low_res_device'], 
            lora_r=llm_config['lora_r'], lora_target_modules=llm_config['lora_target_modules'],
            lora_alpha=llm_config['lora_alpha'],lora_dropout=llm_config['lora_dropout'],
                )

        self.visual_encoder, self.ln_vision = self.init_vision_encoder(vit_config['model_path'],
            vit_config['image_size'],vit_config['drop_path_rate'],vit_config['use_grad_checkpoint'],
            vit_config['vit_precision'],vit_config['freeze_vit'],
                )
        img_f_dim = self.visual_encoder.num_features * 4

        self.llama_proj = nn.Linear(img_f_dim, self.llama_model.config.hidden_size)

        self.max_context_len = model_conf['max_context_len']
        self.max_txt_len = model_conf['max_txt_len']
        self.chat_template = model_conf['chat_template']
        self.prompt_template = model_conf['prompt_template'][0]
        self.end_sym = model_conf['end_sym'][0]

    #####eva_clip_g
    @classmethod
    def init_vision_encoder(cls, model_path, img_size, drop_path_rate,
                    use_grad_checkpoint, precision, freeze):
        print('====Loading VIT====')

        # assert model_name == "eva_clip_g", "vit model must be eva_clip_g for current version of MiniGPT-4"
        if not freeze:
            precision = "fp32"  # fp16 is not for training

        visual_encoder = create_eva_vit_g(
            model_path, img_size, drop_path_rate, use_grad_checkpoint, precision
        )
        # print("visual_encoder.num_features", visual_encoder.num_features)
        ln_vision = LayerNorm(visual_encoder.num_features)
        # print("VIT model", visual_encoder)
        # print("ln_vision: ", ln_vision)
        if not freeze:
            for name, param in visual_encoder.named_parameters():
                param.requires_grad = True
            visual_encoder = visual_encoder.eval()
            visual_encoder.train = disabled_train
            for name, param in ln_vision.named_parameters():
                param.requires_grad = True
            ln_vision = ln_vision.eval()
            ln_vision.train = disabled_train

        if freeze:
            for name, param in visual_encoder.named_parameters():
                param.requires_grad = False
            visual_encoder = visual_encoder.eval()
            visual_encoder.train = disabled_train
            for name, param in ln_vision.named_parameters():
                param.requires_grad = False
            ln_vision = ln_vision.eval()
            ln_vision.train = disabled_train
            print("freeze vision encoder")

        print('====Loading VIT Done====')
        return visual_encoder, ln_vision
    ####

    ####llama2
    @classmethod
    def init_llm(cls, llama_model_path, low_resource=False, low_res_device=0, lora_r=0,
                 lora_target_modules=["q_proj","v_proj"], **lora_kargs):
        print('====Loading LLAMA====')
        llama_tokenizer = LlamaTokenizer.from_pretrained(llama_model_path, use_fast=False)
        # llama_tokenizer = AutoTokenizer.from_pretrained(llama_model_path,use_fast=False)
        llama_tokenizer.pad_token = "$$"
        print('====tokenizer OK====')

        if low_resource:
            llama_model = LlamaForCausalLM.from_pretrained(
                llama_model_path,
                torch_dtype=torch.float16,
                load_in_8bit=True,
                device_map={'': low_res_device}
            )
        else:
            llama_model = LlamaForCausalLM.from_pretrained(
                llama_model_path,
                torch_dtype=torch.float16,
            )

        if lora_r > 0:
            print(f"=======================LORA not 0 = {lora_r}=======================")
            llama_model = prepare_model_for_int8_training(llama_model)
            loraconfig = LoraConfig(
                r=lora_r,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=lora_target_modules,
                **lora_kargs
            )
            llama_model = get_peft_model(llama_model, loraconfig)

            llama_model.print_trainable_parameters()

        else:
            for name, param in llama_model.named_parameters():
                param.requires_grad = False
        print('====Loading LLAMA Done====')
        return llama_model, llama_tokenizer
    ####

    ####model processor
    @property
    def device(self):
        return list(self.parameters())[-1].device

    def maybe_autocast(self,dtype=torch.float16):
        # if on cpu, don't use autocast
        # if on gpu, use autocast with dtype if provided, otherwise use torch.float16
       
        enable_autocast = self.device != torch.device("cpu")

        if enable_autocast:
            return torch.cuda.amp.autocast(dtype=dtype)
        else:
            return contextlib.nullcontext()

    def encode_img(self, image):
        device = image.device
        # print(device)
        # breakpoint()
        ##測試
        # print('encode_image_is_here')

        ##
        if len(image.shape) > 4:
            image = image.reshape(-1, *image.shape[-3:])

        with self.maybe_autocast():
            # image_embeds = self.ln_vision(self.visual_encoder(image)).to(device)
            image_embeds = self.visual_encoder(image).to(device)
            image_embeds = self.ln_vision(image_embeds)
            # print("ViT OutPut-1: ", image_embeds.shape)
            image_embeds = image_embeds[:, 1:, :]
            # print("ViT OutPut-2: ", image_embeds.shape)
            bs, pn, hs = image_embeds.shape
            # print("image_embeds", image_embeds.shape)
            image_embeds = image_embeds.view(bs, int(pn / 4), int(hs * 4))
            # print("image_embeds", image_embeds.shape)
            inputs_llama = self.llama_proj(image_embeds)
            # print("Input llama: ", inputs_llama.shape)
            atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(image.device)
            # print("atts_llama: ", atts_llama.shape)
        return inputs_llama, atts_llama

    def embed_tokens(self, token_ids):
        # print("token_ids: ", token_ids)
        if hasattr(self.llama_model.base_model, 'model'): ## lora wrapped model
            embeds = self.llama_model.base_model.model.model.embed_tokens(token_ids)
        else:
            embeds = self.llama_model.base_model.embed_tokens(token_ids)
        # print("[embed_tokens]embeds: ", embeds.shape)
        return embeds

    def prompt_wrap(self, img_embeds, atts_img, prompts, lengths=None):
        if prompts is None or len(prompts) == 0:
            # prompts is not provided, just return the original image embedding
            return img_embeds, atts_img
        elif img_embeds is None:
            # prompt is provided but there is no image embedding. return the prompt embedding in right padding
            self.llama_tokenizer.padding_side = "right"
            prompt_tokens = self.llama_tokenizer(
                prompts,
                return_tensors="pt",
                padding="longest",
                add_special_tokens=False
            ).to(self.device)
            prompt_embeds = self.embed_tokens(prompt_tokens.input_ids)
            atts_prompt = prompt_tokens.attention_mask
            return prompt_embeds, atts_prompt
        else:
            # return the multi-modal embedding in right padding
            emb_lists = []
            # print("New prompts", prompts)
            if isinstance(prompts, str):
                prompts = [prompts] * len(img_embeds)

            for idx, (each_img_embed, each_prompt) in enumerate(zip(img_embeds, prompts)):
                pn = each_img_embed.shape[-2]
                if lengths is not None:
                    each_img_embed = each_img_embed.reshape(-1, each_img_embed.shape[-1])
                    each_img_embed = each_img_embed[:lengths[idx] * pn]
                p_segs = each_prompt.split('<ImageHere>')
                interleave_emb = []
                for idx, seg in enumerate(p_segs[:-1]):
                    p_tokens = self.llama_tokenizer(
                        seg, return_tensors="pt", add_special_tokens=False).to(img_embeds.device)
                    p_embed = self.embed_tokens(p_tokens.input_ids)
                    interleave_emb.append(torch.cat([p_embed, each_img_embed[None][:, idx * pn:(idx + 1) * pn]], dim=1))
                wrapped_emb = torch.cat(interleave_emb, dim=1)
                p_tokens = self.llama_tokenizer(
                    p_segs[-1], return_tensors="pt", add_special_tokens=False).to(img_embeds.device)
                p_embed = self.embed_tokens(p_tokens.input_ids)
                wrapped_emb = torch.cat([wrapped_emb, p_embed], dim=1)
                emb_lists.append(wrapped_emb)

            emb_lens = [emb.shape[1] for emb in emb_lists]
            pad_emb = self.embed_tokens(torch.tensor(self.llama_tokenizer.pad_token_id, device=img_embeds.device))

            max_length = max(emb_lens) if max(emb_lens) < self.max_context_len else self.max_context_len
            wrapped_embs = pad_emb.expand(len(emb_lens), max_length, -1).clone()
            wrapped_atts = torch.zeros([len(emb_lens), max_length], dtype=torch.int, device=img_embeds.device)
            
            for i, emb in enumerate(emb_lists):
                length = emb_lens[i] if emb_lens[i] < self.max_context_len else self.max_context_len
                wrapped_embs[i, :length] = emb[:, :length]
                wrapped_atts[i, :length] = 1
            return wrapped_embs, wrapped_atts

    def tokenize_conversation(self, conv_q, conv_a):
        """concatenate conversation and make sure the model is only trained to regress the answer"""

        to_regress_token_ids_list = []
        targets_list = []

        batch_size = len(conv_q)
        for batch_idx in range(batch_size):
            questions, answers = conv_q[batch_idx], conv_a[batch_idx]
            questions = [self.llama_tokenizer(self.llama_tokenizer.bos_token + q,
                                              return_tensors="pt",
                                              add_special_tokens=False).to(self.device) for q in questions[1:]]  # the first question is handled in the prompt wrap function, skip it
            answers = [self.llama_tokenizer(a + self.end_sym,
                                            return_tensors="pt",
                                            add_special_tokens=False).to(self.device) for a in answers]
            cur_id = []
            cur_target = []
            for i in range(len(questions)):
                cur_id.append(answers[i].input_ids)
                cur_target.append(answers[i].input_ids)
                cur_id.append(questions[i].input_ids)
                cur_target.append(torch.ones_like(questions[i].input_ids) * -100)

            cur_id.append(answers[-1].input_ids)
            cur_target.append(answers[-1].input_ids)

            cur_id = torch.cat(cur_id, dim=1)
            cur_target = torch.cat(cur_target, dim=1)
            to_regress_token_ids_list.append(cur_id)
            targets_list.append(cur_target)

        max_len = min(max([target.shape[1] for target in targets_list]), self.max_txt_len)
        to_regress_token_ids = torch.ones([batch_size, max_len],
                                          dtype=cur_id.dtype, device=self.device) * self.llama_tokenizer.pad_token_id
        targets = torch.ones([batch_size, max_len],
                                          dtype=cur_id.dtype, device=self.device) * -100
        for batch_idx in range(batch_size):
            cur_len = to_regress_token_ids_list[batch_idx].shape[1]
            to_regress_token_ids[batch_idx, :cur_len] = to_regress_token_ids_list[batch_idx][0, :max_len]
            targets[batch_idx, :cur_len] = targets_list[batch_idx][0, :max_len]

        to_regress_token_attn = (to_regress_token_ids != self.llama_tokenizer.pad_token_id).to(torch.int)

        return to_regress_token_ids, to_regress_token_attn, targets

    def preparing_embedding(self, samples):
        # print("Samples", samples)
        ### prepare input tokens
        if 'image' in samples:
            img_embeds, img_atts = self.encode_img(samples["image"])
    
        else:
            img_embeds = img_atts = None

        if 'conv_q' in samples:
            # handeling conversation datasets
            conv_q, conv_a = samples['conv_q'], samples['conv_a']

            connect_sym = samples['connect_sym'][0]
            conv_q = [q.split(connect_sym)for q in conv_q]
            conv_a = [a.split(connect_sym) for a in conv_a]

            conv_q = [[self.prompt_template.format(item) for item in items] for items in conv_q]

            cond_embeds, cond_atts = self.prompt_wrap(img_embeds, img_atts, [q[0] for q in conv_q])
            regress_token_ids, regress_atts, part_targets = self.tokenize_conversation(conv_q, conv_a)

        else:##走這邊
            if "instruction_input" in samples:
                instruction = samples["instruction_input"]
            elif self.prompt_list:
                instruction = random.choice(self.prompt_list)
            else:
                instruction = None

            if hasattr(self, 'chat_template') and self.chat_template:
                instruction = [self.prompt_template.format(instruct) for instruct in instruction]

            if 'length' in samples:
                # the input is a image train (like videos)
                bsz, pn, hs = img_embeds.shape
                img_embeds = img_embeds.reshape(len(samples['image']), -1, pn, hs)
                cond_embeds, cond_atts = self.prompt_wrap(img_embeds, img_atts, instruction, samples['length'])
            else:
                cond_embeds, cond_atts = self.prompt_wrap(img_embeds, img_atts, instruction)

            ### prepare target tokens
            self.llama_tokenizer.padding_side = "right"
            text = [t + self.end_sym for t in samples["answer"]]

            regress_tokens = self.llama_tokenizer(
                text,
                return_tensors="pt",
                padding="longest",
                truncation=True,
                max_length=self.max_txt_len,
                add_special_tokens=False
            ).to(self.device)

            regress_token_ids = regress_tokens.input_ids
            regress_atts = regress_tokens.attention_mask
            part_targets = regress_token_ids.masked_fill(
                regress_token_ids == self.llama_tokenizer.pad_token_id, -100
            )

        regress_embeds = self.embed_tokens(regress_token_ids)

        return cond_embeds, cond_atts, regress_embeds, regress_atts, part_targets

    def concat_emb_input_output(self, input_embs, input_atts, output_embs, output_atts):
        """
        Concatenate the batched input embedding and batched output embedding together.
        Both the input and the output embedding should be right padded.
        """
        input_lens = []
        cat_embs = []
        cat_atts = []
        for i in range(input_embs.size(0)):
            input_len = input_atts[i].sum()
            input_lens.append(input_len)
            cat_embs.append(
                torch.cat([
                    input_embs[i][:input_len],
                    output_embs[i],
                    input_embs[i][input_len:]
                ])
            )
            cat_atts.append(
                torch.cat([
                    input_atts[i][:input_len],
                    output_atts[i],
                    input_atts[i][input_len:]
                ])
            )
        cat_embs = torch.stack(cat_embs)
        cat_atts = torch.stack(cat_atts)
        return cat_embs, cat_atts, input_lens

    def get_context_emb(self, prompt, img_list):
        # print("prompt", prompt)
        # print("img_list", img_list[0].shape)
        # print(img_list)
        device = img_list[0].device
        # print(device)
        prompt_segs = prompt.split('<ImageHere>')
        # print("描述切割: ", prompt_segs)
        # print("len(prompt_segs), len(img_list): ", len(prompt_segs), len(img_list))
        assert len(prompt_segs) == len(img_list) + 1, "Unmatched numbers of image placeholders and images."
        seg_tokens = [
            self.llama_tokenizer(
                seg, return_tensors="pt", add_special_tokens=i==0).to(device).input_ids # only add bos to the first seg
            for i, seg in enumerate(prompt_segs)
        ]
        # print("輸入文字Token", seg_tokens)
        seg_embs = [self.embed_tokens(seg_t) for seg_t in seg_tokens]

        mixed_embs = [emb for pair in zip(seg_embs[:-1], img_list) for emb in pair] + [seg_embs[-1]]
        ##測試
        # print("mixed_embs-0", len(mixed_embs))
        # print('mixed_embs第1步部分:{}\nmixed_embs第2步部分:{}\nmixed_embs第3步部分:{}'.format(mixed_embs[0].shape,mixed_embs[1].shape,mixed_embs[2].shape))
        # breakpoint()
        ##
        # print(f"前後文字與中間影像embedding shape: 前){mixed_embs[0].shape} 中){mixed_embs[1].shape} 後)mixed_embs[0].shape{2}")
        mixed_embs = torch.cat(mixed_embs, dim=1)
        # print("前後文字與影像合併後shape", mixed_embs.shape)
        return mixed_embs

    def forward(self, samples, reduction='mean'):
        # prepare the embedding to condition and the embedding to regress
        cond_embeds, cond_atts, regress_embeds, regress_atts, part_targets = \
            self.preparing_embedding(samples)
        # print('====promt_embedding OK====')
        # concat the embedding to condition and the embedding to regress
        inputs_embeds, attention_mask, input_lens = \
            self.concat_emb_input_output(cond_embeds, cond_atts, regress_embeds, regress_atts)
        # print('====concat_emb_input_output OK====')
        # get bos token embedding
        bos = torch.ones_like(part_targets[:, :1]) * self.llama_tokenizer.bos_token_id
        bos_embeds = self.embed_tokens(bos)
        bos_atts = cond_atts[:, :1]

        # add bos token at the begining
        inputs_embeds = torch.cat([bos_embeds, inputs_embeds], dim=1)
        attention_mask = torch.cat([bos_atts, attention_mask], dim=1)

        # ensemble the final targets
        targets = torch.ones([inputs_embeds.shape[0], inputs_embeds.shape[1]],
                             dtype=torch.long).to(self.device).fill_(-100)

        for i, target in enumerate(part_targets):
            targets[i, input_lens[i]+1:input_lens[i]+len(target)+1] = target  # plus 1 for bos
        # print("inputs_embeds: ", inputs_embeds.shape)
        # print("attention_mask", attention_mask.shape)
        # print("targets: ", targets.shape)
        with self.maybe_autocast():
            outputs = self.llama_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets,
                reduction=reduction
            )
        loss = outputs.loss

        return {"loss": loss}

    @torch.no_grad()
    def generate(
        self,
        images,
        texts,
        num_beams=1,
        max_new_tokens=20,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1,
        length_penalty=1,
        temperature=1,
        do_sample=False,
        stop_words_ids=[2],
    ):
        '''
            function for generate test use
        '''

        stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(
            stops=[torch.tensor([i]).to(self.device) for i in stop_words_ids])])

        img_embeds, atts_img = self.encode_img(images.to(self.device))
        image_lists = [[image_emb[None]] for image_emb in img_embeds]
        # print("Texts: ", texts)
        batch_embs = [self.get_context_emb(text, img_list) for text, img_list in zip(texts, image_lists)]

        batch_size = len(batch_embs)
        max_len = max([emb.shape[1] for emb in batch_embs])
        emb_dim = batch_embs[0].shape[2]
        dtype = batch_embs[0].dtype
        device = batch_embs[0].device
        # print("batch_embs", batch_embs)
        embs = torch.zeros([batch_size, max_len, emb_dim], dtype=dtype, device=device)
        attn_mask = torch.zeros([batch_size, max_len], dtype=torch.int, device=device)
        for i, emb in enumerate(batch_embs):
            emb_len = emb.shape[1]
            embs[i, -emb_len:] = emb[0]
            attn_mask[i, -emb_len:] = 1
        # print("Llama generate embed:", embs.shape)
        # print("Llama generate attn_mask:", attn_mask.shape)
        # print("Llama token mask", max_new_tokens)
        with self.maybe_autocast():
            outputs = self.llama_model.generate(
                inputs_embeds=embs,
                attention_mask=attn_mask,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                length_penalty=length_penalty,
                temperature=temperature,
                do_sample=do_sample,
                min_length=min_length,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                # stopping_criteria=stopping_criteria,
            )

        # with self.maybe_autocast():
        #     outputs = self.llama_model.generate(
        #         inputs_embeds=embs,
        #         attention_mask=attn_mask,
        #         max_new_tokens=max_new_tokens,
        #         num_beams=num_beams,
        #         do_sample=do_sample,
        #         # stopping_criteria=stopping_criteria,
        #     )
        answers = []
        # print("Answers:", answers)
        for output_token in outputs:
            if output_token[0] == 0:
                output_token = output_token[1:]
            output_texts = self.llama_tokenizer.decode(output_token, skip_special_tokens=True)
            output_texts = output_texts.split('</s>')[0]  # remove the stop sign </s>
            output_texts = output_texts.replace("<s>", "")
            output_texts = output_texts.split(r'[/INST]')[-1].strip()
            answers.append(output_texts)

        return answers

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""
    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)

def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self