import datetime
import json,re
import logging
import os
import time
import torch
import numpy as np
import torch.nn as nn
from pathlib import Path

from torch.nn.parallel import DistributedDataParallel as DDP
from process_block.Dataset.coco_vqa_dataset import COCOVQADataset, COCOVQAEvalDataset
from torch.utils.data import DataLoader
from urllib.parse import urlparse

from process_block.common.dist_utils import (
                                    download_cached_file,
                                    get_rank,
                                    get_world_size,
                                    is_main_process,
                                    main_process,
                                    )
from process_block.common.logger import SmoothedValue, MetricLogger

class Task:

    def __init__(self, model, data_loader,optimizer,lr_scheduler, model_conf=dict):
        super().__init__()
        self._model = model
        self.model_conf = model_conf
        self.save_path = os.path.join(model_conf['output_dir'])
        os.makedirs(self.save_path, exist_ok=True)
        self._data_loader = data_loader
        self._optimizer = optimizer
        self._lr_scheduler = lr_scheduler
        self.start_epoch = 0
        self.resume_ckpt_path=None
        self.use_distributed = model_conf['use_distributed']
        if self.model_conf['amp']:
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None
        self.evaluate_only = False
        if model_conf['cuda_enabled']:
            self._device = "cuda"
        else:
            self._device = "cpu"
    def unwrap_dist_model(self, model):
        if self.use_distributed:
            return model.module
        else:
            return model
    def is_url(self,url_or_filename):
        parsed = urlparse(url_or_filename)
        return parsed.scheme in ("http", "https")

    def _load_checkpoint(self, url_or_filename):
        """
        Resume from a checkpoint.
        """
        if self.is_url(url_or_filename):
            cached_file = download_cached_file(
                url_or_filename, check_hash=False, progress=True
            )
            checkpoint = torch.load(cached_file, map_location=self.device)

        elif os.path.isfile(url_or_filename):
            checkpoint = torch.load(url_or_filename, map_location=self.device)

        else:
            raise RuntimeError("checkpoint url or path is invalid")

        state_dict = checkpoint["model"]
        # breakpoint()

        message = self.unwrap_dist_model(self.model).load_state_dict(state_dict,strict=False)
        
        self._optimizer.load_state_dict(checkpoint["optimizer"])
        if self.scaler and "scaler" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler"])

        self.start_epoch = checkpoint["epoch"] + 1
        print("resume the checkpoint")
        print("Resume checkpoint from {}".format(url_or_filename))
        


    def _save_checkpoint(self, cur_epoch, is_best=False):
        """
        Save the checkpoint at the current epoch.
        """
        model_no_ddp = self.unwrap_dist_model(self.model)
        param_grad_dic = {
            k: v.requires_grad for (k, v) in model_no_ddp.named_parameters()
        }
        state_dict = model_no_ddp.state_dict()
        for k in list(state_dict.keys()):
            if k in param_grad_dic.keys() and not param_grad_dic[k]:
                # delete parameters that do not require gradient
                del state_dict[k]
        save_obj = {
            "model": state_dict,
            "optimizer": self._optimizer.state_dict(),
            "config": {
                        'model_config':self.model_conf
                        },
            "scaler": self.scaler.state_dict() if self.scaler else None,
            "epoch": cur_epoch,
        }
        save_to = os.path.join(
            self.save_path,
            "checkpoint_{}.pth".format("best" if is_best else cur_epoch),
        )
        print("Saving checkpoint at epoch {} to {}.".format(cur_epoch, save_to))
        torch.save(save_obj, save_to)

    def _reload_best_model(self, model):
        """
        Load the best checkpoint for evaluation.
        """
        checkpoint_path = os.path.join(self.save_path, "checkpoint_best.pth")

        print("Loading checkpoint from {}.".format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        try:
            model.load_state_dict(checkpoint["model"])
        except RuntimeError as e:
            print(
                """
                Key mismatch when loading checkpoint. This is expected if only part of the model is saved.
                Trying to load the model with strict=False.
                """
            )
            model.load_state_dict(checkpoint["model"], strict=False)
        return model

    @property
    def device(self):
        if self._device is None:
            self._device = torch.device('cpu')

        return self._device

    @property
    def model(self):
        """
        A property to get the DDP-wrapped model on the device.
        """
        # move model to device
        if self._model.device != self.device:
            self._model = self._model.to(self.device)

            # distributed training wrapper
            if self.use_distributed:
                if self._wrapped_model is None:
                    self._wrapped_model = DDP(
                        self._model, device_ids=0, find_unused_parameters=True
                    )
            else:
                self._wrapped_model = self._model

        return self._wrapped_model

    def train_step(self, model, samples):
        loss = model(samples)["loss"]
        return loss

    def _train_inner_loop(
        self,
        epoch,
        iters_per_epoch,
        model,
        data_loader,
        optimizer,
        lr_scheduler,
        scaler=None,
        start_iters=None,
        log_freq=100,
        cuda_enabled=False,
        accum_grad_iters=1,
    ):
        """
        An inner training loop compatible with both epoch-based and iter-based training.

        When using epoch-based, training stops after one epoch; when using iter-based,
        training stops after #iters_per_epoch iterations.
        """
        use_amp = scaler is not None

        if not hasattr(data_loader, "__next__"):
            # convert to iterator if not already
            data_loader = iter(data_loader)

        metric_logger = MetricLogger(delimiter="  ")
        metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
        metric_logger.add_meter("loss", SmoothedValue(window_size=1, fmt="{value:.4f}"))

        # if iter-based runner, schedule lr based on inner epoch.
        logging.info(
            "Start training epoch {}, {} iters per inner epoch.".format(
                epoch, iters_per_epoch
            )
        )
        header = "Train: data epoch: [{}]".format(epoch)
        if start_iters is None:
            # epoch-based runner
            inner_epoch = epoch
        else:
            # In iter-based runner, we schedule the learning rate based on iterations.
            inner_epoch = start_iters // iters_per_epoch
            header = header + "; inner epoch [{}]".format(inner_epoch)

        for i in metric_logger.log_every(range(iters_per_epoch), log_freq, header):
            # if using iter-based runner, we stop after iters_per_epoch iterations.
            if i >= iters_per_epoch:
                break

            samples = next(data_loader)
            samples = prepare_sample(samples, cuda_enabled=cuda_enabled)

            samples.update(
                {
                    "epoch": inner_epoch,
                    "num_iters_per_epoch": iters_per_epoch,
                    "iters": i,
                }
            )

            lr_scheduler.step(cur_epoch=inner_epoch, cur_step=i)

            with torch.cuda.amp.autocast(enabled=use_amp):
                loss = self.train_step(model=model, samples=samples)

            # after_train_step()
            if use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            # update gradients every accum_grad_iters iterations
            if (i + 1) % accum_grad_iters == 0:
                if use_amp:
                    scaler.step(optimizer)
                    scaler.update()                     
                else:    
                    optimizer.step()
                optimizer.zero_grad()
                # if self.cfg.wandb_log:
                # if self.cfg.run_cfg.wandb_log:
                #     wandb.log({"epoch": inner_epoch, "loss": loss})
            metric_logger.update(loss=loss.item())
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        # after train_epoch()
        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        print("Averaged stats: " + str(metric_logger.global_avg()))
        return {
            k: "{:.3f}".format(meter.global_avg)
            for k, meter in metric_logger.meters.items()
        }
    def train_epoch(
        self,
        epoch,
        model,
        data_loader,
        optimizer,
        lr_scheduler,
        scaler=None,
        cuda_enabled=False,
        log_freq=50,
        accum_grad_iters=1,
    ):
        return self._train_inner_loop(
            epoch=epoch,
            iters_per_epoch=lr_scheduler.iters_per_epoch,
            model=model,
            data_loader=data_loader,
            optimizer=optimizer,
            scaler=scaler,
            lr_scheduler=lr_scheduler,
            log_freq=log_freq,
            cuda_enabled=cuda_enabled,
            accum_grad_iters=accum_grad_iters,
        )

    def before_evaluation(self, **kwargs):
        pass
    def after_evaluation(self, **kwargs):
        pass

    def valid_step(self, model, samples):
        raise NotImplementedError

    @property
    def valid_splits(self):
        valid_splits = []

        if len(valid_splits) == 0:
            print("No validation splits found.")

        return valid_splits

    @property
    def test_splits(self):
        test_splits = []

        return test_splits

    def evaluation(self, model, data_loader, cuda_enabled=True):
        metric_logger = MetricLogger(delimiter="  ")
        header = "Evaluation"
        # TODO make it configurable
        print_freq = 10

        results = []

        for samples in metric_logger.log_every(data_loader, print_freq, header):
            samples = prepare_sample(samples, cuda_enabled=cuda_enabled)

            eval_output = self.valid_step(model=model, samples=samples)
            results.extend(eval_output)

        if is_dist_avail_and_initialized():
            dist.barrier()

        return results

    @torch.no_grad()
    def eval_epoch(self, split_name, cur_epoch, skip_reload=False):
        """
        Evaluate the model on a given split.

        Args:
            split_name (str): name of the split to evaluate on.
            cur_epoch (int): current epoch.
            skip_reload_best (bool): whether to skip reloading the best checkpoint.
                During training, we will reload the best checkpoint for validation.
                During testing, we will use provided weights and skip reloading the best checkpoint .
        """
        data_loader = self._data_loader.get(split_name, None)
        assert data_loader, "data_loader for split {} is None.".format(split_name)

        # TODO In validation, you need to compute loss as well as metrics
        # TODO consider moving to model.before_evaluation()
        model = self.unwrap_dist_model(self.model)
        if not skip_reload and cur_epoch == "best":
            model = self._reload_best_model(model)
        model.eval()

        self.before_evaluation(
            model=model,
            dataset=self.datasets[split_name],
        )
        results = self.evaluation(model, data_loader)

        if results is not None:
            return self.after_evaluation(
                val_result=results,
                split_name=split_name,
                epoch=cur_epoch,
            )

    def evaluate(self, cur_epoch="best", skip_reload=False):
        test_logs = dict()

        if len(self.test_splits) > 0:
            for split_name in self.test_splits:
                test_logs[split_name] = self.eval_epoch(
                    split_name=split_name, cur_epoch=cur_epoch, skip_reload=skip_reload
                )

            return test_logs

    def train(self):
        start_time = time.time()
        best_agg_metric = 0
        best_epoch = 0

        self.log_config()##紀錄training過程

        # resume from checkpoint if specified

        if not self.evaluate_only and self.resume_ckpt_path is not None:
            self._load_checkpoint(self.resume_ckpt_path)
            print('====use last time weight====')
        print('====Training set====')
        print('====Epoch:{0:05d}===='.format(self.model_conf['max_epoch']))
        print('====Bacth size:{0:04}===='.format(self.model_conf['batch_size']))
        # print(f"==[Total] Training {self._data_loader.__len__()} Samples==")
        for cur_epoch in range(self.start_epoch, self.model_conf['max_epoch']):
            # training phase
            if not self.evaluate_only:
                print("=====Start training====")
                train_stats = self.train_epoch(
                                            epoch = cur_epoch,
                                            model = self.model,
                                            data_loader = self._data_loader,
                                            optimizer=self._optimizer,
                                            lr_scheduler= self._lr_scheduler,
                                            scaler = self.scaler,
                                            cuda_enabled = self.device
                                            )
                self.log_stats(split_name="train", stats=train_stats)
                print("=====END training====")
                # breakpoint()
            ####原始code
            # if not self.evaluate_only:
                # logging.info("Start training")
                # train_stats = self.train_epoch(cur_epoch)
                # self.log_stats(split_name="train", stats=train_stats)
            ####
            # evaluation phase

            if len(self.valid_splits) > 0:
                print("====Strat evaluating====")
                for split_name in self.valid_splits:
                    val_log = self.eval_epoch(
                        split_name=split_name, cur_epoch=cur_epoch)
                    if val_log is not None:
                        if is_main_process():
                            assert (
                                "agg_metrics" in val_log
                            ), "No agg_metrics found in validation log."

                            agg_metrics = val_log["agg_metrics"]
                            if agg_metrics > best_agg_metric and split_name == "val":
                                best_epoch, best_agg_metric = cur_epoch, agg_metrics

                                self._save_checkpoint(cur_epoch, is_best=True)

                            val_log.update({"best_epoch": best_epoch})
                            self.log_stats(val_log, split_name)
                print("====END evaluating====")
            else:
                # if no validation split is provided, we just save the checkpoint at the end of each epoch.
                if not self.evaluate_only:
                    self._save_checkpoint(cur_epoch, is_best=False)

            if self.use_distributed:
                dist.barrier()
            ####原始
            # if len(self.valid_splits) > 0:
            #     for split_name in self.valid_splits:
            #         logging.info("Evaluating on {}.".format(split_name))

            #         val_log = self.eval_epoch(
            #             split_name=split_name, cur_epoch=cur_epoch
            #         )
            #         if val_log is not None:
            #             if is_main_process():
            #                 assert (
            #                     "agg_metrics" in val_log
            #                 ), "No agg_metrics found in validation log."

            #                 agg_metrics = val_log["agg_metrics"]
            #                 if agg_metrics > best_agg_metric and split_name == "val":
            #                     best_epoch, best_agg_metric = cur_epoch, agg_metrics

            #                     self._save_checkpoint(cur_epoch, is_best=True)

            #                 val_log.update({"best_epoch": best_epoch})
            #                 self.log_stats(val_log, split_name)

            # else:
            #     # if no validation split is provided, we just save the checkpoint at the end of each epoch.
            #     if not self.evaluate_only:
            #         self._save_checkpoint(cur_epoch, is_best=False)

            # if self.evaluate_only:
            #     break

            # if self.config.run_cfg.distributed:
            #     dist.barrier()
            ####

        # testing phase

        test_epoch = "best" if len(self.valid_splits) > 0 else cur_epoch
        self.evaluate(cur_epoch=test_epoch, skip_reload=self.evaluate_only)

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print("Training time {}".format(total_time_str))
        ####原始code
        # test_epoch = "best" if len(self.valid_splits) > 0 else cur_epoch
        # self.evaluate(cur_epoch=test_epoch, skip_reload=self.evaluate_only)

        # total_time = time.time() - start_time
        # total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        # logging.info("Training time {}".format(total_time_str))
        ####
    @main_process
    def log_stats(self, stats, split_name):
        if isinstance(stats, dict):
            log_stats = {**{f"{split_name}_{k}": v for k, v in stats.items()}}
            with open(os.path.join(self.save_path, "log.txt"), "a") as f:
                f.write(json.dumps(log_stats) + "\n")
        elif isinstance(stats, list):
            pass

    @main_process
    def log_config(self):
        with open(os.path.join(self.save_path, "log.txt"), "a") as f:
            f.write(json.dumps([], indent=4) + "\n")

def apply_to_sample(f, sample):
    if len(sample) == 0:
        return {}

    def _apply(x):
        if torch.is_tensor(x):
            return f(x)
        elif isinstance(x, dict):
            return {key: _apply(value) for key, value in x.items()}
        elif isinstance(x, list):
            return [_apply(x) for x in x]
        else:
            return x

    return _apply(sample)

def move_to_cuda(sample):
        def _move_to_cuda(tensor):
            return tensor.cuda()

        return apply_to_sample(_move_to_cuda, sample)

def prepare_sample(samples, cuda_enabled=True):
        if cuda_enabled:
            samples = move_to_cuda(samples)

        # TODO fp16 support

        return samples