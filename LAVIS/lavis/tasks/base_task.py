"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import logging
import os

import torch
import numpy as np
from PIL import Image
import cv2

import torch.distributed as dist
from lavis.common.dist_utils import get_rank, get_world_size, is_main_process, is_dist_avail_and_initialized
from lavis.common.logger import MetricLogger, SmoothedValue
from lavis.common.registry import registry
from lavis.datasets.data_utils import prepare_sample


class BaseTask:
    def __init__(self, **kwargs):
        super().__init__()

        self.inst_id_key = "instance_id"
        ### start gradcam
        self.target = None
        self.feature = None  # To store the features from the target layer
        self.gradient = None  # To store the gradients from the target layer
        self.handlers = []  # List to keep track of hooks

    # Hook to get features from the forward pass
    def _get_features_hook(self, module, input, output):
        print(output.shape, "oiuyt")
        self.feature = self.reshape_transform(output)  # Store and reshape the output features

    # Hook to get gradients from the backward pass
    def _get_grads_hook(self, module, input_grad, output_grad):
        print(output_grad.max(), "output_grad")
        self.gradient = self.reshape_transform(output_grad)  # Store and reshape the output gradients
        print("\\\\\\\\\\\\self.gradient", self.gradient[0].max(), self.gradient.shape)
        def _store_grad(grad):
            self.gradient = self.reshape_transform(grad)  # Store gradients for later use
            print("\\\\\\\\\\\\self.gradientStore", self.gradient[0].max(), self.gradient.shape)
        

        #output_grad.register_hook(_store_grad)  # Register hook to store gradients

    # Register forward hooks to the target layer
    def _get_hook(self):
        self.target.register_forward_hook(self._get_features_hook)
        self.target.register_forward_hook(self._get_grads_hook)

    # Function to reshape the tensor for visualization
    def reshape_transform(self, tensor, height=24, width=24):
        tensor = tensor.permute(1, 0, 2)
        result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
        result = result.transpose(2, 3).transpose(1, 2)  # Rearrange dimensions to (C, H, W)
        return result

    ### en gradCam

    @classmethod
    def setup_task(cls, **kwargs):
        return cls()

    def build_model(self, cfg):
        model_config = cfg.model_cfg

        model_cls = registry.get_model_class(model_config.arch)
        return model_cls.from_config(model_config)

    def build_datasets(self, cfg):
        """
        Build a dictionary of datasets, keyed by split 'train', 'valid', 'test'.
        Download dataset and annotations automatically if not exist.

        Args:
            cfg (common.config.Config): _description_

        Returns:
            dict: Dictionary of torch.utils.data.Dataset objects by split.
        """

        datasets = dict()

        datasets_config = cfg.datasets_cfg

        assert len(datasets_config) > 0, "At least one dataset has to be specified."

        for name in datasets_config:
            dataset_config = datasets_config[name]

            builder = registry.get_builder_class(name)(dataset_config)
            dataset = builder.build_datasets()
            datasets[name] = dataset

        return datasets

    def train_step(self, model, samples):
        output = model(samples)
        output.intermediate_output.image_embeds_proj.retain_grad()
        loss = output.loss
        return output#loss

    def valid_step(self, model, samples):
        raise NotImplementedError

    def before_evaluation(self, model, dataset, **kwargs):
        model.before_evaluation(dataset=dataset, task_type=type(self))

    def after_evaluation(self, **kwargs):
        pass

    def inference_step(self):
        raise NotImplementedError

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
            iters_per_epoch=len(data_loader),
            model=model,
            data_loader=data_loader,
            optimizer=optimizer,
            scaler=scaler,
            lr_scheduler=lr_scheduler,
            log_freq=log_freq,
            cuda_enabled=cuda_enabled,
            accum_grad_iters=accum_grad_iters,
        )

    def train_iters(
        self,
        epoch,
        start_iters,
        iters_per_inner_epoch,
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
            start_iters=start_iters,
            iters_per_epoch=iters_per_inner_epoch,
            model=model,
            data_loader=data_loader,
            optimizer=optimizer,
            scaler=scaler,
            lr_scheduler=lr_scheduler,
            log_freq=log_freq,
            cuda_enabled=cuda_enabled,
            accum_grad_iters=accum_grad_iters,
        )

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
        log_freq=50,
        cuda_enabled=False,
        accum_grad_iters=1,
        gradCam = False
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
        samples = next(data_loader)
        samples = prepare_sample(samples, cuda_enabled=cuda_enabled)

        if gradCam:
            for index in range (0,len(samples.get("image"))):
                self.target = model.visual.transformer.resblocks[-1].ln_2
                self._get_hook()
                model.zero_grad()
                with torch.cuda.amp.autocast(enabled=use_amp):
                    output = self.train_step(model=model, samples=samples)
                loss = output.loss
                #index = np.argmax(output.loss[0].cpu().data.numpy())
                target = output.loss#[1][index]  # Get the target score
                print(output.intermediate_output.logits_image, output.intermediate_output.logits_text, "/////////////////////")
                target.backward()

                # Get the gradients and features
                gradient = self.gradient[index].cpu().data.numpy()
                print("grad", gradient.max(), "768,14,14")
                weight = np.mean(gradient, axis=(1, 2))  # Average the gradients
                print("weight", weight.max(), "768")
                feature = self.feature[index].cpu().data.numpy()
                print("feature", feature.max(), "768,14,14")

                # Compute the weighted sum of the features
                cam = feature * weight[:, np.newaxis, np.newaxis]
                print("cam", cam.max(), "768,14,14")
                cam = np.sum(cam, axis=0)  # Sum over the channels
                print("cam", cam.max(), "14,14")
                cam = np.maximum(cam, 0)  # Apply ReLU to remove negative values
                print("cam", cam.max(), "14,14")

                # Normalize the heatmap
                cam -= np.min(cam)
                print("cam", cam.max(), "14,14")
                cam /= np.max(cam)
                print("cam", cam.max(), "14,14")
                cam *= 255
                print("cam", cam.max(), "14,14")
                cam_resized = cv2.resize(cam, (336, 336), interpolation=cv2.INTER_LINEAR)
                cam_rgb = np.repeat(cam_resized[:, :, np.newaxis], 3, axis=2).astype(np.uint8)

                # Appliquer un flou gaussien avec un noyau de taille (15, 15)
                cam_blurred = cv2.GaussianBlur(cam_rgb, (11,11), sigmaX=0)
                # Appliquer une carte de couleur (par exemple COLORMAP_JET) pour un effet de chaleur
                cam_colored = cv2.applyColorMap(cam_blurred, cv2.COLORMAP_JET)

                # Convertir en image PIL pour sauvegarder en tant qu'image
                attn_map = Image.fromarray(cam_colored)
                attn_map = attn_map.convert("RGBA")
                attn_map.save('cam.png')

                image= samples.get("image").cpu().numpy()[index][0]
                image /= np.max(image)
                image *= 255
                image = Image.fromarray(image).convert("RGBA")
                image.save("image.png")

                # 3. Créer un canal alpha pour la carte d'attention (transparence par défaut)
                attn_map_alpha = np.array(attn_map)
                attn_map_alpha[..., 3] = attn_map_alpha[..., 0]  # Utiliser la luminosité de la carte d'attention comme alpha

                # Mettre à jour l'image d'attention avec l'alpha ajusté
                attn_map_rgba = Image.fromarray(attn_map_alpha.astype(np.uint8))

                # 4. Superposer la carte d'attention sur l'image avec transparence
                combined_image = Image.alpha_composite(image, attn_map_rgba)

                # 5. Sauvegarder l'image combinée
                combined_image.save(str(samples.get("text_input")[0][:15]) + "_"+ str(epoch) +'image_cam.png')

        else:
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

            print(iters_per_epoch, "////////////")
            for i in metric_logger.log_every(range(iters_per_epoch), log_freq, header):
                # if using iter-based runner, we stop after iters_per_epoch iterations.
                if i >= iters_per_epoch:
                    break

                samples.update(
                    {
                        "epoch": inner_epoch,
                        "num_iters_per_epoch": iters_per_epoch,
                        "iters": i,
                    }
                )


                lr_scheduler.step(cur_epoch=inner_epoch, cur_step=i)
                with torch.cuda.amp.autocast(enabled=use_amp):
                    output = self.train_step(model=model, samples=samples)

                loss = output.loss

                # after_train_step()
                if use_amp:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                # update gradients every accum_grad_iters iterations
                if (i + 1) % accum_grad_iters == 0:
                    optimizer.step()
                    optimizer.zero_grad()

                        
                metric_logger.update(loss=loss.item())
                metric_logger.update(lr=optimizer.param_groups[0]["lr"])

            # after train_epoch()
            # gather the stats from all processes
            metric_logger.synchronize_between_processes()
            logging.info("Averaged stats: " + str(metric_logger.global_avg()))
            
            return {
                k: "{:.3f}".format(meter.global_avg)
                for k, meter in metric_logger.meters.items()
            }

    @staticmethod
    def save_result(result, result_dir, filename, remove_duplicate=""):
        import json

        result_file = os.path.join(
            result_dir, "%s_rank%d.json" % (filename, get_rank())
        )
        final_result_file = os.path.join(result_dir, "%s.json" % filename)

        json.dump(result, open(result_file, "w"))

        if is_dist_avail_and_initialized():
            dist.barrier()

        if is_main_process():
            logging.warning("rank %d starts merging results." % get_rank())
            # combine results from all processes
            result = []

            for rank in range(get_world_size()):
                result_file = os.path.join(
                    result_dir, "%s_rank%d.json" % (filename, rank)
                )
                res = json.load(open(result_file, "r"))
                result += res

            if remove_duplicate:
                result_new = []
                id_list = []
                for res in result:
                    if res[remove_duplicate] not in id_list:
                        id_list.append(res[remove_duplicate])
                        result_new.append(res)
                result = result_new

            json.dump(result, open(final_result_file, "w"))
            print("result file saved to %s" % final_result_file)

        return final_result_file
