"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import logging
import torch
import torch.distributed.nn
from torch import distributed as dist, nn as nn
from torch.nn import functional as F
import json
import os

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None


def gather_features(
    image_features,
    text_features,
    local_loss=False,
    gather_with_grad=False,
    rank=0,
    world_size=1,
    use_horovod=False,
):
    if use_horovod:
        assert hvd is not None, "Please install horovod"
        if gather_with_grad:
            all_image_features = hvd.allgather(image_features)
            all_text_features = hvd.allgather(text_features)
        else:
            with torch.no_grad():
                all_image_features = hvd.allgather(image_features)
                all_text_features = hvd.allgather(text_features)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_features = list(
                    all_image_features.chunk(world_size, dim=0)
                )
                gathered_text_features = list(
                    all_text_features.chunk(world_size, dim=0)
                )
                gathered_image_features[rank] = image_features
                gathered_text_features[rank] = text_features
                all_image_features = torch.cat(gathered_image_features, dim=0)
                all_text_features = torch.cat(gathered_text_features, dim=0)
    else:
        # We gather tensors from all gpus
        if gather_with_grad:
            all_image_features = torch.cat(
                torch.distributed.nn.all_gather(image_features), dim=0
            )
            all_text_features = torch.cat(
                torch.distributed.nn.all_gather(text_features), dim=0
            )
        else:
            gathered_image_features = [
                torch.zeros_like(image_features) for _ in range(world_size)
            ]
            gathered_text_features = [
                torch.zeros_like(text_features) for _ in range(world_size)
            ]
            dist.all_gather(gathered_image_features, image_features)
            dist.all_gather(gathered_text_features, text_features)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_features[rank] = image_features
                gathered_text_features[rank] = text_features
            all_image_features = torch.cat(gathered_image_features, dim=0)
            all_text_features = torch.cat(gathered_text_features, dim=0)

    return all_image_features, all_text_features


class ClipLoss(nn.Module):
    def __init__(
        self,
        local_loss=False,
        gather_with_grad=False,
        cache_labels=False,
        rank=0,
        world_size=1,
        use_horovod=False,
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod

        self.logits_per_text=None
        self.logits_per_image=None
        self.result=None

        # cache state
        self.prev_num_logits = 0
        self.labels = {}

    def forward(self, image_features, text_features, logit_scale, text_classif_features=None, targets=None, mode=None):
        device = image_features.device
        if self.world_size > 1:
            all_image_features, all_text_features = gather_features(
                image_features,
                text_features,
                self.local_loss,
                self.gather_with_grad,
                self.rank,
                self.world_size,
                self.use_horovod,
            )

            if self.local_loss:
                logits_per_image = logit_scale * image_features @ all_text_features.T
                logits_per_text = logit_scale * text_features @ all_image_features.T
            else:
                logits_per_image = (
                    logit_scale * all_image_features @ all_text_features.T
                )
                logits_per_text = logits_per_image.T
        else:
            logits_per_image = logit_scale * image_features @ text_features.T
            logits_per_text = logit_scale * text_features @ image_features.T

        self.logits_per_text= logits_per_text
        self.logits_per_image= logits_per_image

        # calculated ground-truth and cache if enabled
        num_logits = logits_per_image.shape[0]
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.world_size > 1 and self.local_loss:
                labels = labels + num_logits * self.rank
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]
       
        total_loss = (
            F.cross_entropy(logits_per_image, labels)
            + F.cross_entropy(logits_per_text, labels)
        ) / 2

        if text_classif_features!=None and targets!=None and False:
            targets=targets[targets!=2].unsqueeze(1)
            result = torch.sigmoid(text_classif_features)
            result =  (result >= 0.5).long()
            print("bien classif:", len(targets[targets==result]), "mal classif:", len(targets[targets!=result]), "total:", len(targets))

            classif_loss = F.binary_cross_entropy_with_logits(text_classif_features.to('cuda:0'), targets.float().to('cuda:0'), pos_weight=torch.tensor(1))

            if mode:
                self.result = result.squeeze(1) ==1

                self.logits_per_image = logit_scale * image_features[self.result] @ text_features[self.result].T
                self.logits_per_text = logit_scale * text_features[self.result] @ image_features[self.result].T
            total_loss = (total_loss+classif_loss)/2
        return total_loss

class ClipLoss_classRet(nn.Module):
    def __init__(
        self,
        classifier,
        local_loss=False,
        gather_with_grad=False,
        cache_labels=False,
        rank=0,
        world_size=1,
        use_horovod=False,
        hidden_size=512, 
        num_classes=2
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod

        # cache state
        self.prev_num_logits = 0
        self.labels = {}

        self.classifier = classifier

    def forward(self, image_features, text_features, logit_scale, alpha,
        text_embeds, targets, training=True):
        device = image_features.device

        #print("class", text_embeds.size())

        logits_classif = self.classifier(text_embeds.to('cuda:0')).to('cuda:0')#ne jamais faire de grosse modif sur les logits: couche de classif ne s'entraine pas sinon
        logits_classif= logits_classif[:,:,1]#pour récupérer que la partie valant 1 (si plus que 0.5 alors on est plus proche de la classe 1 sinon de 0)
        #print("class", logits_classif.size(), logits_classif)

        bestTxt = logits_classif[:, 1] 

        if self.world_size > 1:
            all_image_features, all_text_features = gather_features(
                image_features,
                text_features,
                self.local_loss,
                self.gather_with_grad,
                self.rank,
                self.world_size,
                self.use_horovod,
            )

            if self.local_loss:
                logits_per_image = (logit_scale * (image_features @ all_text_features.T))
                logits_per_text = logit_scale * text_features @ all_image_features.T#*bestTxt.unsqueeze(0)
            else:
                logits_per_image = (
                    logit_scale * all_image_features @ all_text_features.T
                )
                logits_per_text = logits_per_image.T
        else:
            logits_per_image = (logit_scale * (image_features @ text_features.T))
            logits_per_text = logit_scale * text_features @ image_features.T #* bestTxt.unsqueeze(0)

        #print("loss", image_features.size(), text_features.size())
        #print("after combine", logits_per_image.size(), logits_per_text.size())

  
        # calculated ground-truth and cache if enabled
        num_logits = logits_per_image.shape[0]
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.world_size > 1 and self.local_loss:
                labels = labels + num_logits * self.rank
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device] 

        loss_retrieval = F.cross_entropy(logits_per_text, labels)
        loss_retrieval += F.cross_entropy(logits_per_image, labels)
        #print(min(bestTxt), max(bestTxt), alpha, bestTxt.size(), bestTxt[bestTxt>0.5].size(), targets.size())
        #if training==True:
        #    loss_classif = F.binary_cross_entropy_with_logits(bestTxt.to('cuda:0'), targets.float().to('cuda:0'), pos_weight=torch.tensor(1.7))#targets[targets==0].sum()/targets[targets==1].sum()))#torch.ones_like(bestTxt).to('cuda:0'))
            #loss = (alpha*loss_classif+loss_retrieval)/3

        #if training==False:
        #    loss_classif = F.binary_cross_entropy_with_logits(bestTxt.to('cuda:0'), torch.ones_like(bestTxt).to('cuda:0'))#, pos_weight=torch.tensor(targets[targets==0].sum()/targets[targets==1].sum()))#torch.ones_like(bestTxt).to('cuda:0'))
            #loss=loss_retrieval/2  
        #if alpha==10:
        #    loss = loss_classif
        #else:
        #    loss = (loss_classif+loss_retrieval)/3

        loss = (loss_retrieval)/2

        return loss

class ClipLoss_classif(nn.Module):
    def __init__(
        self,
        classifier,
        local_loss=False,
        gather_with_grad=False,
        cache_labels=False,
        rank=0,
        world_size=1,
        use_horovod=False,
        hidden_size=512, 
        num_classes=2
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod

        # cache state
        self.prev_num_logits = 0
        self.labels = {}

        self.classifier = classifier

    def forward(self, text_embeds, targets, train=False):
        device = text_embeds.device

        logits_classif = self.classifier(text_embeds.to('cuda:0')).to('cuda:0')#ne jamais faire de grosse modif sur les logits: couche de classif ne s'entraine pas sinon

        loss = F.binary_cross_entropy_with_logits(logits_classif[(targets != 2)].to('cuda:0'), targets[(targets != 2)].float().to('cuda:0'), pos_weight=torch.tensor(100))
        
        #if train:
        #    self.append_to_json_file("jsonFile", newdata)
        return loss


    def append_to_json_file(self,file_path, new_data):
        if os.path.exists(file_path):
            with open(file_path, 'r') as file:
                try:
                    existing_data = json.load(file)
                except json.JSONDecodeError:
                    existing_data = []
        else:
            existing_data = []
        if isinstance(existing_data, list):
            existing_data.extend(new_data)
        else:
            raise ValueError("Le fichier JSON existant ne contient pas une liste.")

        with open(file_path, 'w') as file:
            json.dump(existing_data, file, indent=4, ensure_ascii=False)
