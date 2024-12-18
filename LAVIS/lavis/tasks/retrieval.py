"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import json
import logging
import os
import csv

import numpy as np
import torch
from lavis.common.dist_utils import is_main_process
from lavis.common.registry import registry
from lavis.tasks.base_task import BaseTask


@registry.register_task("retrieval")
class RetrievalTask(BaseTask):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

    @classmethod
    def setup_task(cls, cfg):
        run_cfg = cfg.run_cfg

        return cls(cfg=run_cfg)

    def evaluation(self, model, data_loader, **kwargs):
        # score_i2t, score_t2i = model.compute_sim_matrix(model, data_loader)
        score_i2t, score_t2i, result = model.compute_sim_matrix(data_loader, task_cfg=self.cfg)

        if is_main_process():
            eval_result = { "retrieval": self._report_metrics(
                score_i2t,
                score_t2i,
                data_loader.dataset.txt2img,
                data_loader.dataset.img2txt,
                result,
            )}
            if result is not None:
                targets = torch.tensor([item for sublist in data_loader.dataset.label for item in sublist])
                results = torch.tensor([item for sublist in result for item in sublist])

                eval_result["classification"] = {"good_class": len(targets[targets==results]), 
                                                "bad_class": len(targets[targets!=results]), 
                                                "all_class": len(results), 
                                                "class_1_targetv": torch.sum(targets == 1).item(),
                                                "class_0_targetc": torch.sum(targets == 0).item(),
                                                "class_1_resultv": torch.sum(results == 1).item(),
                                                "class_0_resultc": torch.sum(results == 0).item(),
                                                }
            logging.info(eval_result)
        else:
            eval_result = None

        return eval_result["retrieval"]

    def after_evaluation(self, val_result, **kwargs):
        return val_result

    @staticmethod
    @torch.no_grad()
    def _report_metrics(scores_i2t, scores_t2i, txt2img, img2txt, result=None):

        with open('donnees2.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(["img2txt"])
            writer.writerows([img2txt])
            writer.writerows(["txt2img"])
            writer.writerows([txt2img])

        #score_i2t=score_i2t[result.squeeze(1) ==1]

        # Images->Text
        ranks = np.zeros(scores_i2t.shape[0])
        if result is not None:
            img2txtTest = {key: [val for val in values if result[val] == 1] for key, values in img2txt.items()}
            img2txt={}
            counter=0
            for key, values in img2txtTest.items():
                img2txt[key] = list(range(counter, counter + len(values)))
                counter += len(values)

        for index, score in enumerate(scores_i2t):
            inds = np.argsort(score)[::-1]
            # Score
            rank = 1e20
            for i in img2txt[index]:
                tmp = np.where(inds == i)[0][0]
                if tmp < rank:
                    rank = tmp
            ranks[index] = rank

        # Compute metrics
        tr1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
        tr5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
        tr10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)

        # Text->Images
        ranks = np.zeros(scores_t2i.shape[0])

        for index, score in enumerate(scores_t2i):
            inds = np.argsort(score)[::-1]
            ranks[index] = np.where(inds == txt2img[index])[0][0]

        # Compute metrics
        ir1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
        ir5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
        ir10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)

        tr_mean = (tr1 + tr5 + tr10) / 3
        ir_mean = (ir1 + ir5 + ir10) / 3
        r_mean = (tr_mean + ir_mean) / 2

        agg_metrics = (tr1 + tr5 + tr10) / 3

        eval_result = {
            "txt_r1": tr1,
            "txt_r5": tr5,
            "txt_r10": tr10,
            "txt_r_mean": tr_mean,
            "img_r1": ir1,
            "img_r5": ir5,
            "img_r10": ir10,
            "img_r_mean": ir_mean,
            "r_mean": r_mean,
            "agg_metrics": agg_metrics,
        }
        with open(
            os.path.join(registry.get_path("output_dir"), "evaluate.txt"), "a"
        ) as f:
            f.write(json.dumps(eval_result) + "\n")
        return eval_result
