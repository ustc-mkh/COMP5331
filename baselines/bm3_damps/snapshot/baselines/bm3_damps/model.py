"""Trainable DAMPS on BM3's projected modalities; original scoring/loss retained."""
from pathlib import Path
import sys
import torch
from torch.nn import functional as F
from torch.nn.functional import cosine_similarity

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "bm3/runtime"))
from bm3baseline.model import BM3 as RawBM3
from damps import DAMPS

VARIANTS = {"full": (True, True, True), "no_apc": (False, True, True),
            "no_avrf": (True, False, True), "no_imcf": (True, True, False)}


class BM3Damps(RawBM3):
    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        self.variant = config.get("damps_variant", "baseline")
        if self.variant == "baseline":
            self.damps = None
            return
        apc, avrf, imcf = VARIANTS[self.variant]
        self.damps = DAMPS(self.feat_embed_dim, apc=apc, avrf=avrf, imcf=imcf).to(self.device)
        # BM3 creates projections on CPU and moves the model after construction.
        # Estimate fixed priors on the same device as those initial projections.
        self.damps.to(self.image_trs.weight.device)
        with torch.no_grad():
            image = self.image_trs(self.image_embedding.weight.to(self.image_trs.weight.device))
            text = self.text_trs(self.text_embedding.weight.to(self.text_trs.weight.device))
            self.damps.initialize(image, text)

    def calculate_loss(self, interactions):
        if self.damps is None:
            return super().calculate_loss(interactions)
        # Exact upstream objective; insert DAMPS before creating dropout targets.
        u_online_ori, i_online_ori = self.forward()
        t_feat_online = self.text_trs(self.text_embedding.weight)
        v_feat_online = self.image_trs(self.image_embedding.weight)
        v_feat_online, t_feat_online = self.damps(v_feat_online, t_feat_online)
        with torch.no_grad():
            u_target = F.dropout(u_online_ori.clone(), self.dropout)
            i_target = F.dropout(i_online_ori.clone(), self.dropout)
            t_feat_target = F.dropout(t_feat_online.clone(), self.dropout)
            v_feat_target = F.dropout(v_feat_online.clone(), self.dropout)
        u_online, i_online = self.predictor(u_online_ori), self.predictor(i_online_ori)
        users, items = interactions[0], interactions[1]
        u_online, i_online = u_online[users, :], i_online[items, :]
        u_target, i_target = u_target[users, :], i_target[items, :]
        t_feat_online = self.predictor(t_feat_online)[items, :]
        v_feat_online = self.predictor(v_feat_online)[items, :]
        t_feat_target, v_feat_target = t_feat_target[items, :], v_feat_target[items, :]
        loss_t = 1 - cosine_similarity(t_feat_online, i_target.detach(), dim=-1).mean()
        loss_tv = 1 - cosine_similarity(t_feat_online, t_feat_target.detach(), dim=-1).mean()
        loss_v = 1 - cosine_similarity(v_feat_online, i_target.detach(), dim=-1).mean()
        loss_vt = 1 - cosine_similarity(v_feat_online, v_feat_target.detach(), dim=-1).mean()
        loss_ui = 1 - cosine_similarity(u_online, i_target.detach(), dim=-1).mean()
        loss_iu = 1 - cosine_similarity(i_online, u_target.detach(), dim=-1).mean()
        return ((loss_ui + loss_iu).mean() + self.reg_weight * self.reg_loss(u_online_ori, i_online_ori)
                + self.cl_weight * (loss_t + loss_v + loss_tv + loss_vt).mean())
