from transformers import logging, RobertaForTokenClassification
from torchcrf import CRF

from .base import NerOutput

import torch
import torch.nn as nn
import torch.nn.functional as F

logging.set_verbosity_error()


def _disable_amp(device_type: str = 'cuda'):
    if hasattr(torch, 'amp') and hasattr(torch.amp, 'autocast'):
        return torch.amp.autocast(device_type, enabled=False)
    return torch.cuda.amp.autocast(enabled=False)


class ProjectionHead(nn.Module):
    """
    Projection Head for Supervised Contrastive Learning (CUA-MNER design)
    Projects 768-dim contextual representations to a 128-dim normalized contrastive space.
    """
    def __init__(self, input_dim=768, hidden_dim=256, output_dim=128):
        super(ProjectionHead, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.mlp(x)


class TokenSupConLoss(nn.Module):
    """
    Token-level Supervised Contrastive Loss for NER entities.
    Supports both Exact BIO Matching and Hierarchical Entity-Type Matching.
    """
    def __init__(self, temperature: float = 0.07, mode: str = 'exact', alpha: float = 0.3):
        super(TokenSupConLoss, self).__init__()
        self.temperature = temperature
        self.mode = mode
        self.alpha = alpha

    def forward(self, features, labels):
        # features: [N_entity, 128]
        # labels:   [N_entity]
        if features.size(0) < 2:
            return features.new_tensor(0.0)

        # 1. Ep kieu sang float32 de dam bao on dinh so hoc tuyet doi trong moi truong AMP
        features = features.float()
        features = F.normalize(features, p=2, dim=1)

        # 2. Cosine similarity scaled by temperature
        similarity = torch.matmul(features, features.T) / self.temperature
        N = features.size(0)

        # 3. Self-mask
        self_mask = torch.eye(N, device=features.device, dtype=torch.bool)

        # 4. Weight mask (Exact BIO vs Hierarchical Matching)
        labels_col = labels.contiguous().view(-1, 1)
        same_bio = torch.eq(labels_col, labels_col.T)

        if self.mode == 'exact' or self.alpha == 0.0:
            weight_mask = (same_bio & (~self_mask)).float()
        else:
            # Map BIO label ID sang Entity Type ID:
            # ID 1,2 -> 1 (CQ); 3,4 -> 2 (ĐT); 5,6 -> 3 (VBPL); 7,8 -> 4 (NG); 9,10 -> 5 (SL)
            entity_types = (labels_col + 1) // 2
            same_type = torch.eq(entity_types, entity_types.T)
            strong_pos = same_bio.float()
            weak_pos = (same_type & (~same_bio)).float() * self.alpha
            weight_mask = (strong_pos + weak_pos) * (~self_mask).float()

        # 5. Denominator mask (all candidates except self)
        # Sử dụng -1e9 thay cho -inf để tránh hoàn toàn phép toán 0.0 * (-inf) = NaN
        similarity = similarity.masked_fill(self_mask, -1e9)

        # 6. Numerical stability (subtract max)
        logits_max, _ = torch.max(similarity, dim=1, keepdim=True)
        logits = similarity - logits_max.detach()

        # 7. Log probability over denominator
        exp_logits = torch.exp(logits) * (~self_mask).float()
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12))

        # 8. Gán đường chéo bằng 0.0 để triệt tiêu hoàn toàn nguy cơ NaN khi nhân ma trận
        log_prob = log_prob.masked_fill(self_mask, 0.0)

        # 9. Filter anchors that have positive weight > 0
        weight_count = weight_mask.sum(dim=1)
        valid_anchor = weight_count > 0

        if not valid_anchor.any():
            return features.new_tensor(0.0)

        # 10. Mean positive log-likelihood per valid anchor (Weighted Expectation)
        mean_log_prob_pos = (weight_mask * log_prob).sum(dim=1) / weight_count.clamp_min(1e-12)

        # 11. Scalar SupCon loss
        loss = -mean_log_prob_pos[valid_anchor].mean()
        if torch.isnan(loss) or torch.isinf(loss):
            return features.new_tensor(0.0)
        return loss


class PhoBertCrf(RobertaForTokenClassification):
    """
    PhoBERT + CRF (+ optional SupCon) Architecture for Vietnamese Named Entity Recognition (NER)
    
    Pipeline:
      1. Input tokens (word-segmented) -> PhoBERT Tokenizer (BPE subwords)
      2. PhoBERT Encoder -> Contextual representations H in R^(T x 768) (aligned via valid_ids)
      3. Dropout + Linear Classifier -> Emission scores (768 -> N_tags)
      4. CRF Layer -> Transition scores & Negative Log-Likelihood Loss
      5. (Optional) Projection Head + SupCon Loss -> Entity representation learning
      6. Viterbi Decoding -> Optimal BIO entity tag sequence
    """
    def __init__(self, config, use_supcon=None, lambda_supcon=None, supcon_temp=None, supcon_mode=None, supcon_alpha=None, o_label_id=None):
        super(PhoBertCrf, self).__init__(config=config)
        self.num_labels = config.num_labels
        self.crf = CRF(config.num_labels, batch_first=True)

        self.use_supcon = use_supcon if use_supcon is not None else getattr(config, 'use_supcon', False)
        self.lambda_supcon = lambda_supcon if lambda_supcon is not None else getattr(config, 'lambda_supcon', 0.3)
        self.supcon_temp = supcon_temp if supcon_temp is not None else getattr(config, 'supcon_temp', 0.07)
        self.supcon_mode = supcon_mode if supcon_mode is not None else getattr(config, 'supcon_mode', 'exact')
        self.supcon_alpha = supcon_alpha if supcon_alpha is not None else getattr(config, 'supcon_alpha', 0.3)
        self.o_label_id = o_label_id if o_label_id is not None else getattr(config, 'o_label_id', 0)

        if self.use_supcon:
            self.projection_head = ProjectionHead(input_dim=config.hidden_size, hidden_dim=256, output_dim=128)
            self.supcon_loss = TokenSupConLoss(temperature=self.supcon_temp, mode=self.supcon_mode, alpha=self.supcon_alpha)
        else:
            self.projection_head = None
            self.supcon_loss = None

        self.init_weights()
        # Đảm bảo CRF transition parameters được khởi tạo chuẩn xác trong [-0.1, 0.1]
        self.crf.reset_parameters()

    def _init_weights(self, module):
        super(PhoBertCrf, self)._init_weights(module)
        if isinstance(module, CRF):
            module.reset_parameters()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, valid_ids=None,
                label_masks=None):
        # 1. PhoBERT contextual representations
        seq_outputs = self.roberta(input_ids=input_ids,
                                   token_type_ids=token_type_ids,
                                   attention_mask=attention_mask,
                                   head_mask=None)[0]

        # 2. Subword alignment to word-level representations H in R^(T x 768)
        batch_size, max_len, feat_dim = seq_outputs.shape
        range_vector = torch.arange(0, batch_size, dtype=torch.long, device=seq_outputs.device).unsqueeze(1)
        word_representations = seq_outputs[range_vector, valid_ids]
        
        # 3. Dropout + Linear emission layer: 768 -> N_tags
        seq_outputs_dropped = self.dropout(word_representations)
        logits = self.classifier(seq_outputs_dropped)  # Emission scores

        # 4. Viterbi decoding via CRF (chỉ giải mã khi eval/predict, bỏ qua lúc train để tăng tốc tối đa)
        # Ép kiểu logits sang FP32 và tắt autocast để chống tràn số cực đại FP16 trong CRF
        logits_fp32 = logits.float()
        dev_type = 'cuda' if logits.is_cuda else 'cpu'
        if labels is None or not self.training:
            with _disable_amp(dev_type):
                seq_tags = self.crf.decode(logits_fp32, mask=label_masks != 0)
        else:
            seq_tags = []

        # 5. Training loss
        if labels is not None:
            with _disable_amp(dev_type):
                log_likelihood = self.crf(logits_fp32, labels, mask=label_masks.bool(), reduction='mean')
                loss_crf = -1.0 * log_likelihood

            if self.use_supcon and self.projection_head is not None:
                # Representation before dropout passed into projection head
                projected = self.projection_head(word_representations)
                valid_entity_mask = (label_masks != 0) & (labels != self.o_label_id)
                entity_feats = projected[valid_entity_mask]
                entity_tags = labels[valid_entity_mask]
                with _disable_amp(dev_type):
                    loss_supcon = self.supcon_loss(entity_feats.float(), entity_tags)
                    if torch.isnan(loss_supcon) or torch.isinf(loss_supcon):
                        loss_supcon = loss_crf.new_tensor(0.0)
                loss_total = loss_crf + self.lambda_supcon * loss_supcon
                return NerOutput(loss=loss_total, tags=seq_tags, loss_crf=loss_crf, loss_supcon=loss_supcon)
            else:
                return NerOutput(loss=loss_crf, tags=seq_tags)
        else:
            return NerOutput(tags=seq_tags)
