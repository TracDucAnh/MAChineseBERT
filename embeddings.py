import torch
import torch.nn as nn
from transformers.models.bert.modeling_bert import BertEmbeddings  # ← Đổi từ Roberta sang Bert

class BoundaryAwareEmbeddings(BertEmbeddings):
    def __init__(self, config, adaptive=True, boundary_ratio=0.2):
        """
        Embedding layer tương thích với chinese-bert-wwm (BERT architecture)
        
        Args:
            config: BertConfig từ HuggingFace
            adaptive: True = dùng gating động, False = weight cố định
            boundary_ratio: tỉ lệ weight cho boundary embedding (khi adaptive=False)
        """
        super().__init__(config)  # Load pretrained weights từ BERT
        
        self.adaptive = adaptive
        self.boundary_ratio = boundary_ratio
        
        # Custom layers cho BMES
        self.bmes_embeddings = nn.Embedding(4, config.hidden_size)
        
        if self.adaptive:
            self.gate = nn.Linear(config.hidden_size * 2, config.hidden_size)
            self.sigmoid = nn.Sigmoid()
        
        # LayerNorm riêng cho fusion (tránh conflict với LayerNorm gốc)
        self.fusion_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # Khởi tạo weights cho các layer mới
        self._init_custom_weights()
    
    def _init_custom_weights(self):
        """Khởi tạo weights cho các layer custom"""
        nn.init.normal_(self.bmes_embeddings.weight, mean=0.0, std=0.02)
        if self.adaptive:
            nn.init.xavier_uniform_(self.gate.weight)
            nn.init.zeros_(self.gate.bias)
    
    def forward(
        self,
        input_ids=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        bmes_ids=None,  # ← BMES labels: 0=B, 1=M, 2=E, 3=S
        past_key_values_length=0
    ):
        """
        Forward pass với boundary-aware embeddings
        
        Args:
            bmes_ids: Tensor shape [batch_size, seq_len] với giá trị 0-3
                     Ví dụ: "我爱中国" → [3, 3, 0, 2] (S, S, B, E)
        """
        # 1. Lấy embedding chuẩn từ BERT
        E_token = super().forward(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length
        )
        
        # 2. Nếu không có BMES, trả về embedding gốc
        if bmes_ids is None:
            return E_token
        
        # 3. Lấy boundary embeddings
        E_boundary = self.bmes_embeddings(bmes_ids)
        
        # 4. Fusion
        if self.adaptive:
            # Gating mechanism động
            concat = torch.cat([E_boundary, E_token], dim=-1)
            gate_weights = self.sigmoid(self.gate(concat))
            E_fused = gate_weights * E_boundary + (1 - gate_weights) * E_token
        else:
            # Weight cố định
            alpha = self.boundary_ratio
            E_fused = alpha * E_boundary + (1 - alpha) * E_token
        
        # 5. LayerNorm cuối
        embeddings = self.fusion_layernorm(E_fused)
        
        return embeddings