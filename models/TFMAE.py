import math
import torch
import torch.nn as nn
import torch.fft

# -----------------------------------------------------------------------------
# 1. 辅助模块：PositionalEmbedding 和 PatchEmbedding
# -----------------------------------------------------------------------------
class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe) # pe shape: [1, max_len, d_model]

    def forward(self, x):  # x: [bs, seq_len, d_model]
        return self.pe[:, :x.size(1)]  # [1, seq_len, d_model]

class PatchEmbedding(nn.Module):
    def __init__(self, d_model, patch_len, stride, padding, dropout):
        super(PatchEmbedding, self).__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch_layer = nn.ReplicationPad1d((0, padding))
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):  # x: [bs, n_vars, seq_len]
        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride) # [bs, n_vars, patch_num, patch_len]
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        x = self.value_embedding(x)  # [bs * n_vars, patch_num, d_model]
        return self.dropout(x)

# -----------------------------------------------------------------------------
# 2. 核心模型：多任务 TF-MAE
# -----------------------------------------------------------------------------
class Model(nn.Module):
    """
    多任务、通道独立的 TF-MAE
    兼容 time-series-library 的 configs 对象。
    """
    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.num_class = configs.num_class
        self.patch_len = configs.patch_len
        self.stride = configs.stride
        self.padding = configs.padding

        # --- 从 configs 中提取参数 ---
        self.seq_len = configs.seq_len
        self.d_model = configs.d_model
        self.mask_ratio = configs.mask_ratio
        
        # --- 计算 Patch 数量 ---
        L_padded = self.seq_len + self.padding
        self.num_patches = (L_padded - self.patch_len) // self.stride + 1
        self.num_keep = int(self.num_patches * (1 - self.mask_ratio))
        
        # --- 频域目标维度 ---
        self.freq_len = self.seq_len // 2 + 1
        
        # === 1. E_time (编码器骨干) ===
        self.patch_embed = PatchEmbedding(configs.d_model, self.patch_len, self.stride, self.padding, configs.dropout)
        
        # mask_token 直接参与编码阶段
        self.mask_token = nn.Parameter(torch.zeros(1, 1, configs.d_model))  # [1, 1, D] for broadcast
        self.pos_embed = PositionalEmbedding(configs.d_model, max_len=self.num_patches)  # 只需要 patch 数量
        
        encoder_layer = nn.TransformerEncoderLayer(configs.d_model, configs.n_heads, configs.d_ff, 
                                                   configs.dropout, batch_first=True, activation='gelu')
        self.encoder = nn.TransformerEncoder(encoder_layer, configs.e_layers)
        
        # === 2. 预训练任务头 (Pretraining Heads) ===
        # 使用与下游任务相似的 MLP 结构（Linear -> GELU -> Dropout -> Linear）
        # 先定义 input_dim 供各任务头使用（避免后面重复定义）
        input_dim = self.num_patches * configs.d_model

        # 2a. 时域重建头 (flatten all tokens -> MLP -> seq_len)
        hidden_dim_time = max(input_dim, self.seq_len * 2)
        self.head_time = nn.Sequential(
            nn.Linear(input_dim, hidden_dim_time),
            nn.GELU(),
            nn.Dropout(configs.dropout),
            nn.Linear(hidden_dim_time, self.seq_len)
        )

        # 2b. 频域重建头 (flatten all tokens -> MLP -> freq_len * 2)  (*2 for real + imag)
        # 输出维度为 self.freq_len * 2，因此 hidden_dim 至少为输出*2
        hidden_dim_freq = max(input_dim, (self.freq_len * 2) * 2)
        self.head_freq = nn.Sequential(
            nn.Linear(input_dim, hidden_dim_freq),
            nn.GELU(),
            nn.Dropout(configs.dropout),
            nn.Linear(hidden_dim_freq, self.freq_len * 2)
        )

        # === 3. 下游任务头 (Downstream Heads) ===
        
        # 3a. 预测头 - 中间层应该更大
        # input_dim 已在上面定义（self.num_patches * configs.d_model）
        hidden_dim = max(input_dim, configs.pred_len * 2)  # 确保中间层 >= max(输入, 输出*2)

        self.head_forecasting = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(configs.dropout),
            nn.Linear(hidden_dim, configs.pred_len)
        )

        # 3b. 分类头 - 同样的原则
        hidden_dim_cls = max(input_dim, configs.d_model * 2)
        self.head_classification = nn.Sequential(
            nn.Linear(input_dim, hidden_dim_cls),
            nn.GELU(),
            nn.Dropout(configs.dropout),
            nn.Linear(hidden_dim_cls, configs.num_class)
        )

        # 3c. 插补头 - 同样的原则
        input_dim_imp = configs.enc_in * self.num_patches * configs.d_model
        output_dim_imp = self.seq_len * configs.enc_in
        hidden_dim_imp = max(input_dim_imp // 2, output_dim_imp * 2)

        self.head_imputation = nn.Sequential(
            nn.Linear(input_dim_imp, hidden_dim_imp),
            nn.GELU(),
            nn.Dropout(configs.dropout),
            nn.Linear(hidden_dim_imp, output_dim_imp)
        )

        self.initialize_weights()

    def initialize_weights(self):
        nn.init.normal_(self.mask_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    # --- 预训练辅助函数 ---
    
    def _random_masking_and_fill(self, x_patches):
        """随机掩码并用 mask_token 填充被掩码的位置"""
        B_total, N, D = x_patches.shape
        noise = torch.rand(B_total, N, device=x_patches.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        
        # 计算保留和掩码的索引
        ids_keep = ids_shuffle[:, :self.num_keep]
        ids_mask = ids_shuffle[:, self.num_keep:]
        
        # 用 mask_token 替换被掩码的 patches
        mask_tokens = self.mask_token.expand(B_total, N - self.num_keep, D)
        x_masked = x_patches.clone()
        ids_mask_expanded = ids_mask.unsqueeze(-1).expand(-1, -1, D)
        x_masked.scatter_(dim=1, index=ids_mask_expanded, src=mask_tokens)
        
        return x_masked, ids_mask

    def _get_freq_target(self, x_t_full):
        B_orig, N_vars, L = x_t_full.shape
        x_t_flat = x_t_full.reshape(B_orig * N_vars, L)
        x_f = torch.fft.rfft(x_t_flat, n=self.seq_len, norm='ortho')
        x_f_real = x_f.real
        x_f_imag = x_f.imag
        x_f_target = torch.stack((x_f_real, x_f_imag), dim=-1)
        return x_f_target

    # --- 核心编码函数 ---
    
    def pretrain_encode(self, x_t):
        """为预训练任务进行编码 (带掩码填充)"""
        B_orig, N_vars, L_orig = x_t.shape
        B_total = B_orig * N_vars
        
        # 1a. Patch 化
        x_patches = self.patch_embed(x_t) # (B_total, N, D)
        
        # 1b. 随机掩码并填充 mask_token
        x_masked, ids_mask = self._random_masking_and_fill(x_patches) # (B_total, N, D)
        
        # 1c. 添加位置嵌入
        x_masked = x_masked + self.pos_embed(x_masked)  # (B_total, N, D)
        
        # 1d. 通过编码器
        z = self.encoder(x_masked) # (B_total, N, D)
        
        return z, ids_mask, B_orig, N_vars

    def downstream_encode(self, x_t):
        """为下游任务进行编码 (不掩码)"""
        B_orig, N_vars, L_orig = x_t.shape

        # 1. Patch 化
        x_patches = self.patch_embed(x_t) # (B_orig* N_vars, N, D)
        
        # 2. 添加位置嵌入
        x_patches = x_patches + self.pos_embed(x_patches)  # (B_orig* N_vars, N, D)
        
        # 3. 通过编码器
        z = self.encoder(x_patches) # (B_orig* N_vars, N, D)
        
        return z, B_orig, N_vars

    # --- 任务方法 ---

    def pretrain(self, x_t):
        """
        任务1: 预训练
        x_t: [bs, n_vars, seq_len]
        """
        B_orig, N_vars, L_orig = x_t.shape
        B_total = B_orig * N_vars
        
        # --- 准备目标 ---
        x_t_target = x_t.reshape(B_total, L_orig)
        x_f_target = self._get_freq_target(x_t) # (B_total, freq_len, 2)
        
        # --- 1. 编码 (带掩码填充) ---
        z, ids_mask, _, _ = self.pretrain_encode(x_t)  # (B_total, N, D)
        
        # --- 2. 展平所有 tokens ---
        z_flat = z.reshape(B_total, -1)  # (B_total, N * D)
        
        # --- 3. 时域重建 (T->T) ---
        x_t_pred = self.head_time(z_flat)  # (B_total, seq_len)
        
        # --- 4. 频域重建 (T->F) ---
        x_f_pred_flat = self.head_freq(z_flat)  # (B_total, freq_len * 2)
        x_f_pred = x_f_pred_flat.reshape(B_total, self.freq_len, 2)
        
        # --- 5. 计算损失 ---
        # 5a. 时域损失 L(T->T) - 只在被掩码的位置计算损失
        # 创建掩码：被掩码的patch对应的时间步应该被考虑
        # ids_mask: (B_total, num_masked_patches) 包含被掩码的patch索引
        
        # 构建patch级别的mask (1表示被掩码, 0表示未被掩码)
        mask_patches = torch.zeros(B_total, self.num_patches, device=x_t.device)
        mask_patches.scatter_(1, ids_mask, 1.0)  # 被掩码的patch标记为1
        
        # 将patch级别的mask扩展到时间步级别
        # 每个patch对应 stride 个时间步（考虑重叠）
        mask_time = torch.zeros(B_total, self.seq_len, device=x_t.device)
        for i in range(self.num_patches):
            start_idx = i * self.stride
            end_idx = min(start_idx + self.patch_len, self.seq_len)
            mask_time[:, start_idx:end_idx] += mask_patches[:, i:i+1].expand(-1, end_idx - start_idx)
        
        # 归一化mask (处理重叠区域)
        mask_time = torch.clamp(mask_time, 0, 1)
        
        loss_t = torch.sum(((x_t_pred - x_t_target) ** 2) * mask_time) / (mask_time.sum() + 1e-8)
        
        # 5b. 频域损失 L(T->F)
        loss_f = nn.MSELoss()(x_f_pred, x_f_target)
        
        # 5c. 总损失
        loss = (self.configs.lambda_t * loss_t) + (self.configs.lambda_f * loss_f)
        
        return loss, x_t_pred.reshape(B_orig, N_vars, -1), x_f_pred.reshape(B_orig, N_vars, self.freq_len, 2), loss_t, loss_f

    def freeze_encoder(self):
        """冻结编码器参数,只训练任务头"""
        # 冻结 patch embedding
        for param in self.patch_embed.parameters():
            param.requires_grad = False
        
        # 冻结位置嵌入
        for param in self.pos_embed.parameters():
            param.requires_grad = False
        
        # 冻结 transformer encoder
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        # 冻结 mask_token
        self.mask_token.requires_grad = False
        
        print("Encoder frozen! Only task-specific heads will be trained.")
    
    def unfreeze_encoder(self):
        """解冻编码器参数"""
        for param in self.patch_embed.parameters():
            param.requires_grad = True
        for param in self.pos_embed.parameters():
            param.requires_grad = True
        for param in self.encoder.parameters():
            param.requires_grad = True
        self.mask_token.requires_grad = True
        print("Encoder unfrozen! All parameters will be trained.")

    def forecast(self, x_t, x_mark_enc=None):
        """
        任务2: 预测
        x_t: [bs, n_vars, seq_len] (历史数据)
        返回: [bs, pred_len, n_vars] (未来预测)
        """
        # Normalization
        means = x_t.mean(2, keepdim=True).detach()
        x_t = x_t - means
        stdev = torch.sqrt(torch.var(x_t, dim=2, keepdim=True, unbiased=False) + 1e-5)
        x_t = x_t / stdev
        
        # 1. 编码 (不掩码)
        z, B_orig, N_vars = self.downstream_encode(x_t) # (B_orig*N_vars, num_patches, d_model)
        
        # 2. 展平所有 tokens (每个变量独立)
        z_flat = z.reshape(B_orig * N_vars, self.num_patches * self.d_model) # (B_orig*N_vars, num_patches*d_model)
        
        # 3. 通过预测头 (每个变量独立预测)
        y_pred_flat = self.head_forecasting(z_flat) # (B_orig*N_vars, pred_len)
        
        # 4. 恢复形状
        y_pred = y_pred_flat.reshape(B_orig, N_vars, self.configs.pred_len) # (B_orig, N_vars, pred_len)
        
        # De-Normalization
        y_pred = y_pred * stdev
        y_pred = y_pred + means
        
        # 转换为 [bs, pred_len, n_vars]
        y_pred = y_pred.permute(0, 2, 1)
        
        return y_pred

    def impute(self, x_t, mask):
        """
        任务3: 插补 (重建)
        x_t: [bs, n_vars, seq_len] (包含缺失值, 通常用 0 填充)
        返回: [bs, seq_len, n_vars] (重建的完整时序)
        """
        # Normalization
        means = torch.sum(x_t, dim=2) / (torch.sum(mask == 1, dim=2) + 1e-8)
        means = means.unsqueeze(2).detach()
        x_t = x_t - means
        x_t = x_t.masked_fill(mask == 0, 0)
        stdev = torch.sqrt(torch.sum(x_t * x_t, dim=2) / (torch.sum(mask == 1, dim=2) + 1e-8) + 1e-5)
        stdev = stdev.unsqueeze(2).detach()
        x_t = x_t / stdev
        
        B_orig, N_vars, _ = x_t.shape
        
        # 1. 编码 (不掩码)
        z, _, _ = self.downstream_encode(x_t) # (B_total, N, D)
        
        # 2. 展平所有 tokens
        z_flat = z.reshape(B_orig, N_vars * self.num_patches * self.d_model)  # (B, N_vars * N * D)
        
        # 3. 通过插补头
        x_t_pred_flat = self.head_imputation(z_flat)  # (B, seq_len * N_vars)
        
        # 4. 恢复形状
        x_t_pred = x_t_pred_flat.reshape(B_orig, N_vars, self.seq_len)
        
        # De-Normalization
        x_t_pred = x_t_pred * stdev
        x_t_pred = x_t_pred + means
        
        # 转换为 [bs, seq_len, n_vars]
        x_t_pred = x_t_pred.permute(0, 2, 1)
        
        return x_t_pred

    def classify(self, x_t, x_mark_enc=None):
        """
        任务4: 分类
        x_t: [bs, n_vars, seq_len]
        返回: [bs, num_class]
        """
        # Normalization
        means = x_t.mean(2, keepdim=True).detach()
        x_t = x_t - means
        stdev = torch.sqrt(torch.var(x_t, dim=2, keepdim=True, unbiased=False) + 1e-5)
        x_t = x_t / stdev
        
        # 1. 编码 (不掩码)
        z, B_orig, N_vars = self.downstream_encode(x_t) # (B_total, N, D)
        
        # 2. 恢复 n_vars 维度并展平
        z_reshaped = z.reshape(B_orig, N_vars, self.num_patches, self.d_model) # (B, N_vars, N, D)
        
        # 3. 在 n_vars 维度上进行平均池化
        z_pooled = torch.mean(z_reshaped, dim=1)  # (B, N, D)
        
        # 4. 展平所有 tokens
        z_flat = z_pooled.reshape(B_orig, self.num_patches * self.d_model)  # (B, N * D)
        
        # 5. 通过分类头
        y_pred = self.head_classification(z_flat)
        
        return y_pred

    def detect_anomaly(self, x_t):
        """
        任务5: 异常检测
        x_t: [bs, n_vars, seq_len]
        返回: [bs, seq_len, n_vars] (时域重建)
        """
        # Normalization
        means = x_t.mean(2, keepdim=True).detach()
        x_t = x_t - means
        stdev = torch.sqrt(torch.var(x_t, dim=2, keepdim=True, unbiased=False) + 1e-5)
        x_t = x_t / stdev
        
        B_orig, N_vars, L_orig = x_t.shape
        B_total = B_orig * N_vars
        
        # --- 1. 编码 (不掩码) ---
        z, _, _ = self.downstream_encode(x_t) # (B_total, N, D)
        
        # --- 2. 展平所有 tokens ---
        z_flat = z.reshape(B_total, -1)  # (B_total, N * D)
        
        # --- 3. 重建时域 (T->T) ---
        x_t_pred = self.head_time(z_flat)  # (B_total, seq_len)
        
        # 恢复形状
        x_t_pred = x_t_pred.reshape(B_orig, N_vars, L_orig)
        
        # De-Normalization
        x_t_pred = x_t_pred * stdev
        x_t_pred = x_t_pred + means
        
        # 转换为 [bs, seq_len, n_vars]
        x_t_pred = x_t_pred.permute(0, 2, 1)
        
        return x_t_pred


    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        """
        主 'forward' 函数 (路由器)
        x_enc: [bs, seq_len, n_vars] (ts-library 标准输入)
        """
        if self.task_name == 'pretrain':
            # [bs, seq_len, n_vars] -> [bs, n_vars, seq_len]
            x_enc = x_enc.permute(0, 2, 1)
            # 返回: loss, x_t_pred, x_f_pred, loss_t, loss_f
            return self.pretrain(x_enc)
        
        elif self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            # [bs, seq_len, n_vars] -> [bs, n_vars, seq_len]
            x_enc = x_enc.permute(0, 2, 1)
            # 返回: [bs, pred_len, n_vars]
            dec_out = self.forecast(x_enc, x_mark_enc)
            return dec_out[:, -self.pred_len:, :]
        
        elif self.task_name == 'imputation':
            # [bs, seq_len, n_vars] -> [bs, n_vars, seq_len]
            x_enc = x_enc.permute(0, 2, 1)
            # 返回: [bs, seq_len, n_vars]
            dec_out = self.impute(x_enc, mask.permute(0, 2, 1) if mask is not None else None)
            return dec_out
        
        elif self.task_name == 'classification':
            # [bs, seq_len, n_vars] -> [bs, n_vars, seq_len]
            x_enc = x_enc.permute(0, 2, 1)
            # 返回: [bs, num_class]
            return self.classify(x_enc, x_mark_enc)
        
        elif self.task_name == 'anomaly_detection':
            # [bs, seq_len, n_vars] -> [bs, n_vars, seq_len]
            x_enc = x_enc.permute(0, 2, 1)
            # 返回: [bs, seq_len, n_vars]
            return self.detect_anomaly(x_enc)
        
        else:
            raise ValueError(f"未知的 task_name: {self.task_name}")