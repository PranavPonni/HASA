import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------
# modReLU
# ---------------------------
def modrelu(inputs, bias):
    """
    modReLU activation function.
    inputs: Tensor of arbitrary shape.
    bias: 1D Tensor（チャネル数と同じ長さ）またはブロードキャスト可能な形.
    """
    norm = torch.abs(inputs) + 0.001
    biased_norm = norm + bias
    magnitude = F.relu(biased_norm)
    phase = torch.sign(inputs)
    return phase * magnitude

# ---------------------------
# インデックス生成 (tunable 用)
# ---------------------------
def generate_index_tunable(s, L):
    """
    tunable 用のインデックスリストを生成する．
    s: 隠れ層のサイズ (偶数でなければならない)
    L: capacity (偶数でなければならない)
    戻り値:
      ind_exe: 長さ L のリスト，各要素は 0～s-1 の順序のリスト（回転の実行順序用）
      ind_param: [ind3, ind4] の2要素リスト．各々長さ s のインデックスリスト（パラメータのシェア用）
    """
    ind1 = list(range(s))
    ind2 = list(range(s))
    for i in range(s):
        if i % 2 == 1:
            ind1[i] = ind1[i] - 1
            if i == s - 1:
                continue
            else:
                ind2[i] = ind2[i] + 1
        else:
            ind1[i] = ind1[i] + 1
            if i == 0:
                continue
            else:
                ind2[i] = ind2[i] - 1

    # L は偶数であることを前提とする．
    ind_exe = [ind1, ind2] * (L // 2)

    ind3 = []
    ind4 = []
    for i in range(s // 2):
        ind3.append(i)
        ind3.append(i + s // 2)
    ind4.append(0)
    for i in range(s // 2 - 1):
        ind4.append(i + 1)
        ind4.append(i + s // 2)
    ind4.append(s - 1)
    ind_param = [ind3, ind4]
    return ind_exe, ind_param

# ---------------------------
# インデックス生成 (FFT 用)
# ---------------------------
def generate_index_fft(s):
    """
    fft 用のインデックスリストを生成する．
    s: 隠れ層のサイズ（2の冪でなければならない）
    戻り値:
      ind_exe: 長さ log2(s) のリスト，各要素は回転順序用のインデックス (torch.LongTensor)
      ind_param: log2(s) 個の torch.LongTensor のリスト（パラメータシェア用）
    """
    def ind_s(k):
        if k == 0:
            return [np.array([1, 0], dtype=np.int32)]
        else:
            temp = np.arange(2**k, dtype=np.int32)
            list0 = [np.concatenate([temp + 2**k, temp])]
            list1 = ind_s(k - 1)
            for i in range(k):
                list0.append(np.concatenate([list1[i], list1[i] + 2**k]))
            return list0

    k_val = int(math.log(s / 2, 2))
    t = ind_s(k_val)

    capacity = int(math.log(s, 2))
    ind_exe = [torch.tensor(t[i], dtype=torch.long) for i in range(capacity)]

    ind_param = []
    for i in range(capacity):
        # 各ブロック毎にインデックスを作成
        arr_list = []
        for j in range(2**i):
            arr = np.arange(0, s, 2**i, dtype=np.int32) + j
            arr_list.append(arr)
        ind_param.append(torch.tensor(np.concatenate(arr_list), dtype=torch.long))
    return ind_exe, ind_param

# ---------------------------
# GORUCell
# ---------------------------
class GORUCell(nn.Module):
    """
    Gated Orthogonal Recurrent Unit (GORU) のセル．
    
    fft=True の場合は FFT スタイルの直交変換を，
    fft=False の場合は tunable スタイルの直交変換を用いる．
    """
    def __init__(self, input_size, num_units, capacity=2, fft=True, activation=modrelu):
        """
        Args:
            input_size: 入力の次元数
            num_units: 隠れ層の次元数
            capacity: 直交変換の capacity．fft=False の場合は偶数でなければならない．
            fft: bool，True の場合は FFT スタイル，False の場合は tunable スタイルを使用する．
            activation: 非線形変換（modReLU）．
        """
        super(GORUCell, self).__init__()
        self.input_size = input_size
        self.num_units = num_units
        self.fft = fft
        self.activation = activation

        if capacity > num_units:
            raise ValueError("capacity must not exceed num_units")
        
        if self.fft:
            # FFT style: num_units は2の冪でなければならない．
            if (num_units & (num_units - 1)) != 0:
                raise ValueError("FFT style only supports num_units that is a power of 2")
            self.capacity = int(math.log(num_units, 2))
            # fft のための学習パラメータ theta (shape: [capacity, num_units//2])
            self.theta = nn.Parameter(torch.empty(self.capacity, num_units // 2))
            nn.init.uniform_(self.theta, -3.14, 3.14)
            # fft 用インデックス
            self.ind_exe, self.index_fft = generate_index_fft(num_units)
        else:
            # tunable style: num_units, capacity は偶数
            if num_units % 2 != 0:
                raise ValueError("Tunable style only supports even num_units")
            if capacity % 2 != 0:
                raise ValueError("Tunable style only supports even capacity")
            self.capacity = capacity
            self.capacity_A = capacity // 2
            self.capacity_B = capacity - self.capacity_A  # (通常 capacity_A == capacity_B)
            # tunable 用のパラメータ
            self.theta_A = nn.Parameter(torch.empty(self.capacity_A, num_units // 2))
            nn.init.uniform_(self.theta_A, -3.14, 3.14)
            self.theta_B = nn.Parameter(torch.empty(self.capacity_B, num_units // 2 - 1))
            nn.init.uniform_(self.theta_B, -3.14, 3.14)
            # tunable 用インデックス
            tunable_ind_exe, tunable_ind_param = generate_index_tunable(num_units, capacity)
            # tunable_ind_exe はリスト（長さ capacity）のリストで，各要素が長さ num_units のインデックス列
            self.ind_exe = [torch.tensor(idx, dtype=torch.long) for idx in tunable_ind_exe]
            # tunable_ind_param = [index_A, index_B]，各長さは num_units
            self.tunable_index_A = torch.tensor(tunable_ind_param[0], dtype=torch.long)
            self.tunable_index_B = torch.tensor(tunable_ind_param[1], dtype=torch.long)
        
        # 入力から各ゲートへの変換（重み U: shape [input_size, 3*num_units]）
        self.U = nn.Linear(input_size, num_units * 3, bias=False)
        nn.init.uniform_(self.U.weight, -0.01, 0.01)
        
        # 隠れ状態から各ゲートへの変換用重み（W_r, W_g: shape [num_units, num_units]）
        self.W_r = nn.Parameter(torch.empty(num_units, num_units))
        self.W_g = nn.Parameter(torch.empty(num_units, num_units))
        nn.init.uniform_(self.W_r, -0.01, 0.01)
        nn.init.uniform_(self.W_g, -0.01, 0.01)
        
        # バイアス項
        self.bias_r = nn.Parameter(torch.full((num_units,), 2.0))
        self.bias_g = nn.Parameter(torch.zeros(num_units))
        self.bias_c = nn.Parameter(torch.full((num_units,), 0.01))
        
    def compute_unitary(self):
        """
        直交変換（ユニタリ変換）の重み v1, v2 を計算する．
        戻り値:
          v1, v2: 各 shape は [capacity, num_units]．
          ind_list: capacity 個の 1D インデックステンソルのリスト
        """
        if self.fft:
            # FFT style の場合
            cos_theta = torch.cos(self.theta)   # [capacity, num_units//2]
            sin_theta = torch.sin(self.theta)
            # 同じものを2回連結して [capacity, num_units] とする
            cos_list = torch.cat([cos_theta, cos_theta], dim=1)
            sin_list = torch.cat([sin_theta, -sin_theta], dim=1)
            v1_list = []
            v2_list = []
            for i in range(self.capacity):
                # self.index_fft[i] は 1D tensor
                idx = self.index_fft[i].to(self.theta.device)
                v1_list.append(cos_list[i, idx])
                v2_list.append(sin_list[i, idx])
            v1 = torch.stack(v1_list, dim=0)  # [capacity, num_units]
            v2 = torch.stack(v2_list, dim=0)
            return v1, v2, self.ind_exe
        else:
            # Tunable style の場合
            # theta_A: [capacity_A, num_units//2]
            cos_theta_A = torch.cos(self.theta_A)
            sin_theta_A = torch.sin(self.theta_A)
            cos_list_A = torch.cat([cos_theta_A, cos_theta_A], dim=1)  # [capacity_A, num_units]
            sin_list_A = torch.cat([sin_theta_A, -sin_theta_A], dim=1)
            
            # theta_B: [capacity_B, num_units//2 - 1]
            cos_theta_B = torch.cos(self.theta_B)
            sin_theta_B = torch.sin(self.theta_B)
            ones = torch.ones(self.capacity_B, 1, device=self.theta_B.device)
            zeros = torch.zeros(self.capacity_B, 1, device=self.theta_B.device)
            # 以下により [capacity_B, num_units] となる
            cos_list_B = torch.cat([ones, cos_theta_B, cos_theta_B, ones], dim=1)
            sin_list_B = torch.cat([zeros, sin_theta_B, -sin_theta_B, zeros], dim=1)
            
            # 各リストの列の順序を入れ替える
            index_A = self.tunable_index_A.to(self.theta_A.device)
            index_B = self.tunable_index_B.to(self.theta_B.device)
            # PyTorch の index_select を用いる．
            diag_list_A = cos_list_A.index_select(dim=1, index=index_A)  # [capacity_A, num_units]
            off_list_A  = sin_list_A.index_select(dim=1, index=index_A)
            diag_list_B = cos_list_B.index_select(dim=1, index=index_B)   # [capacity_B, num_units]
            off_list_B  = sin_list_B.index_select(dim=1, index=index_B)
            # 元の TF 実装では，concat 後に reshape している．
            # ここでは，まず横方向に連結し，その後容量方向に reshape する
            temp_v1 = torch.cat([diag_list_A, diag_list_B], dim=1)  # shape: [capacity/2, 2*num_units]
            temp_v2 = torch.cat([off_list_A, off_list_B], dim=1)
            v1 = temp_v1.view(self.capacity, self.num_units)
            v2 = temp_v2.view(self.capacity, self.num_units)
            return v1, v2, self.ind_exe

    def loop(self, h, v1, v2, ind_list):
        """
        入力 h に対して，capacity 回の直交変換（回転）を順次適用する．
        h: [batch_size, num_units]
        v1, v2: [capacity, num_units]
        ind_list: capacity 個の 1D インデックステンソル（各長さ num_units）
        """
        for i in range(self.capacity):
            diag = h * v1[i]  # broadcasting, shape: [batch, num_units]
            off = h * v2[i]
            idx = ind_list[i].to(h.device)
            # off の各バッチについて，idx で並べ替え
            off_perm = off.index_select(dim=1, index=idx)
            h = diag + off_perm
        return h

    def forward(self, inputs, state):
        """
        Args:
            inputs: [batch_size, input_size]
            state: [batch_size, num_units]
        戻り値:
            new_state: [batch_size, num_units]
            output: 同 new_state
        """
        # 入力変換: Ux の shape は [batch, 3*num_units]
        Ux = self.U(inputs)
        # 3 つに分割
        U_cx, U_rx, U_gx = torch.chunk(Ux, chunks=3, dim=1)
        
        # 隠れ状態からの変換
        W_rh = torch.matmul(state, self.W_r)
        W_gh = torch.matmul(state, self.W_g)
        
        r_tmp = U_rx + W_rh + self.bias_r
        g_tmp = U_gx + W_gh + self.bias_g
        r = torch.sigmoid(r_tmp)
        g = torch.sigmoid(g_tmp)
        
        # 直交変換部分 (unitary evolution)
        v1, v2, ind_list = self.compute_unitary()
        Unitaryh = self.loop(state, v1, v2, ind_list)
        
        c = self.activation(r * Unitaryh + U_cx, self.bias_c)
        new_state = g * state + (1 - g) * c
        
        # 出力と状態は同じ
        return new_state, new_state

# ---------------------------
# 使用例
# ---------------------------
if __name__ == "__main__":
    # サンプル入力 (バッチサイズ 4, 入力次元 10)
    batch_size = 4
    input_size = 10
    num_units = 8  # FFT style の場合は 2 の冪 (例: 8, 16, …)
    
    # fft=True で GORUCell を生成
    cell = GORUCell(input_size=input_size, num_units=num_units, fft=True)
    
    # 初期状態 (ゼロ初期化)
    state = torch.zeros(batch_size, num_units)
    # ダミー入力
    inputs = torch.randn(batch_size, input_size)
    
    # 順伝播
    output, new_state = cell(inputs, state)
    print("Output shape:", output.shape)
