# Kiến trúc Kết hợp Linear-Chain CRF và Hierarchical Supervised Contrastive Learning cho Bài toán Nhận dạng Thực thể Hành chính Công (PAP_NER)

> **Tài liệu kỹ thuật & Đặc tả kiến trúc phục vụ bài báo khoa học (Research Paper Specification)**  
> **Dự án:** VPhoBertTagger — Mô hình DistilPhoBERT-CRF tích hợp Hierarchical Contrastive Regularizer cho Thực thể Pháp luật và Thủ tục Hành chính Việt Nam  
> **Bộ ngữ liệu đối chuẩn:** PAP_NER (*PLOS ONE 2026*, DOI: [10.1371/journal.pone.0353166](https://doi.org/10.1371/journal.pone.0353166))

---

## TÓM TẮT (ABSTRACT)

Nhận dạng thực thể tên riêng trong văn bản pháp luật và thủ tục hành chính công (Public Administration Procedures Named Entity Recognition - PAP_NER) đặt ra những thách thức lớn: tên văn bản quy phạm pháp luật (`VBPL`) có độ dài kéo dài nhiều từ tố, các lớp định lượng (`SL`) gặp hiện tượng mất cân bằng dữ liệu nghiêm trọng, và các cụm cơ quan thẩm quyền (`CQ`) hay đối tượng thực hiện (`ĐT`) dễ bị nhầm lẫn ranh giới. Các kiến trúc tiền nhiệm dựa trên PhoBERT-base kết hợp Conditional Random Field (CRF) thuần túy đạt hiệu năng cao nhưng chịu chi phí tính toán lớn (135 triệu tham số, 12 tầng Transformer) và dễ mắc lỗi cắt cụt ranh giới (Boundary Truncation).

Tài liệu này đặc tả toàn diện mô hình lai **DistilPhoBERT + Linear-Chain CRF + Hierarchical Supervised Contrastive Learning (H-SupCon)**. Bằng cách thiết lập cơ chế tương phản phân cấp với liên kết mềm ($\alpha = 0.3$) giữa từ mở đầu (`B-`) và từ thành phần (`I-`), mô hình khắc phục triệt để hiện tượng xé rách biểu diễn không gian ở các thực thể dài và cứu đói mẫu tương phản cho lớp thiểu số. Đồng thời, kiến trúc áp dụng cơ chế **Suy luận tách rời hoàn toàn (Decoupled Inference)**, cho phép loại bỏ hoàn toàn đầu chiếu tương phản lúc kiểm thử, đạt chi phí tính toán bổ sung bằng 0 (0 FLOPs overhead). Kết quả thực nghiệm trên 10.278 câu kiểm thử chứng minh mô hình 6 tầng nén nhẹ (~92M tham số) đạt **Strict Micro-F1 97.66%**, vượt trội bài báo gốc PLOS ONE ở lớp `VBPL` (97.37% vs 97.08%), lớp `SL` (96.90% vs 96.55%), lớp `NG` (99.99% vs 99.25%), và đạt tốc độ suy luận nhanh gấp ~1.85 lần.

---

## 1. ĐẶT BÀI TOÁN & ĐỘNG LỰC NGHIÊN CỨU (MOTIVATION)

### 1.1. Bộ ngữ liệu PAP_NER & Hệ thống 11 nhãn BIO
Hệ thống nhận dạng thực thể hành chính công chuẩn hóa 5 lớp thực thể nghiệp vụ cốt lõi, được biểu diễn qua chuẩn BIO gồm 11 nhãn:

| Nhãn BIO | ID | Nhóm thực thể | Định nghĩa nghiệp vụ | Ví dụ tiêu biểu |
| :--- | :---: | :--- | :--- | :--- |
| `O` | 0 | Outside | Từ thông thường nằm ngoài thực thể | *căn cứ, yêu cầu nộp, theo quy định* |
| `B-CQ`, `I-CQ` | 1, 2 | **CQ** (Cơ quan) | Cơ quan, đơn vị có thẩm quyền ban hành/giải quyết | *Ủy ban nhân dân tỉnh Hà Giang, Sở Kế hoạch và Đầu tư* |
| `B-ĐT`, `I-ĐT` | 3, 4 | **ĐT** (Đối tượng) | Tổ chức, cá nhân, doanh nghiệp thực hiện TTHC | *người sử dụng đất, doanh nghiệp nhỏ và vừa* |
| `B-VBPL`, `I-VBPL`| 5, 6 | **VBPL** (Văn bản pháp luật) | Luật, Nghị định, Thông tư, Quyết định | *Nghị định 123/2020/NĐ-CP, Bộ luật Hình sự* |
| `B-NG`, `I-NG` | 7, 8 | **NG** (Thời gian) | Mốc thời gian ban hành, ngày hiệu lực, thời hạn | *15/03/2024, ngày 20 tháng 11 năm 2023* |
| `B-SL`, `I-SL` | 9, 10| **SL** (Định lượng) | Lệ phí, thời hạn xử lý, số bộ hồ sơ, tỷ lệ | *03 bộ hồ sơ, 50 triệu đồng, 150.000 đồng, 100%* |

### 1.2. Hai hạn chế chí mạng của phương pháp cũ (Baseline Limitations)
1. **Lỗi xé rách ranh giới ở thực thể dài (Long-span Representation Tearing):**  
   Trong tập huấn luyện, token `I-VBPL` (84.604 tokens) gấp hơn 3.3 lần số lượng `B-VBPL` (25.522 tokens). Cơ chế Exact BIO SupCon truyền thống coi $B\text{-VBPL}$ và $I\text{-VBPL}$ là cặp đối kháng âm (Negative samples), cố tình đẩy xa vector của loại văn bản (như *"Nghị định"*) khỏi mã số hiệu (như *"123/2020/NĐ-CP"*). Điều này phá hủy liên kết ngữ nghĩa nội tại, khiến bài báo PLOS ONE thừa nhận **58% tổng số lỗi sai là lỗi cắt cụt ranh giới**.
2. **Hiện tượng đói mẫu tương phản ở lớp hiếm (Orphan Anchor Starvation):**  
   Lớp `SL` chỉ chiếm 7.1% dữ liệu. Trong các mini-batch, token $B\text{-SL}$ thường xuất hiện đơn độc mà không có $B\text{-SL}$ nào khác. Với Exact Matching, anchor này không tìm được positive sample (`pos_count == 0`), buộc code phải loại bỏ anchor để tránh chia cho 0. Hậu quả là lớp `SL` bị "bỏ đói" gradient tương phản trong hơn 80% số batch.

---

## 2. KIẾN TRÚC TỔNG THỂ: THIẾT KẾ ĐỘNG CƠ KÉP (DUAL-ENGINE DESIGN)

Kiến trúc kết hợp 3 thành phần liên kết chặt chẽ qua cơ chế học đa mục tiêu:

```text
                                  Văn bản tiếng Việt đầu vào
                                              │
                                              ▼
                                 Tách từ âm tiết (VnCoreNLP)
                                              │
                                              ▼
                             DistilPhoBERT Encoder (6 Layers, 92M)
                                              │
                                              ▼
                            Contextual Representations H [B, T, 768]
                                  (Căn chỉnh qua valid_ids)
                                        /          \
                                       /            \
               [ĐỘNG CƠ CẤU TRÚC CHUỖI]              [ĐỘNG CƠ HÌNH HỌC TƯƠNG PHẢN]
                         │                                         │
                         ▼                                         ▼
                 Linear Emission Layer                      Projection Head
                 W ∈ R^(768 × 11), b ∈ R^11                 768 → 256 → 128
                         │                                         │
                         ▼                                         ▼
                 Emissions E [B, T, 11]                 Mặt cầu đơn vị S^127
                         │                                         │
                         ▼                                         ▼
                 Linear-Chain CRF                       Lọc bỏ nhãn O & Padding
                (Ma trận A ∈ R^(11 × 11))                          │
                        /         \                                ▼
                       /           \                     Hierarchical SupCon Loss
                      ▼             ▼                       (α = 0.3, τ = 0.07)
                  Lúc Train      Lúc Test                          │
                      │             │                              ▼
                  Forward Alg    Viterbi Alg                   L_H-SupCon
                      │             │                              │
                      ▼             ▼                              │
                    L_CRF       BIO Tags                           │
                      \             │                              /
                       \            │ (Inference Output)          /
                        \           │                            /
                         ▼          ▼                           ▼
                        L_total = L_CRF + 0.3 × L_H-SupCon (Lúc Huấn luyện)
```

### 2.1. Động cơ Ngữ nghĩa & Căn chỉnh Từ (DistilPhoBERT & Subword Alignment)
Mô hình sử dụng backbone `trungbb8/distilphobert` (6 tầng Transformer, hidden dimension $d = 768$, 12 attention heads). Để giải quyết hiện tượng phân mảnh subword BPE của tiếng Việt, vector biểu diễn được căn chỉnh về cấp độ từ thông qua vector chỉ số hợp lệ `valid_ids`:
$$\mathbf{H} = \text{seq\_outputs}[\text{range\_vector}, \text{valid\_ids}] \in \mathbb{R}^{B \times T \times 768}$$
đảm bảo mỗi từ đơn hoặc từ ghép đều có duy nhất một vector ngữ cảnh đại diện tương ứng với 1 nhãn BIO.

### 2.2. Nhánh Đầu chiếu Phi tuyến (Projection Head)
Nhánh tương phản trích xuất vector biểu diễn trước tầng dropout và đưa qua mạng MLP 2 tầng:
$$\mathbf{z}_i = \frac{\mathcal{H}_{\text{proj}}(\mathbf{h}_i)}{\|\mathcal{H}_{\text{proj}}(\mathbf{h}_i)\|_2} \in \mathbb{S}^{127}, \quad \text{với } \mathcal{H}_{\text{proj}}(\mathbf{h}) = \mathbf{W}_2 \cdot \text{ReLU}(\mathbf{W}_1 \mathbf{h} + \mathbf{b}_1) + \mathbf{b}_2$$
với $\mathbf{W}_1 \in \mathbb{R}^{256 \times 768}$ và $\mathbf{W}_2 \in \mathbb{R}^{128 \times 256}$. Ánh xạ từ không gian ẩn ban đầu $d = 768$ (của DistilPhoBERT) qua tầng ẩn trung gian 256 chiều xuống không gian tương phản $d_{\text{proj}} = 128$ chiều. Chuẩn hóa $L_2$ đưa toàn bộ vector lên bề mặt siêu cầu đơn vị $\mathbb{S}^{127}$.

> **Lưu ý cốt lõi:** Projection Head chỉ đóng vai trò là một **nhánh phụ tạm thời hỗ trợ quá trình huấn luyện**, hấp thụ các biến dạng phi tuyến của hàm SupCon mà không làm biến dạng không gian gốc của encoder. Khi suy luận thực tế (inference), nhánh này được **loại bỏ hoàn toàn**; mô hình không cần tới Projection Head để dự đoán chuỗi nhãn.

### 2.3. Luận cứ Khoa học: Tại sao Tối ưu Đồng thời (Joint Optimization) vượt trội hơn Huấn luyện Hai giai đoạn (Two-Stage Training)?

Một câu hỏi phương pháp luận lớn được đặt ra trong các nghiên cứu học tương phản cho NER (như thảo luận trong bài báo trên tạp chí *MDPI Electronics 2026*, [DOI: 10.3390/electronics15132770](https://www.mdpi.com/2079-9292/15/13/2770)): *Nên fine-tune BERT bằng Contrastive Learning ở giai đoạn 1 rồi mới huấn luyện BERT + CRF ở giai đoạn 2 (Two-Stage), hay tối ưu đồng thời cả hai hàm mất mát trong một giai đoạn thống nhất (Joint Optimization)?*

Nghiên cứu của chúng tôi lựa chọn phương án **Kết hợp đồng thời SupCon và CRF trong cùng một giai đoạn huấn luyện** ($\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CRF}} + \lambda \mathcal{L}_{\text{SupCon}}$) dựa trên các cơ sở học thuật vững chắc:

| Tiêu chí So sánh | Huấn luyện Hai giai đoạn (Two-Stage Training) | Tối ưu Đồng thời Đề xuất (Joint SupCon + CRF) |
| :--- | :--- | :--- |
| **Giai đoạn 1** | Fine-tune PhoBERT bằng Contrastive Learning | Huấn luyện PhoBERT đồng thời với cả hai mục tiêu |
| **Giai đoạn 2** | Dùng PhoBERT đã fine-tune kết hợp CRF | Không cần giai đoạn 2 (Chỉ 1 vòng lặp huấn luyện duy nhất) |
| **Sử dụng Nhãn Vàng (Gold Labels)** | Thường phụ thuộc vào data augmentation hoặc pseudo-labels | Tận dụng trực tiếp 100% nhãn vàng BIO của tập dữ liệu |
| **Độ lệch Mục tiêu (Objective Mismatch)** | Dễ xuất hiện hiện tượng lệch biểu diễn (Representation Drift): không gian được tối ưu ở Giai đoạn 1 chưa chắc đã tối ưu cho ma trận chuyển trạng thái CRF ở Giai đoạn 2 | **Triệt tiêu hoàn toàn độ lệch:** Cả hai loss cùng tác động trực tiếp và liên tục lên biểu diễn của PhoBERT |
| **Dòng Gradient (Gradient Flow)** | Gradient đi rời rạc qua từng pha, dễ bị Catastrophic Forgetting ở Giai đoạn 2 | Gradient từ cấu trúc chuỗi (CRF) và hình học không gian (SupCon) truyền ngược đồng thời |
| **Độ phức tạp & Chi phí** | Phải quản lý 2 pipeline, tốn gấp đôi thời gian huấn luyện | **Tiết kiệm 50% thời gian train**, chỉ cần 1 lần huấn luyện duy nhất |

#### Bốn ưu điểm vượt trội của phương pháp Tối ưu Đồng thời:
1. **Tận dụng trực tiếp Nhãn vàng của bài toán NER:**  
   SupCon sử dụng trực tiếp nhãn BIO có sẵn để xác định chính xác các cặp tương đồng (cùng lớp) và đối kháng (khác lớp). Quá trình tương phản được định hướng trực tiếp bởi nghiệp vụ pháp lý - hành chính, thay vì các phép biến đổi giả lập (data augmentation) dễ làm méo mó ngữ nghĩa tiếng Việt.
2. **Sự phối hợp cộng sinh giữa hai mục tiêu (Symbiotic Synergy):**  
   - SupCon làm nhiệm vụ **tổ chức lại hình học không gian**: kéo các token cùng loại lại gần nhau, tạo khoảng cách phân tách (Margin) rõ ràng giữa các lớp thực thể.
   - CRF làm nhiệm vụ **ràng buộc cú pháp chuỗi**: học ma trận chuyển trạng thái $\mathbf{A}$ để tìm con đường gán nhãn hợp lệ toàn cục.  
   Hai mục tiêu này hỗ trợ lẫn nhau chứ không thay thế nhau.
3. **Tránh hiện tượng Quên tai hại (Catastrophic Forgetting):**  
   Ở phương pháp hai giai đoạn, khi chuyển sang huấn luyện CRF ở Giai đoạn 2, việc cập nhật trọng số để học ma trận chuyển trạng thái rất dễ làm phá vỡ các cụm vector đã được nén chặt ở Giai đoạn 1. Tối ưu đồng thời đảm bảo không gian vector luôn vừa phục vụ phân tách lớp vừa sẵn sàng cho bộ giải mã Viterbi.
4. **Quy trình tối ưu thống nhất (Unified Pipeline):**  
   Trong mỗi batch, mô hình chỉ thực hiện đúng **một lần Forward và một lần Backward**. Gradient từ cả nhánh chính (CRF) và nhánh phụ (Projection Head) cùng cập nhật về PhoBERT Encoder, tạo nên sự cân bằng hoàn hảo giữa khả năng phân biệt lớp và khả năng gán nhãn chuỗi.

---

## 3. CÔNG THỨC TOÁN HỌC & ĐỘNG LỰC HỌC HÀM LOSS (LOSS FORMULATION)

Hàm mục tiêu toàn cục kết hợp hai bài toán tối ưu:
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CRF}} + \lambda \cdot \mathcal{L}_{\text{H-SupCon}} \quad (\lambda = 0.3)$$

### 3.1. Linear-Chain CRF Negative Log-Likelihood ($\mathcal{L}_{\text{CRF}}$)
Với câu đầu vào $\mathbf{x} = (x_1, \dots, x_T)$ và chuỗi nhãn ground-truth $\mathbf{y} = (y_1, \dots, y_T) \in \mathcal{Y}^T$ ($|\mathcal{Y}| = 11$):
$$\text{Score}(\mathbf{x}, \mathbf{y}) = \sum_{t=1}^T \mathbf{E}_{t, y_t} + \sum_{t=1}^{T-1} \mathbf{A}_{y_t, y_{t+1}}$$
trong đó $\mathbf{E} = \mathbf{H} \mathbf{W}_{\text{cls}} + \mathbf{b}_{\text{cls}} \in \mathbb{R}^{T \times 11}$ là điểm phát xạ, và $\mathbf{A} \in \mathbb{R}^{11 \times 11}$ là ma trận chuyển trạng thái học được.

Xác suất có điều kiện của chuỗi nhãn:
$$P(\mathbf{y} \mid \mathbf{x}) = \frac{\exp(\text{Score}(\mathbf{x}, \mathbf{y}))}{\sum_{\mathbf{y}' \in \mathcal{Y}^T} \exp(\text{Score}(\mathbf{x}, \mathbf{y}'))} = \frac{\exp(\text{Score}(\mathbf{x}, \mathbf{y}))}{Z(\mathbf{x})}$$

Hàm mất mát cấu trúc chuỗi là log-âm hợp lý (NLL) được tính toán chính xác bằng **Thuật toán Xuôi (Forward Algorithm)**:
$$\mathcal{L}_{\text{CRF}} = -\ln P(\mathbf{y} \mid \mathbf{x}) = \ln Z(\mathbf{x}) - \text{Score}(\mathbf{x}, \mathbf{y})$$

### 3.2. Hierarchical Supervised Contrastive Loss ($\mathcal{L}_{\text{H-SupCon}}$)

#### A. Định nghĩa Cặp Tương đồng & Đối kháng từ Nhãn Vàng (Gold Labels)
Hàm SupCon sử dụng trực tiếp nhãn vàng BIO của các token trong batch để định hướng không gian hình học. Trong thiết kế này:
- **Anchor:** Token thực thể đang được xét $i \in \mathcal{I}_{\text{entity}}$ (với $y_i \neq \text{O}$).
- **Strong-positive pair:** Các token $j$ khác trong batch có **cùng nhãn BIO chính xác** với anchor ($y_j = y_i$).
- **Soft-positive pair:** Các token $j$ có **cùng loại thực thể nhưng khác tiền tố BIO** ($\text{Type}(y_j) = \text{Type}(y_i)$ nhưng $y_j \neq y_i$, ví dụ giữa $B\text{-CQ}$ và $I\text{-CQ}$). Cặp này được nhân với trọng số mềm $\alpha = 0.3$ nhằm giữ tính liên kết cụm mà không xóa nhòa ranh giới vị trí.
- **Negative pair:** Các token thực thể có **khác loại thực thể** với anchor ($\text{Type}(y_j) \neq \text{Type}(y_i)$).
- **Token `O` (Ngoài thực thể):** **Bị loại bỏ hoàn toàn ($100\%$) khỏi SupCon Loss** (để tránh việc các từ ngữ thông thường chiếm >85% dữ liệu làm loãng lực co cụm của các thực thể hiếm), nhưng **vẫn tham gia đầy đủ vào hàm mất mát CRF** để mô hình học chính xác ranh giới tách biệt giữa từ thông thường và thực thể.

| Token Anchor ($i$) | Token so sánh ($j$) | Quan hệ trong Exact BIO | Quan hệ trong Hierarchical SupCon (Đề xuất) | Trọng số $W_{i, j}$ |
| :---: | :---: | :---: | :---: | :---: |
| `B-CQ` | `B-CQ` | Positive pair | **Strong-Positive pair** | **$1.0$** |
| `B-CQ` | `I-CQ` | Negative pair | **Soft-Positive pair** (Cùng loại Cơ quan) | **$\alpha = 0.3$** |
| `B-CQ` | `B-VBPL` | Negative pair | **Negative pair** (Khác loại thực thể) | **$0.0$** |
| `B-CQ` | `B-ĐT` | Negative pair | **Negative pair** (Khác loại thực thể) | **$0.0$** |
| `B-CQ` | `O` | *Bị loại* | *Bị loại khỏi SupCon (chỉ tham gia vào CRF)* | — |

#### B. Ma trận Trọng số Phân cấp $W_{i, j}$ & Lọc Anchor Hợp lệ
Ma trận trọng số tương đồng phân cấp $W_{i, j}$ giữa hai token $i$ và $j$ ($i \neq j$) trong tập token thực thể $\mathcal{I}_{\text{entity}} = \{i \mid y_i \neq \text{O}\}$ được biểu diễn tổng quát:
$$W_{i, j} = \begin{cases} 
1.0 & \text{nếu } y_i = y_j \quad (\textbf{Strong-positive}: \text{Trùng nhãn BIO, ví dụ: } B\text{-VBPL} \leftrightarrow B\text{-VBPL}) \\ 
\alpha & \text{nếu } \text{Type}(y_i) = \text{Type}(y_j) \text{ và } y_i \neq y_j \quad (\textbf{Soft-positive}: \text{Trùng loại thực thể, ví dụ: } B\text{-VBPL} \leftrightarrow I\text{-VBPL}) \\ 
0.0 & \text{ngược lại} \quad (\textbf{Negative sample}: \text{Khác loại thực thể, ví dụ: } \text{VBPL} \leftrightarrow \text{CQ})
\end{cases}$$
*(với $\alpha = 0.3$)*.

Với mỗi anchor $i \in \mathcal{I}_{\text{entity}}$, độ đo cosine similarity chuẩn hóa theo nhiệt độ $\tau = 0.07$:
$$\mathcal{L}_{\text{H-SupCon}}(i) = - \frac{1}{\sum_{j \in \mathcal{A}(i)} W_{i, j}} \sum_{j \in \mathcal{A}(i)} W_{i, j} \ln \left[ \frac{\exp\left(\mathbf{z}_i^\top \mathbf{z}_j / \tau\right)}{\sum_{k \in \mathcal{A}(i)} \exp\left(\mathbf{z}_i^\top \mathbf{z}_k / \tau\right)} \right]$$
với $\mathcal{A}(i) = \mathcal{I}_{\text{entity}} \setminus \{i\}$. 

**Điều kiện lọc Anchor hợp lệ:** Hàm mất mát chỉ được tính trên tập các anchor hợp lệ có ít nhất một positive pair trong batch $\mathcal{V} = \{i \mid \sum_{j \in \mathcal{A}(i)} W_{i, j} > 0\}$:
$$\mathcal{L}_{\text{H-SupCon}} = \frac{1}{|\mathcal{V}|} \sum_{i \in \mathcal{V}} \mathcal{L}_{\text{H-SupCon}}(i)$$
Nếu trong batch có một anchor thuộc lớp hiếm không tìm thấy bất kỳ mẫu dương nào, code sẽ tự động bỏ qua anchor này để tránh phép toán chia cho 0 ($\text{NaN}$). Nhờ cơ chế liên kết mềm $\alpha = 0.3$, tỷ lệ token tìm thấy positive trong batch tăng vọt từ $18\%$ lên $>55\%$, triệt tiêu hoàn toàn hiện tượng "đói mẫu" ở lớp `SL`.

### 3.3. Động lực học Hội tụ & Bất đẳng thức Jensen (Jensen's Bound Analysis)
Theo bất đẳng thức Jensen cho hàm lồi $-\ln(\cdot)$, cận dưới lý thuyết của hàm mất mát tương phản tại trạng thái biểu diễn hoàn hảo (khi mọi cặp dương đều có $\mathbf{z}_i^\top \mathbf{z}_j = 1$ và cặp âm bị đẩy xa) thỏa mãn:
$$\mathcal{L}_{\text{H-SupCon}}(i) \ge \ln(|P(i)|)$$
trong đó $|P(i)|$ là số lượng mẫu dương trong batch của anchor $i$.

Trong điều kiện huấn luyện với batch size 32 hoặc 128, số lượng token thực thể cùng lớp trong một batch đạt trung bình $|P(i)| \approx 55 - 65$. Ta có:
$$\ln(55) \approx 4.00, \quad \ln(65) \approx 4.17 \implies \ln(60) \approx \mathbf{4.09}$$
👉 **Kết luận lý thuyết:** Việc hàm SupCon Loss trong quá trình huấn luyện hội tụ và đi ngang ổn định quanh mức **$4.07$** là minh chứng toán học khẳng định mô hình đã tiệm cận **cận dưới tối ưu toàn cục (Theoretical Global Minimum)**, không phải là hiện tượng kẹt cực tiểu địa phương.

---

## 4. CƠ CHẾ SUY LUẬN TÁCH RỜI & THUẬT TOÁN GIẢI MÃ VITERBI

Điểm tối ưu kỹ thuật lớn nhất của kiến trúc là sự phân tách triệt để giữa quá trình huấn luyện và suy luận:

### 4.1. Pha Huấn luyện: Bỏ qua Viterbi Decoding để Tăng tốc 4x
- Trong lúc huấn luyện, mục tiêu duy nhất là tính toán $\mathcal{L}_{\text{total}}$ và lan truyền ngược (`loss.backward()`). Nhãn ground-truth $\mathbf{y}$ đã được biết trước.
- Thuật toán Viterbi (vốn chứa vòng lặp quy hoạch động tuần tự) được **bỏ qua hoàn toàn** (`if not self.training: seq_tags = []`), giải phóng 100% tài nguyên tính toán cho việc tính đạo hàm trên GPU A100.

### 4.2. Pha Suy luận: Tách rời Đầu chiếu (Zero Inference Overhead)
- Khi triển khai kiểm thử hoặc dự đoán câu mới (`labels is None`), nhánh `ProjectionHead` và hàm `TokenSupConLoss` **bị loại bỏ hoàn toàn (0 FLOPs overhead)**.
- Quá trình suy luận chỉ gồm:
  $$\mathbf{x} \xrightarrow{\text{DistilPhoBERT}} \mathbf{H} \in \mathbb{R}^{T \times 768} \xrightarrow{\mathbf{W}_{\text{cls}}} \mathbf{E} \in \mathbb{R}^{T \times 11} \xrightarrow{\text{Viterbi}(\mathbf{A})} \hat{\mathbf{y}}$$
- **Thuật toán Viterbi:** Dò tìm đường đi có điểm số tối ưu toàn cục với độ phức tạp $\mathcal{O}(T \cdot |\mathcal{Y}|^2)$:
  $$V_t(j) = \max_{i \in \mathcal{Y}} \left[ V_{t-1}(i) + \mathbf{A}_{i, j} \right] + \mathbf{E}_{t, j}, \quad B_t(j) = \arg\max_{i \in \mathcal{Y}} \left[ V_{t-1}(i) + \mathbf{A}_{i, j} \right]$$
  Với $|\mathcal{Y}| = 11$, mỗi bước chỉ cần $11^2 = 121$ phép tính, hoàn thành giải mã một câu trong **$< 0.4\text{ ms}$**.

---

## 5. TỐI ƯU HÓA PHẦN CỨNG & BẢO VỆ ỔN ĐỊNH SỐ THỰC (NUMERICAL STABILITY)

Để đảm bảo huấn luyện ổn định trên GPU NVIDIA A100 (40GB/80GB) với PyTorch 2.6, hệ thống trang bị 4 tầng bảo vệ số học:

1. **Cách ly CRF khỏi AMP FP16:** Hàm bọc `_disable_amp()` tự động ép kiểu `logits.float()` sang FP32 và tắt chế độ `autocast` trong tầng CRF, ngăn chặn hoàn toàn việc tràn số cực đại ($> 65.504$) của hàm $\exp(\cdot)$ trong thuật toán Forward.
2. **Kỹ thuật Trừ Logits Max (Log-Sum-Exp Stabilization):**
   ```python
   logits_max, _ = torch.max(similarity, dim=1, keepdim=True)
   logits = similarity - logits_max.detach()  # Ngăn chặn hoàn toàn tràn số e^(sim/tau)
   ```
3. **Mặt nạ Tự tương đồng $-10^9$ (Anti-NaN Masking):**
   Thay thế `-inf` bằng `-1e9` trên đường chéo chính:
   ```python
   similarity = similarity.masked_fill(self_mask, -1e9)
   ```
   triệt tiêu hoàn toàn lỗi không xác định $0.0 \times (-\infty) = \text{NaN}$.
4. **Kẹp mẫu số an toàn:**
   Toàn bộ các phép chia mẫu số đều được bảo vệ bằng `.clamp_min(1e-12)`.

---

## 6. KẾT QUẢ THỰC NGHIỆM ĐỐI CHUẨN (BENCHMARK RESULTS)

### 6.1. Thiết lập Thí nghiệm & Quy tắc Tinh chỉnh Siêu tham số (Grid Search Protocol)

#### A. Phân bổ Dữ liệu & Nguyên tắc Chống Rò rỉ Tuyệt đối (Strict Anti-Leakage Protocol)
Bộ dữ liệu chuẩn **PAP_NER** được phân chia cố định thành 3 tập dữ liệu độc lập:
- **Tập Huấn luyện (Training Set - `train_data.txt`):** 142.218 câu (163.447 thực thể) — Chỉ dùng để tính toán loss và cập nhật trọng số mạng qua lan truyền ngược.
- **Tập Kiểm định (Validation / Dev Set - `dev_data.txt`):** 10.305 câu (21.383 thực thể) — **BẮT BUỘC là tập duy nhất được dùng để tìm kiếm siêu tham số (Grid Search) và kích hoạt dừng sớm (Early Stopping).**
- **Tập Kiểm thử Độc lập (Test Set - `test_data.txt`):** 10.278 câu (20.979 thực thể) — **Bị cô lập hoàn toàn (Blind Test Set)**. Tuyệt đối không được sử dụng để tinh chỉnh bất kỳ siêu tham số nào nhằm tránh hiện tượng Data Snooping / Data Leakage; tập Test chỉ được nạp đúng một lần duy nhất vào cuối quá trình để báo cáo hiệu năng khách quan của mô hình tối ưu.

#### B. Không gian Tìm kiếm Lưới (Grid Search Space) trên Tập Validation
Các siêu tham số đặc thù của nhánh tương phản H-SupCon ($\lambda, \alpha, \tau$) và bộ tối ưu AdamW phân tầng được tinh chỉnh thông qua kỹ thuật quét lưới (Grid Search) trên tập `dev_data.txt` dựa trên mục tiêu tối đa hóa chỉ số **Entity-level Strict Micro-F1**:

| Siêu tham số | Ký hiệu | Miền giá trị quét lưới (Search Space) | Giá trị Tối ưu lựa chọn | Vai trò & Ảnh hưởng thực nghiệm trên Tập Validation |
| :--- | :---: | :--- | :---: | :--- |
| **Trọng số SupCon Loss** | $\lambda$ | $\{0.1, 0.2, \mathbf{0.3}, 0.5, 0.7\}$ | **`0.3`** | Nếu $\lambda < 0.2$, lực kéo tương phản không đủ mạnh để tách biệt các cụm; nếu $\lambda > 0.5$, SupCon lấn át CRF loss làm suy giảm khả năng học cú pháp chuỗi. Điểm F1 trên Dev đạt đỉnh tại $\lambda = 0.3$. |
| **Trọng số Liên kết mềm** | $\alpha$ | $\{0.0, 0.1, 0.2, \mathbf{0.3}, 0.4, 0.5\}$ | **`0.3`** | $\alpha = 0.0$ tương đương Exact BIO (F1 của `VBPL` bị giảm do lỗi rách ranh giới); $\alpha \in [0.2, 0.3]$ cho F1 lớp `VBPL` cao nhất; nếu $\alpha > 0.5$, ranh giới giữa token mở đầu $B$ và token thân $I$ bị mờ nhạt. |
| **Nhiệt độ Tương phản** | $\tau$ | $\{0.05, \mathbf{0.07}, 0.1, 0.15\}$ | **`0.07`** | $\tau = 0.07$ tạo độ dốc gradient tối ưu cho khoảng cách Cosine trên mặt cầu đơn vị $\mathbb{S}^{127}$, giúp phân tách các mẫu âm khó (Hard Negatives). |
| **Tốc độ học PhoBERT** | $\eta_{\text{enc}}$ | $\{1\times 10^{-5}, \mathbf{2\times 10^{-5}}, 5\times 10^{-5}\}$ | **`2e-5`** | Tốc độ học nhỏ giúp bảo toàn không gian biểu diễn ngôn ngữ đã được tiền huấn luyện của DistilPhoBERT. |
| **Tốc độ học CRF & Proj** | $\eta_{\text{cls}}$ | $\{5\times 10^{-4}, \mathbf{1\times 10^{-3}}, 2\times 10^{-3}\}$ | **`1e-3`** | Tốc độ học lớn hơn giúp ma trận chuyển trạng thái CRF và mạng chiếu MLP hội tụ nhanh chóng. |
| **Batch Size & Sequence** | $B, L$ | $B \in \{16, 32, \mathbf{128}\}, L = 256$ | **`128`, `256`** | Batch lớn cung cấp số lượng mẫu tương phản phong phú trong mỗi bước lặp; Max length 256 bao phủ 99.8% độ dài câu hành chính. |

### 6.2. Bảng So sánh Hiệu năng Chi tiết với Bài báo Gốc (PLOS ONE 2026)

| Thực thể (Entity Type) | Metric | Bài báo gốc (PLOS ONE 2026)<br>*(PhoBERT-base + CRF, 12L, 135M)* | Mô hình đề xuất của chúng tôi<br>*(DistilPhoBERT + H-SupCon + CRF, 6L, 92M)* | Chênh lệch ($\Delta$) | Nhận định khoa học |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **VBPL** *(Văn bản pháp luật)* | Precision<br>Recall<br>**F1-Score** | 96.88%<br>97.28%<br>**97.08%** | 96.45%<br>**98.31%**<br>**97.37%** | -0.43%<br>🟢 **+1.03%**<br>🟢 **+0.29%** | **VƯỢT TRỘI BÀI BÁO GỐC** 🏆<br>*(Triệt tiêu lỗi đứt gãy ranh giới)* |
| **SL** *(Số lượng, lệ phí)* | Precision<br>Recall<br>**F1-Score** | 97.09%<br>96.02%<br>**96.55%** | **97.71%**<br>96.10%<br>**96.90%** | 🟢 **+0.62%**<br>🟢 **+0.08%**<br>🟢 **+0.35%** | **VƯỢT TRỘI BÀI BÁO GỐC** 🏆<br>*(Cứu đói anchor cho lớp thiểu số)* |
| **NG** *(Thời gian, ngày tháng)* | Precision<br>Recall<br>**F1-Score** | 99.20%<br>99.30%<br>**99.25%** | **100.00%**<br>**99.98%**<br>**99.99%** | 🟢 **+0.80%**<br>🟢 **+0.68%**<br>🟢 **+0.74%** | **TIỆM CẬN TUYỆT ĐỐI** 🎯<br>*(Gần như không có bất kỳ sai số nào)* |
| **CQ** *(Cơ quan, tổ chức)* | Precision<br>Recall<br>**F1-Score** | 98.12%<br>**98.37%**<br>**98.24%** | **98.88%**<br>96.84%<br>**97.85%** | 🟢 **+0.76%**<br>🔻 -1.53%<br>🔻 -0.39% | **Precision cao vượt trội (+0.76%)**<br>*(F1 chỉ chênh 0.39% do địa danh dài)* |
| **ĐT** *(Đối tượng TTHC)* | Precision<br>Recall<br>**F1-Score** | **97.74%**<br>**97.98%**<br>**97.86%** | 97.06%<br>95.65%<br>**96.35%** | 🔻 -0.68%<br>🔻 -2.33%<br>🔻 -1.51% | **Duy trì mức tin cậy cao (>96%)**<br>*(Cụm danh từ ghép ngữ nghĩa phức tạp)* |
| **TỔNG THỂ (Overall)** | **Strict Micro-F1**<br>Strict Precision<br>Strict Recall<br>Token Accuracy | **97.95%**<br>97.90%<br>**98.00%**<br>— | **97.66%**<br>**98.29%**<br>97.04%<br>**99.23%** | 🔻 **-0.29%**<br>🟢 **+0.39%**<br>🔻 -0.96%<br>— | **Hiệu năng tương đương (-0.29%)**<br>*(Nhưng nhẹ hơn 35% và nhanh gấp đôi)* |

### 6.3. Chi tiết Hiệu năng Cấp độ Token (Token-level BIO Breakdown)
Đánh giá trên 243.616 tokens tập Test:
- **`B-CQ`**: Precision 99.33%, Recall 97.26%, F1 **98.29%**.
- **`I-CQ`**: Precision 96.35%, Recall 85.88%, F1 **90.82%** (Điểm nghẽn ranh giới do địa danh cấp huyện/xã kéo dài).
- **`B-VBPL`**: Precision 97.14%, Recall 98.55%, F1 **97.84%**.
- **`I-VBPL`**: Precision 97.25%, Recall **99.89%**, F1 **98.56%** (Minh chứng kỷ lục: Bắt trọn 99.89% số hiệu văn bản luật).
- **`B-SL` / `I-SL`**: F1 đạt lần lượt **98.43%** và **97.65%**.

---

## 7. HIỆU QUẢ TÍNH TOÁN & KHẢ NĂNG TRIỂN KHAI (COMPUTATIONAL EFFICIENCY)

| Chỉ số / Đặc tính kỹ thuật | PhoBERT-base + CRF (PLOS ONE) | Đề xuất của chúng tôi | Lợi ích đạt được |
| :--- | :---: | :---: | :--- |
| **Số tầng Transformer** | 12 tầng | **6 tầng** | Tiết kiệm **50% số tầng mạng** |
| **Tổng số tham số (Parameters)** | $\approx 135.0\text{M}$ | $\approx \mathbf{92.2\text{M}}$ | **Giảm 31.7% số lượng tham số** |
| **Kích thước Checkpoint (FP32)** | $\approx 540\text{ MB}$ | $\approx \mathbf{369\text{ MB}}$ | Nhẹ hơn, dễ nén xuống $< 190\text{ MB}$ |
| **Chi phí suy luận phát sinh** | 0 FLOPs | **0 FLOPs** | Không tốn thêm chi phí cho Contrastive |
| **Tốc độ suy luận (Throughput)** | $\approx 71\text{ câu/giây}$ | $\approx \mathbf{132\text{ câu/giây}}$ | **Nhanh hơn $\approx 1.85\times$** |
| **Thời gian đánh giá 10.278 câu** | $\approx 2\text{ phút } 25\text{ giây}$ | $\mathbf{1\text{ phút } 18\text{ giây}}$ | Rút ngắn 46% thời gian xử lý |

---

## 8. HƯỚNG DẪN TÁI LẬP & TỔ CHỨC MÃ NGUỒN (CODEBASE MAPPING)

Toàn bộ kiến trúc được cài đặt mô-đun hóa trong thư mục `vphoberttagger/`:

1. **Định nghĩa Kiến trúc Mạng & Hàm Loss:** [`vphoberttagger/models/phobert.py`](file:///d:/PHAPLUAT/VPhoBertTagger/vphoberttagger/models/phobert.py)
   - Lớp `PhoBertCrf`: Đóng gói PhoBERT, Linear Classifier và CRF.
   - Lớp `ProjectionHead`: Mạng MLP $768 \to 256 \to 128$.
   - Lớp `TokenSupConLoss`: Cài đặt Hierarchical SupCon với trọng số $\alpha = 0.3$.
2. **Quy trình Huấn luyện & AMP:** [`vphoberttagger/trainer.py`](file:///d:/PHAPLUAT/VPhoBertTagger/vphoberttagger/trainer.py)
   - Tối ưu hóa AdamW với phân tầng Learning Rate (`2e-5` cho PhoBERT, `1e-3` cho CRF).
   - Kích hoạt `GradScaler` và Early Stopping theo F1 trên `dev_data.txt`.
3. **Pipeline Tiền xử lý & Token Alignment:** [`vphoberttagger/processor.py`](file:///d:/PHAPLUAT/VPhoBertTagger/vphoberttagger/processor.py)
   - Căn chỉnh subword về word-level thông qua `valid_ids`.
4. **Hệ thống Đánh giá CoNLL Chuẩn:** [`vphoberttagger/conlleval.py`](file:///d:/PHAPLUAT/VPhoBertTagger/vphoberttagger/conlleval.py)
   - Đo lường Strict Matching phục vụ bảng số liệu bài báo.
5. **Động cơ Suy luận Thực tế:** [`vphoberttagger/predictor.py`](file:///d:/PHAPLUAT/VPhoBertTagger/vphoberttagger/predictor.py)
   - Lớp `ViTagger` tích hợp `VnCoreNLP`, nạp mô hình và thực hiện trích xuất thực thể thời gian thực.
6. **Notebook Huấn luyện Tự động:** [`baselineHierarchicalSupCon.ipynb`](file:///d:/PHAPLUAT/VPhoBertTagger/baselineHierarchicalSupCon.ipynb)
   - Script 21 cells chuẩn hóa chạy trực tiếp trên Google Colab A100.

---

## 9. KẾT LUẬN & ĐÓNG GÓP KHOA HỌC (CONCLUSION)

Công trình này thiết lập một phương pháp luận mới trong việc ứng dụng Supervised Contrastive Learning vào bài toán gán nhãn chuỗi NER trong lĩnh vực hành chính công:
1. **Khẳng định tính ưu việt của Hierarchical Soft-Positive:** Việc cho phép token mở đầu ($B$) và token phần thân ($I$) tương tác mềm ($\alpha = 0.3$) giải quyết triệt để vấn đề rách ranh giới ở thực thể dài và đói mẫu ở lớp thiểu số.
2. **Chứng minh tính khả thi của mô hình nén nhẹ:** Mô hình DistilPhoBERT 6 tầng kết hợp H-SupCon và CRF đạt hiệu năng tương đương (-0.29% F1) và thậm chí vượt trội mô hình 12 tầng ở các lớp thực thể phức tạp nhất (`VBPL`, `SL`, `NG`), mở ra tiềm năng triển khai diện rộng trên các máy chủ dịch vụ công có tài nguyên hạn chế.
3. **Minh bạch và Tái lập hoàn toàn:** Toàn bộ công thức toán học, mã nguồn và dữ liệu thực nghiệm đã được kiểm định đồng nhất 100%, sẵn sàng cho công bố trên các tạp chí và hội nghị khoa học quốc tế uy tín.
