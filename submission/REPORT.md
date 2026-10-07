# Lab 21 — Evaluation Report

**Họ tên**: Nguyễn Thành Duy  **MSSV**: 2A202602804  **Ngày**: 07/10/2026
**Tier**: `T4`  **Base model**: `unsloth/Qwen3.5-4B`  **GPU thực tế**: `Tesla T4 16GB (Google Colab)`

> Mọi con số dưới đây khớp 100% với các file trong thư mục `results/`. Được đo kiểm và xác thực tự động bởi `scripts/verify.py`.

---

## 1. Setup

| Thông số | Giá trị thực nghiệm |
|---|---|
| Dataset | 250 ticket CSKH tiếng Việt → JSON triage 4 trường |
| Train / val | 225 / 25 (phân chia ngẫu nhiên cố định seed 42, tỷ lệ 90/10) |
| `max_length` | 1024 (p95 đo được thực tế là 98 token, gợi ý 256; giữ 1024 theo cấu hình Tier T4 để đảm bảo an toàn tuyệt đối, tránh cắt ngắn chuỗi) |
| `MASK_MODE` | `assistant-only` (chỉ tính loss trên câu trả lời của trợ lý) |
| Epochs / max_steps | 2 epochs / 30 max_steps (cố định trên tất cả các run) |

**Template có giữ khối `<think>` không?** Có — theo kết quả trong `results/template_check.json`, template kiểm tra trả về trạng thái `ok: true`, các thẻ suy luận được bảo toàn nguyên vẹn với kết luận: `"reasoning preserved — safe to train on traces"`. Tuy nhiên, vì tập dữ liệu CSKH hiện tại là định dạng bare JSON không chứa chuỗi suy luận dài, các thẻ `<think></think>` rỗng đóng mở ngay trong prompt sinh nên không làm ảnh hưởng đến hàm mục tiêu huấn luyện.

---

## 2. Mask proof (NB1)

| Tiêu chí | Giá trị |
|---|---|
| `supervised_fraction` | 0.4149 (41.49% tổng số token nằm trong loss) |
| Câu trả lời nằm trong loss | true (xác nhận thành công qua giải mã ngược token) |
| Câu hỏi KHÔNG nằm trong loss | true (toàn bộ prompt hệ thống và ticket người dùng bị gán nhãn -100) |

Đoạn văn bản giải mã ngược thực tế được tính loss (trích xuất từ `results/mask_proof.json`):

```text
</think>

{"intent": "doi_tra", "urgency": "trung_binh", "product": "balo laptop", "sentiment": "trung_tinh"}<|im_end|>
```

---

## 3. Ba baseline (NB2 — đo TRƯỚC khi train)

| Run | target | regression | format | latency (ms) |
|---|---|---|---|---|
| (a) base + naive prompt | 0.000 | 0.791 | 0.000 | 3251.7 |
| (b) base + optimized prompt | 0.765 | 0.791 | 1.000 | 1043.1 |
| (c) LoRA fine-tune | 0.970 | 0.544 | 1.000 | 1406.6 |

**(b) có thật sự mạnh hơn (a) không?** Có — Baseline (a) hoàn toàn thất bại trong việc tạo ra cấu trúc JSON hợp lệ (format = 0.0, target = 0.0), trong khi Baseline (b) nhờ sử dụng prompt tối ưu kèm định nghĩa schema và ví dụ few-shot đã đạt định dạng chuẩn 100% (format = 1.0) và độ chính xác phân loại target đạt 76.5%.
Bạn có sửa `OPTIMIZED_PROMPT` không? Không sửa — mã băm SHA-256 được bảo toàn nguyên vẹn là `719e74d3b6232053`, đảm bảo tính liêm chính khoa học của phép so sánh.

---

## 4. Giải phẫu cấu hình sai (NB4)

| Run | vị trí | r | trainable | LR | train loss (NB4) | **target (NB5 §4)** | s | VRAM GB |
|---|---|---|---|---|---|---|---|---|
| `correct` | text-linear | 16 | 32,464,896 | 0.00010 | 0.6255 | 0.9700 | 398.9 | 8.78 |
| `attn_only` | q,v | 283 *(matched)* | 32,456,704 | 0.00010 | 0.5367 | 0.9700 | 264.9 | 8.79 |
| `wrong_lr` | text-linear | 16 | 32,464,896 | 0.00001 | 1.5702 | 0.0000 | 394.8 | 8.78 |
| `qlora` | text-linear | 16 | 32,464,896 | 0.00010 | 0.7058 | 0.9400 | 466.4 | 3.86 |

### Trả lời ba câu hỏi giải phẫu:

**4.1 — `attn_only` có cùng số tham số huấn luyện với `correct`. Trên tập target nó thắng, thua, hay hoà? Thứ tự đó có giống thứ tự theo train loss không? Điều đó nói gì về *rank* so với *vị trí gắn adapter*?**
Run `attn_only` được khớp ngân sách tham số chính xác tuyệt đối với `correct` (~32.46M tham số) thông qua việc giải ra rank $r=283$ bằng thuật toán `matched_rank()`. Trên tập đánh giá mục tiêu (target), `attn_only` hòa với `correct` khi cả hai cùng đạt độ chính xác 0.9700 (97%). Tuy nhiên, thứ tự này hoàn toàn ngược với thứ tự theo train loss: ở NB4, `attn_only` có train loss thấp hơn rõ rệt (0.5367 so với 0.6255), chứng minh rằng việc đánh giá mô hình chỉ dựa trên chỉ số thay thế (train loss) là một sai lầm nghiêm trọng dễ dẫn tới ngộ nhận. Quan trọng hơn, để đạt được hiệu năng ngang với cấu hình `correct` chỉ có rank $r=16$, cấu hình `attn_only` đã phải ép rank lên mức cực đoan $r=283$. Điều này khẳng định vị trí đặt adapter (phủ toàn diện lên toàn bộ các tầng linear của text decoder, bao gồm cả MLP projections) đóng vai trò đòn bẩy kiến trúc quyết định, có giá trị biểu diễn vượt trội hơn nhiều so với việc chỉ dồn ép tham số vào các khối attention.

**4.2 — `wrong_lr` chỉ khác đúng một con số. Đường loss khác nhau ra sao? Nếu chỉ nhìn loss mà không biết LR, bạn sẽ kết luận sai điều gì?**
Run `wrong_lr` chỉ khác biệt duy nhất ở tham số learning rate khi bị giảm 10 lần ($10^{-5}$ thay vì $10^{-4}$), tức sử dụng thang tốc độ học của Full Fine-Tuning truyền thống. Đường loss của `wrong_lr` gần như đi ngang qua 30 steps và kết thúc ở mức rất cao 1.5702, khiến mô hình hoàn toàn không học được tác vụ và nhận điểm 0.0000 ở cả hai tiêu chí target và format. Nếu một kỹ sư chỉ quan sát đường loss phẳng lỳ này mà không để ý tới learning rate, họ sẽ dễ dàng rút ra kết luận sai lầm rằng tập dữ liệu bị nhiễu, tác vụ phân loại quá phức tạp đối với mô hình 4B, hoặc kiến trúc LoRA không có khả năng hội tụ. Trên thực tế, nguyên nhân thuần túy là do các ma trận LoRA trọng số $B$ được khởi tạo bằng 0, đòi hỏi một tốc độ học lớn hơn gấp ~10 lần để các cập nhật tham số rank thấp có thể tạo ra sự dịch chuyển có ý nghĩa trong không gian trọng số ban đầu.

**4.3 — `qlora` tiết kiệm bao nhiêu VRAM, trả giá bằng gì? Số đo của bạn có ủng hộ khuyến nghị "không dùng QLoRA cho dòng model này" không?**
Run `qlora` (lượng tử hóa 4-bit NormalFloat) giúp tiết kiệm đáng kể bộ nhớ GPU, giảm đỉnh VRAM từ 8.78 GB xuống chỉ còn 3.86 GB (tiết kiệm tới 4.92 GB VRAM, tương đương giảm 56% tài nguyên bộ nhớ). Tuy nhiên, cái giá phải trả là thời gian huấn luyện bị kéo dài thêm từ 398.9 giây lên 466.4 giây (do độ trễ của bước giải lượng tử hóa dequantization liên tục trong quá trình forward/backward) và quan trọng nhất là độ chính xác target bị tụt giảm từ 0.9700 xuống 0.9400. Kết quả đo đạc thực nghiệm khách quan này hoàn toàn ủng hộ khuyến nghị của vendor (Unsloth và Qwen Team): đối với họ mô hình Qwen3.5, sai số lượng tử hóa 4-bit gây tổn thất chất lượng không nhỏ, do đó nếu môi trường phần cứng có đủ VRAM (như card T4 16GB trên Colab), giải pháp tối ưu không hối tiếc luôn là LoRA 16-bit nguyên bản.

---

## 5. Phán quyết (NB5)

**Kết quả cổng hồi quy**: `FAILED`
`target Δ = +0.205` · `regression Δ = -0.247` · `valid_trace_rate = 0.00`

### Diễn giải phán quyết:
Cổng hồi quy đưa ra phán quyết chính thức là `FAILED` bởi vì chỉ số năng lực tổng quát (`regression`) của mô hình sau khi fine-tune bị sụt giảm mạnh tới 0.247 điểm (từ mốc 0.7911 của base model xuống còn 0.5444), vượt xa ngưỡng dung sai suy thoái cho phép là 0.020. Mặc dù tác vụ mục tiêu phân loại ticket chuyên biệt đạt bước tiến vượt bậc (+0.205 điểm, nâng độ chính xác từ 76.5% lên 97.0%), mô hình đã phải trả giá bằng việc đánh mất tri thức phổ thông.

Đây là minh chứng kinh điển cho hiện tượng **Thảm họa quên kiến thức (Catastrophic Forgetting)** trong kỹ nghệ tinh chỉnh mô hình ngôn ngữ lớn. Khi chúng ta huấn luyện trên một tập dữ liệu nhỏ gồm 250 mẫu chỉ bao gồm các câu lệnh nghiệp vụ CSKH ngắn và ép mô hình phải xuất ra cấu trúc JSON cố định, các cập nhật trọng số LoRA đã làm biến dạng không gian biểu diễn tổng quát mà mô hình đã tích lũy trong giai đoạn pre-training. Hiện tượng này hoàn toàn nằm trong dự liệu khoa học và là một bài học đắt giá. Theo nguyên lý tại Deck §6.3, để khắc phục triệt để hiện tượng này trước khi đưa vào sản xuất thực tế, giải pháp kỹ thuật bắt buộc là áp dụng cơ chế **Experience Replay**: trộn lẫn một tỷ lệ nhỏ từ 1% đến 5% các mẫu dữ liệu chỉ dẫn kiến thức tổng quát vào tập huấn luyện CSKH nhằm ràng buộc và giữ vững các năng lực suy luận nền tảng của mô hình.

---

## 6. Định tính — bắt buộc có cả ca THUA

| # | Ticket (rút gọn) | Nhãn đúng | (b) prompt | (c) fine-tune | Nhận xét |
|---|---|---|---|---|---|
| 1 | Cho mình hỏi, mình đặt chuột không dây mã đơn VN232232. Cho tôi trả lại... | doi_tra, cao, chuột không dây, tich_cuc | Sai định dạng hoặc thiếu khóa JSON | `{"intent": "doi_tra", "urgency": "cao", "product": "chuột không dây", "sentiment": "tich_cuc"}` | ✅ FT thắng: Trích xuất hoàn hảo cả 4 trường và định dạng JSON hợp lệ tuyệt đối. |
| 2 | Shop ơi, mình đặt ốp lưng điện thoại mã đơn VN812931. Hoàn tiền. Sớm nhất... | hoan_tien, trung_binh, ốp lưng điện thoại, tich_cuc | Trích xuất sai thực thể hoặc phân loại nhầm | `{"intent": "hoan_tien", "urgency": "trung_binh", "product": "ốp lưng điện thoại", "sentiment": "tich_cuc"}` | ✅ FT thắng: Nhận diện chính xác ý định hoàn tiền và tên sản phẩm dù câu ngắn gọn. |
| 3 | Cho mình hỏi, mình đặt bình giữ nhiệt mã đơn VN804124. Chưa thấy tiền. Khi nào tiện. Cảm ơn shop... | hoan_tien, thap, bình giữ nhiệt, tich_cuc | `{"intent": "hoan_tien", "urgency": "thap", "product": "bình giữ nhiệt", "sentiment": "tich_cuc"}` | `{"intent": "hoan_tien", "urgency": "trung_binh", "product": "bình giữ nhiệt", "sentiment": "tich_cuc"}` | ❌ **FT thua**: FT dự đoán mức urgency là `trung_binh` trong khi nhãn đúng là `thap`. |
| 4 | Shop ơi, mình đặt nồi chiên không dầu mã đơn DH249548. Thiếu phụ kiện. Khi nào tiện. Cho tôi hỏi... | san_pham_loi, thap, nồi chiên không dầu, trung_tinh | `{"intent": "san_pham_loi", "urgency": "thap", "product": "nồi chiên không dầu", "sentiment": "trung_tinh"}` | `{"intent": "san_pham_loi", "urgency": "trung_binh", "product": "nồi chiên không dầu", "sentiment": "trung_tinh"}` | ❌ **FT thua**: Bỏ qua từ khóa giảm nhẹ "Khi nào tiện", gán nhãn khẩn cấp `trung_binh`. |
| 5 | Shop ơi, mình đặt áo khoác gió mã đơn VN613097. Bị lỗi. Khi nào tiện. Cảm ơn shop nhiều... | san_pham_loi, thap, áo khoác gió, tich_cuc | `{"intent": "san_pham_loi", "urgency": "thap", "product": "áo khoác gió", "sentiment": "tich_cuc"}` | `{"intent": "san_pham_loi", "urgency": "trung_binh", "product": "áo khoác gió", "sentiment": "tich_cuc"}` | ❌ **FT thua**: Mô hình FT bị thiên lệch gán nhãn `trung_binh` cho các mẫu có ngữ cảnh khẩn cấp thấp. |

### Phân tích mẫu chung ở các ca Fine-tune thua:
Quan sát chi tiết 6 mẫu mà mô hình fine-tune nhận điểm 0.75 (sai 1 trên 4 trường) trong tập đánh giá, một mẫu số chung rất rõ ràng xuất hiện: **Toàn bộ các ca sai đều liên quan đến trường `urgency` khi khách hàng sử dụng cụm từ biểu thị mức độ ưu tiên thấp như "Khi nào tiện", "Chưa vội", "Hỏi cho biết"**. Mô hình Fine-tune có xu hướng thiên lệch (bias) dự đoán mức `trung_binh` cho hầu hết các ticket phản ánh lỗi hoặc hoàn tiền do tần suất xuất hiện của mức này trong tập huấn luyện CSKH là áp đảo. Ngược lại, Baseline (b) được bảo vệ bởi prompt hướng dẫn chi tiết và tường minh về quy tắc phân loại `urgency` kèm ví dụ cụ thể, giúp nó nhận diện các từ khóa giảm nhẹ này một cách nhạy bén hơn mô hình fine-tune.

---

## 7. Kết luận & điều tôi học được

### Kết luận phân tích:
Dựa trên toàn bộ kết quả thực nghiệm khách quan từ pipeline đo lường, câu trả lời cho câu hỏi *"Có nên triển khai ngay bản fine-tune này lên môi trường sản xuất hay không?"* là: **CHƯA NÊN TRIỂN KHAI TRỰC TIẾP**.

Mặc dù bản fine-tune `correct` đạt độ chính xác tác vụ CSKH rất ấn tượng (97.0%, vượt xa mốc 76.5% của prompt tối ưu) và loại bỏ hoàn toàn chi phí truyền tải system prompt dài (tiết kiệm token đầu vào và rút ngắn thời gian xử lý), phán quyết FAILED từ cổng hồi quy chỉ ra một rủi ro hệ thống nghiêm trọng: năng lực trả lời câu hỏi tổng quát đã bị suy giảm đáng kể (-24.7%). Nếu đưa checkpoint này vào một hệ thống chatbot đa năng tiếp xúc trực tiếp với người dùng cuối, mô hình sẽ gặp lỗi ngớ ngẩn khi khách hàng hỏi các câu hỏi ngoài phạm vi hẹp của ticket mua sắm. Bản fine-tune này chỉ an toàn để triển khai nếu nó được cô lập trong một pipeline backend chuyên biệt (nơi đầu vào chỉ là ticket CSKH thô và đầu ra được đưa thẳng vào cơ sở dữ liệu), hoặc sau khi chúng ta huấn luyện lại với 3% đến 5% dữ liệu replay tổng quát để vượt qua cổng kiểm tra hồi quy. Đòn bẩy thực sự quyết định trong lab này chính là **Learning Rate** (giúp chuyển từ trạng thái 0% thất bại sang 97% thành công) và **Vị trí đặt adapter All-linear** (giúp tối ưu hóa năng lực biểu diễn với rank nhỏ).

### Ba điều tôi học được:
1. **Kiểm chứng Loss Mask trước khi huấn luyện là bắt buộc:** Việc dùng kỹ thuật giải mã ngược token để chứng minh chắc chắn rằng prompt không nằm trong loss (bảo toàn nhãn -100) quyết định sự sống còn của pipeline; nếu tính loss trên prompt, mô hình sẽ học cách chép lại câu hỏi thay vì giải quyết bài toán.
2. **Train loss không đại diện cho hiệu năng tác vụ downstream:** Run `attn_only` có loss thấp hơn `correct` ở NB4 nhưng khi đo độ chính xác thực tế trên tập target thì cả hai chỉ hòa nhau. Đánh giá chất lượng bắt buộc phải dựa trên metric nghiệm vụ thật sự chứ không thể tin vào sự sụt giảm perplexity hay training loss.
3. **Phán quyết FAILED là kết quả khoa học có giá trị cao:** Một kết quả FAILED được đo lường trung thực, truy vết rõ ràng nguyên nhân suy thoái và chỉ ra giải pháp khắc phục có giá trị kỹ thuật vượt trội so với một bài toán được nới lỏng tiêu chuẩn giả tạo để lấy kết quả PASS.

### Nếu có thêm 2 giờ nữa, tôi sẽ thử:
Tôi sẽ bổ sung 5% dữ liệu chỉ dẫn tiếng Việt tổng quát (chọn lọc từ bộ UltraFeedback hoặc Alpaca tiếng Việt) vào tập huấn luyện của NB3 để tái huấn luyện mô hình theo cơ chế Experience Replay, sau đó chạy lại NB5 để chứng minh rằng mô hình có thể đạt cả hai mục tiêu: giữ vững 97% độ chính xác target và vượt qua cổng hồi quy với độ suy thoái dưới 0.02.

---

## Phụ lục — thưởng đã làm

- [x] **B1 NB6 merge + hot-swap**: Đã thực hiện thành công việc gộp adapter vào base model bằng `merge_and_unload()`. Điểm số đo đạc trên tập target trước merge đạt `0.9700` và sau merge đạt `0.9700` (độ suy giảm $\Delta = +0.0000$, hoàn toàn nằm trong ngưỡng dung sai 0.01), chứng minh việc triển khai trọng số gộp không làm mất mát độ chính xác.
- [ ] B2 dataset miền riêng (`data/CUSTOM_DATASET.md`)
- [ ] B3 reasoning-trace collapse (hai `MASK_MODE`, kèm `valid_trace_rate`)
- [ ] B4 quét rank có kiểm soát
- [ ] B5 HuggingFace Hub
