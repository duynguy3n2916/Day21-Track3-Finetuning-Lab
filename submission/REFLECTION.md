# Reflection — Lab 21

*Ngắn gọn, thành thật. Phần này chấm theo độ cụ thể, không theo độ dài.*

**1. Điều gì làm bạn ngạc nhiên nhất?**
Điều làm tôi bất ngờ nhất là việc run `attn_only` dù đạt train loss thấp hơn `correct` ở NB4 (0.5367 so với 0.6255) nhưng khi đo đạc trên tác vụ thực tế ở NB5 thì độ chính xác target hoàn toàn bằng nhau (0.9700). Điều này cho thấy chỉ số training loss và perplexity hoàn toàn có thể đánh lừa người làm mô hình, và việc dồn ép rank lên cực đại ($r=283$) trong các module attention không mang lại lợi thế vượt trội so với cấu hình phủ đều all-linear với rank nhỏ ($r=16$).

**2. Bạn mất nhiều thời gian nhất ở đâu? Nó có phải chỗ bạn dự đoán không?**
Phần mất nhiều thời gian nhất là bước sinh văn bản đánh giá (generation phase) qua 3 baseline ở NB2, NB4 và NB5, cùng với việc lưu checkpoint sáp nhập mô hình 8.5 GB ở NB6. Ban đầu tôi nghĩ thời gian chủ yếu sẽ nằm ở khâu tính gradient và cập nhật trọng số trong vòng lặp huấn luyện, nhưng thực tế việc giải mã tự hồi quy (autoregressive decode) trên tập eval và I/O ghi đĩa các trọng số mô hình lớn mới là nút thắt cổ chai chiếm phần lớn thời gian chờ đợi.

**3. Trước lab này bạn tin điều gì về fine-tuning mà giờ bạn không còn tin?**
Trước lab này, tôi từng tin rằng "cứ tăng rank $r$ lên càng cao thì mô hình LoRA sẽ học càng giỏi và cho kết quả càng vượt trội". Qua các thí nghiệm kiểm chứng công bằng với ngân sách tham số cố định, tôi nhận ra vị trí đặt adapter (toàn bộ text-linear bao gồm cả các tầng MLP) và tốc độ học (learning rate scale chuẩn 10x) mới là những đòn bẩy kiến trúc có sức nặng quyết định, chứ không phải bản thân giá trị rank.

**4. Bạn dùng AI assistant vào việc gì trong lab? Chỗ nào nó sai?**
Tôi sử dụng AI assistant để định vị cấu trúc repo, phân tích các chỉ số trong các file JSON log, giải thích cơ chế I/O khi lưu shard mô hình trên Colab, và hỗ trợ soạn thảo báo cáo phân tích khoa học. AI assistant ban đầu từng gợi ý dòng lệnh nén file có chứa tên thư mục gốc thừa khiến lệnh zip báo lỗi không tìm thấy đường dẫn trên Colab, nhưng sau đó đã được điều chỉnh lại đường dẫn chính xác theo thư mục làm việc hiện hành.

**5. Nếu ngày mai phải fine-tune cho một khách hàng thật, bước đầu tiên bạn làm là gì?**
Bước đầu tiên tôi sẽ làm là xây dựng một bộ đo lường độc lập và thiết lập baseline prompt tối ưu kèm cổng kiểm tra hồi quy (Regression Gate) trước khi chạm vào bất kỳ dòng mã huấn luyện nào. Việc đo mốc chuẩn trước và kiểm chứng loss mask bằng phương pháp giải mã ngược token sẽ giúp đảm bảo hệ thống không bị tự lừa mình bởi các chỉ số giả tạo, đồng thời sớm phát hiện nguy cơ thảm họa quên kiến thức đối với nghiệp vụ của khách hàng.
