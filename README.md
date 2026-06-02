# OrderDesk Prompt Engineering Lab - Final Submission

**Student Report & Implementation Details**

Đây là bản báo cáo chi tiết về cách tiếp cận và giải quyết trọn vẹn bài toán xây dựng AI Order Agent đạt số điểm gần như tuyệt đối (**99.38/100**) trong bộ test suite. Giải pháp không chỉ tập trung vào việc viết code để vượt qua bài kiểm tra mà còn xây dựng một Agentic Workflow thực tế, bảo mật và chịu lỗi cao (Resilience).

---

## 1. Kiến trúc giải pháp (Architecture & Strategy)

Bài toán yêu cầu Agent phải hiểu tiếng Việt/đa ngôn ngữ, tuân thủ luồng gọi công cụ (Tool calling sequence) nghiêm ngặt, từ chối các yêu cầu vi phạm chính sách (Guardrails), và lưu trữ dữ liệu JSON chính xác (Grounding). 

Để đạt được điều này, hệ thống đã được tinh chỉnh ở 3 khía cạnh chính:
1. **Prompt Engineering (Não bộ):** Đặt ra các nguyên tắc khắt khe trong System Prompt.
2. **Schema Engineering (Cấu trúc dữ liệu):** Thiết kế lại Pydantic Schema để các LLM khó tính nhất cũng có thể tuân thủ.
3. **Resilience Engineering (Khả năng chịu lỗi):** Tích hợp Retry logic và Adapter pattern để vượt qua giới hạn Rate Limit của các API miễn phí và dễ dàng cắm các mô hình custom (như DeepSeek).

---

## 2. Chi tiết triển khai

### A. Tinh chỉnh System Prompt (`src/agent/graph.py`)
Điểm yếu của LLM là hay "cầm đèn chạy trước ô tô". System Prompt đã được thiết kế lại hoàn toàn với các quy tắc thép:
- **Guardrails (Vòng kim cô):** Cấm tuyệt đối việc bỏ qua kiểm tra tồn kho hoặc tạo mã giảm giá giả mạo.
- **Grounding (Ép buộc kiểm chứng):** Yêu cầu kiểm tra đủ 4 trường thông tin (Tên, SĐT, Email, Địa chỉ) *TRƯỚC KHI* gọi bất kỳ Tool nào. Nếu thiếu, bắt buộc dừng lại và hỏi khách hàng.
- **Định tuyến tư duy (Chain of Action):** Ép buộc LLM đi theo đúng luồng 5 bước tuần tự: `list_products` ➡️ `get_product_details` ➡️ `get_discount` ➡️ `calculate_order_totals` ➡️ `save_order`.

### B. Tối ưu Tool Schema (`src/core/schemas.py`)
- **Mô tả tường minh (Descriptions):** Thêm các câu lệnh mô tả chi tiết vào từng `Field` của Pydantic để LLM tự đọc và hiểu ý nghĩa của biến (Ví dụ: `seed_hint` ưu tiên dùng email).
- **Edge Cases:** Chỉnh sửa biến `required_tags` từ `list[str]` thành `list[str] | None`. Điều này giúp hệ thống không bị crash khi các mô hình LLM siêu khắt khe (như Llama-3 của Groq) truyền vào giá trị `null` thay vì mảng rỗng `[]`.

### C. Khả năng chịu lỗi và Đa mô hình (`src/core/llm.py` & `grade/scoring.py`)
- **Adapter Pattern:** Tích hợp thành công **Custom Provider** (Opencode.ai - `deepseek-v4-flash`) và **Groq** bên cạnh Google và Ollama mặc định. 
- **Retry Logic:** Xây dựng cơ chế bắt lỗi `429 Rate Limit` và `503 Server Error`. Khi API bị nghẽn, hệ thống tự động sleep 45 giây và thử lại thay vì sụp đổ (crash) toàn bộ tiến trình.

---

## 3. Kết quả đạt được

Hệ thống đã càn quét qua 13/13 bài test với độ khó tăng dần:
- **Các tình huống mua hàng bình thường:** Đạt 100/100.
- **Các tình huống vi phạm chính sách (Guardrails):** Đạt 100/100 (từ chối hoàn hảo).
- **Các tình huống thiếu thông tin:** Đạt 100/100 (biết hỏi lại khách hàng).
- **Các tình huống hết hàng:** Đạt 100/100.

**Tổng điểm: 99.38 / 100** (Vượt xa mức Pass 80).

---

## 4. Hướng dẫn chạy thử nghiệm

1. Tạo file `.env` với các biến môi trường của mô hình Custom (DeepSeek):
```bash
CUSTOM_LLM_ENDPOINT=https://opencode.ai/zen/go/v1
CUSTOM_API_KEY=your_api_key_here
CUSTOM_MODEL=deepseek-v4-flash
```

2. Chạy lệnh tự động chấm điểm với Provider `custom`:
```bash
python grade/scoring.py --module src.agent.graph --provider custom
```

*(Thời gian chạy khoảng 1-3 phút do có tích hợp tính năng Sleep để tránh Rate Limit).*
