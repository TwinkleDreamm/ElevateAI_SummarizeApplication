# Hướng dẫn sử dụng trang Settings - ElevateAI

## Tổng quan

Trang Settings cho phép bạn tùy chỉnh các thông số của ứng dụng ElevateAI để tối ưu hóa hiệu suất và trải nghiệm sử dụng.

## Cách truy cập

1. Mở ứng dụng ElevateAI
2. Chuyển đến trang "Settings" trong menu điều hướng
3. Hoặc nhấn nút "⚙️ Open Settings" trong sidebar

## Các nhóm cài đặt

### 🤖 Model Settings
- **Temperature**: Điều khiển độ ngẫu nhiên trong câu trả lời (0.0 - 2.0)
- **Max Tokens**: Số lượng token tối đa trong câu trả lời (100 - 8000)
- **Top P**: Điều khiển đa dạng của câu trả lời (0.0 - 1.0)
- **Frequency Penalty**: Giảm lặp lại từ ngữ (-2.0 - 2.0)
- **Presence Penalty**: Khuyến khích thảo luận về chủ đề mới (-2.0 - 2.0)
- **Model**: Chọn model AI để sử dụng

### 🔍 Search Settings
- **Similarity Threshold**: Ngưỡng tương đồng tối thiểu cho kết quả tìm kiếm (0.0 - 1.0)
- **Max Results**: Số lượng kết quả tìm kiếm tối đa (1 - 50)
- **Chunk Size**: Kích thước chunk khi xử lý tài liệu (100 - 2000)
- **Chunk Overlap**: Độ chồng lấp giữa các chunk (0 - 500)
- **Enable Web Search**: Tìm kiếm web khi kết quả local không đủ
- **Enable Function Calling**: Sử dụng function calling để phân tích nâng cao

### 🔊 Audio Settings
- **Enable TTS**: Tạo âm thanh cho tóm tắt
- **TTS Voice**: Giọng nói cho text-to-speech
- **Audio Sample Rate**: Tần số lấy mẫu âm thanh
- **Noise Reduction**: Độ mạnh giảm tiếng ồn (0.0 - 1.0)
- **Enable Vocal Separation**: Tách giọng nói khỏi nhạc nền

### 🧠 Memory Settings
- **Enable Memory System**: Ghi nhớ ngữ cảnh và thông tin cuộc trò chuyện
- **Max Memory Context**: Số lượng bản ghi bộ nhớ tối đa sử dụng cho ngữ cảnh (1 - 20)
- **Memory Consolidation Threshold**: Ngưỡng để củng cố bộ nhớ ngắn hạn thành dài hạn (5 - 50)
- **Store Conversations**: Lưu lịch sử cuộc trò chuyện
- **Memory Retention (days)**: Số ngày lưu trữ bộ nhớ (1 - 365)
- **Auto Cleanup Old Memories**: Tự động dọn dẹp bộ nhớ cũ

### 🎨 Interface Settings
- **Theme**: Chọn giao diện sáng hoặc tối
- **Language**: Ngôn ngữ giao diện
- **Auto Save Settings**: Tự động lưu cài đặt
- **Show Processing Time**: Hiển thị thời gian xử lý
- **Show Confidence Score**: Hiển thị điểm tin cậy
- **Enable Animations**: Bật hiệu ứng animation

### ⚙️ Advanced Settings
- **Max File Size (MB)**: Kích thước file tối đa được phép upload (10 - 1000 MB)
- **Enable Debug Mode**: Bật chế độ debug để xem thông tin chi tiết
- **Enable Caching**: Bật cache để tăng tốc độ
- **Log Level**: Mức độ log (DEBUG, INFO, WARNING, ERROR)
- **Enable Metrics Collection**: Thu thập metrics để cải thiện hiệu suất
- **Enable Auto Backup**: Tự động sao lưu dữ liệu

## Các thao tác chính

### 💾 Save & Apply
- Lưu tất cả cài đặt và áp dụng ngay lập tức
- Validate cài đặt trước khi lưu
- Hiển thị thông báo thành công hoặc lỗi

### 🔄 Reset to Defaults
- Khôi phục tất cả cài đặt về giá trị mặc định
- Xóa cài đặt trong session state và file cấu hình

### 📁 Import/Export Settings
- **Import**: Tải file cài đặt JSON đã xuất trước đó
- **Export**: Tải xuống cài đặt hiện tại dưới dạng file JSON

### 📊 Settings Summary
- Xem tóm tắt cài đặt hiện tại
- Hiển thị theo nhóm để dễ theo dõi

## Thứ tự ưu tiên cài đặt

1. **File Settings** (vĩnh viễn) - Cao nhất
2. **Session Settings** (tạm thời) - Trung bình
3. **Default Settings** (mặc định) - Thấp nhất

## Lưu ý quan trọng

- Cài đặt được lưu vào file `config/user_settings.json`
- Một số thay đổi có thể cần khởi động lại ứng dụng để có hiệu lực hoàn toàn
- Cài đặt được áp dụng cho toàn bộ ứng dụng
- Luôn backup cài đặt trước khi thay đổi lớn

## Xử lý lỗi

### Lỗi validation
- Kiểm tra giá trị trong khoảng cho phép
- Đảm bảo định dạng dữ liệu đúng
- Xem thông báo lỗi chi tiết

### Lỗi lưu file
- Kiểm tra quyền ghi vào thư mục config
- Đảm bảo đủ dung lượng ổ đĩa
- Thử lưu vào session state nếu không thể lưu file

## Ví dụ cài đặt tối ưu

### Cho tóm tắt nhanh:
- Temperature: 0.3
- Max Tokens: 1000
- Similarity Threshold: 0.3
- Enable Function Calling: ✅

### Cho phân tích chi tiết:
- Temperature: 0.7
- Max Tokens: 3000
- Similarity Threshold: 0.2
- Enable Function Calling: ✅
- Enable Web Search: ✅

### Cho sáng tạo:
- Temperature: 1.0
- Max Tokens: 2000
- Frequency Penalty: 0.5
- Presence Penalty: 0.3
