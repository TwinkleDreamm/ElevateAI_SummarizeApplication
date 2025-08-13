# Hướng dẫn sử dụng Layout 3 Tab - Notebook

## Tổng quan

Trang Notebook đã được cải tiến với layout 3 tab mới, cung cấp trải nghiệm làm việc hiệu quả hơn với khả năng quản lý sources, chat và lưu trữ notes.

## Layout 3 Tab

### 📚 Sources Tab
- **Vị trí**: Tab đầu tiên
- **Chức năng**: Quản lý các nguồn tài liệu
- **Tính năng**:
  - Hiển thị danh sách sources
  - Nút ❌ để xóa source
  - Upload files và thêm links
  - Tận dụng toàn bộ không gian màn hình

### 📓 Notebook Tab
- **Vị trí**: Tab thứ hai
- **Chức năng**: Tương tác chính với notebook
- **Tính năng**:
  - Overview của notebook
  - Example questions
  - Chat interface
  - Nút "💾 Save Note" để lưu câu trả lời

### 🎨 Studio Tab
- **Vị trí**: Tab thứ ba
- **Chức năng**: Quản lý notes đã lưu
- **Tính năng**:
  - Hiển thị notes đã lưu
  - Nút "📚 Add to Source" để chuyển note thành source
  - Nút "🗑️ Delete" để xóa note
  - Tận dụng toàn bộ không gian màn hình

## Cách sử dụng

### 1. Quản lý Sources
- **Chuyển tab**: Click vào tab "📚 Sources"
- **Xem sources**: Danh sách sources hiển thị đầy đủ
- **Xóa source**: Click nút ❌ bên cạnh source
- **Thêm source**: Upload files hoặc nhập URL

### 2. Chat và lưu Notes
- **Chuyển tab**: Click vào tab "📓 Notebook"
- **Đặt câu hỏi**: Nhập câu hỏi vào text area
- **Nhận câu trả lời**: Click "Ask" để nhận câu trả lời
- **Lưu note**: Click "💾 Save Note" để lưu câu trả lời vào Studio
- **Xem sources**: Mở expander "Sources" để xem nguồn tham khảo

### 3. Quản lý Studio
- **Chuyển tab**: Click vào tab "🎨 Studio"
- **Xem notes**: Danh sách notes hiển thị đầy đủ
- **Chuyển thành source**: Click "📚 Add to Source" để chuyển note thành source mới
- **Xóa note**: Click "🗑️ Delete" để xóa note

## Tính năng Tab Navigation

### Cách hoạt động
- Mỗi tab có chức năng riêng biệt
- Chuyển đổi dễ dàng giữa các tab
- Tận dụng toàn bộ không gian màn hình cho mỗi tab

### Lợi ích
- Tập trung vào công việc chính
- Tối ưu không gian màn hình
- Dễ dàng chuyển đổi giữa các chế độ làm việc

## Workflow đề xuất

### 1. Khám phá nội dung
```
Sources Tab → Notebook Tab → Studio Tab
```
- Tập trung vào việc xem sources và chat

### 2. Tạo notes
```
Notebook Tab → Studio Tab → Sources Tab
```
- Tập trung vào việc chat và lưu notes

### 3. Tổ chức lại
```
Sources Tab → Studio Tab → Notebook Tab
```
- Tập trung vào việc quản lý sources và notes

## Phím tắt và Tips

### Phím tắt
- **Ctrl+Enter**: Gửi câu hỏi trong chat
- **Ctrl+S**: Lưu note hiện tại (nếu có)

### Tips hiệu quả
1. **Sử dụng Example Questions**: Click vào câu hỏi ví dụ để bắt đầu nhanh
2. **Lưu notes thường xuyên**: Lưu những câu trả lời quan trọng
3. **Tổ chức sources**: Xóa sources không cần thiết
4. **Chuyển notes thành sources**: Khi note trở nên quan trọng, chuyển thành source

## Xử lý lỗi

### Lỗi thường gặp
1. **Tab không chuyển được**: Refresh trang
2. **Note không lưu được**: Kiểm tra session state
3. **Source không xóa được**: Kiểm tra quyền truy cập

### Khắc phục
- Refresh trang để reset session state
- Kiểm tra console để xem lỗi
- Đảm bảo có quyền ghi vào database

## Tùy chỉnh giao diện

### CSS Customization
File CSS được lưu tại: `src/interface/styles/notebook_layout.css`

### Các class chính
- `.tab-sources`: Styling cho tab Sources
- `.tab-notebook`: Styling cho tab Notebook
- `.tab-studio`: Styling cho tab Studio
- `.stTabs`: Styling cho tab navigation

## Responsive Design

### Desktop (>1200px)
- Layout 3 tab đầy đủ
- Tất cả tính năng có sẵn

### Tablet (768px - 1200px)
- Layout thích ứng
- Tabs có thể scroll ngang

### Mobile (<768px)
- Layout 1 tab tại một thời điểm
- Tabs có thể scroll ngang

## Tương lai

### Tính năng sắp tới
- Drag & drop để sắp xếp sources
- Export notes ra file
- Share notes với người khác
- Advanced filtering cho sources
- Batch operations cho notes
