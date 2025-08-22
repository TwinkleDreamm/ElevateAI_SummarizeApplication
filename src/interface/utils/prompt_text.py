from __future__ import annotations

"""
Centralized prompt and text constants for the interface layer.
Keep VN/EN variants close together with a short comment header.

Usage:
- Use t(key, lang) to fetch localized UI text.
- Define UI_TEXTS with keys and per-language variants.
"""

# ===== Language detection character sets =====
VI_CHAR_SET: str = (
    "àáạảãâầấậẩẫăằắặẳẵ"
    "èéẹẻẽêềếệểễ"
    "ìíịỉĩ"
    "òóọỏõôồốộổỗơờớợởỡ"
    "ùúụủũưừứựửữ"
    "ỳýỵỷỹđ"
)


# ===== Mindmap generation prompts =====
# Mindmap summary instruction (VI): concise, event-focused
MINDMAP_SUMMARY_VI: str = (
    "Tạo bản tóm tắt NGẮN GỌN (≤ 500 từ) để dựng mindmap. "
    "Ưu tiên dữ kiện có cấu trúc: mốc thời gian (ngày/tháng/năm), sự kiện, địa điểm, nhân sự/đơn vị, trạng thái/kết quả. "
    "Dùng tiêu đề ngắn cho mục/tiểu mục; đưa ngày tháng và địa điểm trực tiếp vào tiêu đề khi có, ví dụ: '20/05 – Thông báo chính thức (Hà Nội)'. "
    "Không lặp lại, không diễn giải dài."
)

# Mindmap summary instruction (EN)
MINDMAP_SUMMARY_EN: str = (
    "Create a CONCISE summary (≤ 500 words) for mindmap building. "
    "Prioritize structured facts: timeline (dates), key events, locations, people/units, statuses/outcomes. "
    "Use short headings; include date/location directly in headings when available, e.g., '20/05 – Official announcement (Hanoi)'. "
    "Avoid repetition and lengthy prose."
)

# Mindmap summary instruction (ZH)
MINDMAP_SUMMARY_ZH: str = (
    "生成用于思维导图的精炼摘要（≤500字）。"
    "优先保留结构化要点：时间线（日期）、关键事件、地点、相关人物/单位、状态/结果。"
    "使用简短标题；如有日期/地点请直接放入标题，例如：‘05/20 – 官方通告（河内）’。"
    "避免重复和长篇叙述。"
)

# Mindmap summary instruction (JA)
MINDMAP_SUMMARY_JA: str = (
    "マインドマップ作成のため、簡潔な要約（500語以内）を作成してください。"
    "時系列（日付）、主要イベント、場所、人物/組織、状態/結果などの構造化情報を優先。"
    "見出しは短くし、可能な場合は日付/場所を見出しに直接含める（例：‘05/20 – 公式告知（ハノイ）’）。"
    "重複や冗長な説明は避けてください。"
)

# Mindmap summary instruction (KO)
MINDMAP_SUMMARY_KO: str = (
    "마인드맵 작성을 위한 간결한 요약(≤ 500단어)을 생성하세요."
    "타임라인(날짜), 핵심 사건, 위치, 인물/조직, 상태/결과 등 구조화된 사실을 우선합니다."
    "짧은 제목을 사용하고, 가능하면 날짜/위치를 제목에 직접 포함하세요. 예: ‘05/20 – 공식 발표(하노이)’."
    "중복과 장황한 서술을 피하세요."
)

# Mindmap outline extraction (VI) – JSON schema instruction
MINDMAP_JSON_INSTRUCTION_VI: str = (
    "Trích xuất MINDMAP dạng JSON từ nội dung sau. Trả về DUY NHẤT JSON hợp lệ theo schema: "
    "{\"title\": string, \"nodes\": [{\"label\": string, \"children\": [ ... ]}]}\. "
    "YÊU CẦU: 1) Label ngắn gọn ≤80 ký tự; 2) Nếu có ngày/địa điểm/trạng thái chèn trực tiếp vào label; "
    "3) Tối đa 3 cấp; 4) Tổng số nút ≤100; 5) Chỉ JSON thuần."
)

# Mindmap outline extraction (EN)
MINDMAP_JSON_INSTRUCTION_EN: str = (
    "Extract a MINDMAP in JSON only. Return JSON with schema: "
    "{\"title\": string, \"nodes\": [{\"label\": string, \"children\": [ ... ]}]}\. "
    "REQUIREMENTS: 1) Short labels ≤80 chars; 2) If date/location/status is present, embed directly in label; "
    "3) Max depth 3; 4) Max total nodes 100; 5) JSON only, no explanations."
)

# Mindmap outline extraction (ZH)
MINDMAP_JSON_INSTRUCTION_ZH: str = (
    "仅以JSON形式提取思维导图。返回符合以下模式的JSON："
    "{\"title\": string, \"nodes\": [{\"label\": string, \"children\": [ ... ]}]}。"
    "要求：1）标签简短≤80字符；2）若含日期/地点/状态，请直接写入标签；3）最大深度3；4）节点总数≤100；5）仅返回纯JSON。"
)

# Mindmap outline extraction (JA)
MINDMAP_JSON_INSTRUCTION_JA: str = (
    "JSONのみでマインドマップを抽出してください。以下のスキーマに従うJSONを返します："
    "{\"title\": string, \"nodes\": [{\"label\": string, \"children\": [ ... ]}]}。"
    "要件：1）ラベルは80文字以内で簡潔に；2）日付/場所/状態があればラベルに直接含める；3）最大深さ3；4）総ノード数100以内；5）説明なしでJSONのみ。"
)

# Mindmap outline extraction (KO)
MINDMAP_JSON_INSTRUCTION_KO: str = (
    "JSON 형식으로만 마인드맵을 추출하세요. 다음 스키마의 JSON을 반환합니다: "
    "{\"title\": string, \"nodes\": [{\"label\": string, \"children\": [ ... ]}]}. "
    "요구 사항: 1) 라벨은 80자 이내로 간결하게; 2) 날짜/위치/상태가 있으면 라벨에 직접 포함; 3) 최대 깊이 3; 4) 전체 노드 ≤100; 5) JSON만, 설명 금지."
)


# ===== No-results response prompts =====
# Provide helpful guidance when no relevant notebook chunks are found
NO_RESULTS_SYSTEM_VI: str = (
    "Bạn là trợ lý AI hữu ích. Người dùng vừa tìm kiếm nhưng không tìm thấy kết quả trong notebook hiện tại. "
    "Hãy tạo phản hồi ngắn gọn: 1) Thông báo không có nội dung liên quan; 2) Gợi ý notebook khác; "
    "3) Hướng dẫn người dùng có thể tìm trong notebook khác hoặc tạo notebook mới thủ công. "
    "KHÔNG hứa hẹn khả năng tự động tạo notebook."
)

NO_RESULTS_SYSTEM_EN: str = (
    "You are a helpful AI assistant. The user searched but found no relevant content in this notebook. "
    "Provide a concise response: 1) Acknowledge no relevant content; 2) Suggest other notebooks; "
    "3) Guide the user to search elsewhere or manually create a new notebook. "
    "DO NOT promise automatic notebook creation."
)

NO_RESULTS_SYSTEM_ZH: str = (
    "你是一个有用的AI助手。用户在当前笔记本中没有找到相关内容。"
    "请给出简洁回复：1）说明无相关内容；2）建议查看其他笔记本；"
    "3）引导用户在其他笔记本中搜索或手动创建新笔记本。"
    "不要承诺自动创建笔记本。"
)

NO_RESULTS_SYSTEM_JA: str = (
    "あなたは有用なAIアシスタントです。現在のノートブックでは関連する内容が見つかりませんでした。"
    "簡潔に回答してください：1）関連内容がないことを伝える；2）他のノートブックを提案する；"
    "3）他のノートブックで検索する、または新しいノートブックを手動で作成するよう案内する。"
    "ノートブックを自動作成できると約束しないでください。"
)

NO_RESULTS_SYSTEM_KO: str = (
    "당신은 유용한 AI 도우미입니다. 현재 노트북에서 관련된 내용을 찾을 수 없습니다."
    "간결한 응답을 제공하세요: 1) 관련 내용이 없음을 알림; 2) 다른 노트북을 제안;"
    "3) 다른 노트북에서 검색하거나 새 노트북을 수동으로 만들도록 안내."
    "노트북 자동 생성 가능하다고 약속하지 마세요."
)

# ===== UI Text Registry (vi/en/zh/ja/ko) =====
UI_TEXTS = {
    # Generic
    "loading": {"vi": "Đang tải...", "en": "Loading...", "zh": "加载中...", "ja": "読み込み中...", "ko": "로딩 중..."},
    "search": {"vi": "Tìm kiếm", "en": "Search", "zh": "搜索", "ja": "検索", "ko": "검색"},
    "create": {"vi": "Tạo", "en": "Create", "zh": "创建", "ja": "作成", "ko": "생성"},
    "ask": {"vi": "Hỏi", "en": "Ask", "zh": "提问", "ja": "質問", "ko": "질문"},
    "please_enter_question": {"vi": "Vui lòng nhập câu hỏi", "en": "Please enter a question", "zh": "请输入问题", "ja": "質問を入力してください", "ko": "질문을 입력하세요"},
    "searching_answering": {"vi": "Đang tìm và trả lời…", "en": "Searching and answering…", "zh": "正在检索并回答…", "ja": "検索と回答中…", "ko": "검색 및 응답 중…"},
    # Notebooks page
    "page_notebooks_title": {"vi": "📓 ElevateAI Notebooks", "en": "📓 ElevateAI Notebooks", "zh": "📓 ElevateAI 笔记本", "ja": "📓 ElevateAI ノートブック", "ko": "📓 ElevateAI 노트북"},
    "page_notebooks_subtitle": {"vi": "Tạo, tổ chức và trò chuyện với notebooks của bạn.", "en": "Create, organize and chat with your notebooks.", "zh": "创建、管理并与您的笔记本对话。", "ja": "ノートブックを作成・整理し、対話します。", "ko": "노트북을 생성하고 정리하며 대화하세요."},
    "create_new_notebook": {"vi": "Tạo Notebook Mới", "en": "Create New Notebook", "zh": "创建新笔记本", "ja": "新しいノートブックを作成", "ko": "새 노트북 만들기"},
    "your_notebooks": {"vi": "Notebooks của bạn", "en": "Your Notebooks", "zh": "你的笔记本", "ja": "あなたのノートブック", "ko": "내 노트북"},
    "loading_notebooks": {"vi": "Đang tải notebooks...", "en": "Loading notebooks...", "zh": "正在加载笔记本...", "ja": "ノートブックを読み込み中...", "ko": "노트북 불러오는 중..."},
    "no_notebooks_yet": {"vi": "Chưa có notebook nào. Hãy tạo notebook đầu tiên!", "en": "No notebooks yet. Create your first notebook to get started!", "zh": "还没有笔记本。创建您的第一个笔记本开始吧！", "ja": "まだノートブックがありません。まずは作成しましょう！", "ko": "아직 노트북이 없습니다. 첫 노트북을 만들어 시작하세요!"},
    "refresh_notebooks": {"vi": "🔄 Làm mới Notebooks", "en": "🔄 Refresh Notebooks", "zh": "🔄 刷新笔记本", "ja": "🔄 ノートブックを更新", "ko": "🔄 노트북 새로고침"},
    "refresh_notebooks_help": {"vi": "Xóa cache và tải lại notebooks", "en": "Clear cache and reload notebooks", "zh": "清除缓存并重新加载笔记本", "ja": "キャッシュをクリアして再読み込み", "ko": "캐시를 지우고 다시 로드"},
    "filter_title": {"vi": "🔍 Lọc Notebooks", "en": "🔍 Filter Notebooks", "zh": "🔍 筛选笔记本", "ja": "🔍 ノートブックを絞り込み", "ko": "🔍 노트북 필터"},
    "sort_by": {"vi": "Sắp xếp", "en": "Sort by", "zh": "排序", "ja": "並び替え", "ko": "정렬"},
    "stable_sort": {"vi": "Sắp xếp ổn định", "en": "Stable sorting", "zh": "稳定排序", "ja": "安定した並び替え", "ko": "안정 정렬"},
    "dynamic_sort": {"vi": "Sắp xếp động", "en": "Dynamic sorting", "zh": "动态排序", "ja": "動的な並び替え", "ko": "동적 정렬"},
    "alphabetical_sort": {"vi": "Sắp xếp theo tên", "en": "Alphabetical sorting", "zh": "按名称排序", "ja": "名前順", "ko": "이름 순 정렬"},
    "ask_question": {"vi": "Đặt câu hỏi", "en": "Ask a Question", "zh": "提问", "ja": "質問する", "ko": "질문하기"},
    "chat_history": {"vi": "Lịch sử Chat", "en": "Chat History", "zh": "聊天记录", "ja": "チャット履歴", "ko": "채팅 기록"},
    "no_chat_history": {"vi": "Chưa có chat nào.", "en": "No chat history yet.", "zh": "暂无聊天记录。", "ja": "まだチャット履歴がありません。", "ko": "채팅 기록이 없습니다."},
    "include_sources": {"vi": "Hiển thị nguồn", "en": "Include sources", "zh": "包含来源", "ja": "出典を含める", "ko": "출처 포함"},
    "your_question_placeholder": {"vi": "Hỏi bất cứ điều gì về các nguồn trong notebook…", "en": "Ask anything about the sources in this notebook…", "zh": "就此笔记本的来源提出任何问题…", "ja": "このノートブックの情報源について何でも質問…", "ko": "이 노트북의 소스에 대해 무엇이든 질문하세요…"},
    "answer_generated": {"vi": "✅ Đã tạo câu trả lời!", "en": "✅ Answer generated!", "zh": "✅ 已生成答案！", "ja": "✅ 回答を生成しました！", "ko": "✅ 답변이 생성되었습니다!"},
    "helpful_response_generated": {"vi": "💡 Đã tạo phản hồi hữu ích!", "en": "💡 Helpful response generated!", "zh": "💡 已生成有用的回应！", "ja": "💡 役立つ応答を生成しました！", "ko": "💡 유용한 응답이 생성되었습니다!"},
    "fast_mode": {"vi": "Chế độ nhanh", "en": "Fast mode", "zh": "快速模式", "ja": "高速モード", "ko": "빠른 모드"},
    "cot": {"vi": "Chuỗi suy luận", "en": "Chain-of-thought", "zh": "思维链", "ja": "思考の連鎖", "ko": "사고의 연쇄"},
    # Studio
    "studio_title": {"vi": "Studio", "en": "Studio", "zh": "工作室", "ja": "スタジオ", "ko": "스튜디오"},
    "btn_docx": {"vi": "📄 Tổng quan bằng file DOCX", "en": "📄 DOCX Overview", "zh": "📄 DOCX 概览", "ja": "📄 DOCX 概要", "ko": "📄 DOCX 개요"},
    "btn_audio": {"vi": "🔊 Tổng quan bằng âm thanh", "en": "🔊 Audio Overview", "zh": "🔊 音频概览", "ja": "🔊 音声概要", "ko": "🔊 오디오 개요"},
    "btn_mindmap": {"vi": "🧠 Bản đồ tư duy", "en": "🧠 Mindmap", "zh": "🧠 思维导图", "ja": "🧠 マインドマップ", "ko": "🧠 마인드맵"},
    "open_mindmap": {"vi": "Mở mindmap", "en": "Open mindmap", "zh": "打开思维导图", "ja": "マインドマップを開く", "ko": "마인드맵 열기"},
    "generating_docx": {"vi": "⏳ Đang tạo DOCX...", "en": "⏳ Generating DOCX...", "zh": "⏳ 正在生成 DOCX...", "ja": "⏳ DOCX を生成中...", "ko": "⏳ DOCX 생성 중..."},
    "generating_audio": {"vi": "⏳ Đang tạo audio...", "en": "⏳ Generating audio...", "zh": "⏳ 正在生成音频...", "ja": "⏳ 音声を生成中...", "ko": "⏳ 오디오 생성 중..."},
    "download_docx_help": {"vi": "Tải báo cáo DOCX", "en": "Download DOCX report", "zh": "下载 DOCX 报告", "ja": "DOCX レポートをダウンロード", "ko": "DOCX 보고서 다운로드"},
    "download_audio_help": {"vi": "Tải audio", "en": "Download audio", "zh": "下载音频", "ja": "音声をダウンロード", "ko": "오디오 다운로드"},
    "download_mindmap_help": {"vi": "Tải mindmap", "en": "Download mindmap", "zh": "下载思维导图", "ja": "マインドマップをダウンロード", "ko": "마인드맵 다운로드"},
    # Sources
    "sources": {"vi": "📚 Nguồn", "en": "📚 Sources", "zh": "📚 来源", "ja": "📚 ソース", "ko": "📚 소스"},
    "no_sources": {"vi": "Chưa có nguồn.", "en": "No sources yet.", "zh": "暂无来源。", "ja": "ソースはまだありません。", "ko": "소스가 없습니다."},
    "delete_source": {"vi": "Xóa nguồn", "en": "Delete source", "zh": "删除来源", "ja": "ソースを削除", "ko": "소스 삭제"},
    "search_results": {"vi": "Kết quả tìm kiếm", "en": "Search results", "zh": "搜索结果", "ja": "検索結果", "ko": "검색 결과"},
    "added_at": {"vi": "Thêm lúc", "en": "Added at", "zh": "添加时间", "ja": "追加日時", "ko": "추가 시각"},
    "content_type": {"vi": "Loại nội dung", "en": "Content type", "zh": "内容类型", "ja": "コンテンツタイプ", "ko": "콘텐츠 유형"},
    "chunk_count": {"vi": "Số đoạn", "en": "Chunk count", "zh": "分段数量", "ja": "チャンク数", "ko": "청크 수"},
    # Add sources section
    "add_sources": {"vi": "Thêm nguồn", "en": "Add sources", "zh": "添加来源", "ja": "ソースを追加", "ko": "소스 추가"},
    "upload_files": {"vi": "Tải tệp lên", "en": "Upload files", "zh": "上传文件", "ja": "ファイルをアップロード", "ko": "파일 업로드"},
    "or_add_link": {"vi": "Hoặc thêm liên kết", "en": "Or add a link", "zh": "或添加链接", "ja": "またはリンクを追加", "ko": "또는 링크 추가"},
    "search_internet": {"vi": "Tìm trên Internet", "en": "Search Internet", "zh": "搜索互联网", "ja": "インターネット検索", "ko": "인터넷 검색"},
    "enter_keywords_placeholder": {"vi": "Nhập từ khóa hoặc chủ đề (tối đa 100 ký tự)", "en": "Enter keywords or topic (max 100 chars)", "zh": "输入关键词或主题（最多100字符）", "ja": "キーワードまたはトピックを入力（最大100文字）", "ko": "키워드 또는 주제를 입력하세요(최대 100자)"},
    "select_all": {"vi": "Chọn tất cả", "en": "Select all", "zh": "全选", "ja": "すべて選択", "ko": "모두 선택"},
    "add_to_notebook": {"vi": "Thêm vào notebook", "en": "Add to notebook", "zh": "添加到笔记本", "ja": "ノートブックに追加", "ko": "노트북에 추가"},
    "no_results_for_query": {"vi": "Không có kết quả cho truy vấn của bạn.", "en": "No results found for your query.", "zh": "未找到相关结果。", "ja": "該当する結果は見つかりませんでした。", "ko": "검색 결과가 없습니다."},
    "web_search_failed": {"vi": "Tìm kiếm web thất bại", "en": "Web search failed", "zh": "网页搜索失败", "ja": "ウェブ検索に失敗しました", "ko": "웹 검색 실패"},
    "source_exists": {"vi": "Nguồn đã tồn tại trong notebook!", "en": "Source already exists in this notebook!", "zh": "该来源已存在于此笔记本！", "ja": "このノートブックには既にソースがあります！", "ko": "이 노트북에 이미 소스가 있습니다!"},
    "some_sources_exist": {"vi": "Một số nguồn đã tồn tại trong notebook!", "en": "Some sources already exist in this notebook!", "zh": "部分来源已存在于此笔记本！", "ja": "一部のソースは既に存在します！", "ko": "일부 소스는 이미 존재합니다!"},
    "some_sources_skipped": {"vi": "Một số nguồn đã tồn tại và đã được bỏ qua: ", "en": "Some sources already exist and were skipped: ", "zh": "部分已有来源，已跳过：", "ja": "既存のソースのためスキップ：", "ko": "일부 소스가 이미 있어 건너뜀: "},
    "added_chunks": {"vi": "Đã thêm {n} đoạn.", "en": "Added {n} chunks.", "zh": "已添加 {n} 个分段。", "ja": "{n} 個のチャンクを追加しました。", "ko": "{n}개의 청크를 추가했습니다."},
    # Overview/Examples/Notes labels
    "overview_examples": {"vi": "📘 Tổng quan & Câu hỏi ví dụ", "en": "📘 Overview & Example questions", "zh": "📘 概览与示例问题", "ja": "📘 概要と例示質問", "ko": "📘 개요 및 예시 질문"},
    "overview": {"vi": "Tổng quan", "en": "Overview", "zh": "概览", "ja": "概要", "ko": "개요"},
    "examples": {"vi": "Câu hỏi ví dụ", "en": "Example questions", "zh": "示例问题", "ja": "例示質問", "ko": "예시 질문"},
    "cached_overview_examples": {"vi": "📝 Overview và Examples đã được lưu cache", "en": "📝 Overview and Examples are cached", "zh": "📝 概览与示例已缓存", "ja": "📝 概要と例がキャッシュされています", "ko": "📝 개요와 예시가 캐시되었습니다"},
    "creating_overview": {"vi": "Đang tạo overview...", "en": "Creating overview...", "zh": "正在生成概览...", "ja": "概要を生成中...", "ko": "개요 생성 중..."},
    "creating_examples": {"vi": "Đang tạo câu hỏi ví dụ...", "en": "Generating example questions...", "zh": "正在生成示例问题...", "ja": "例示質問を生成中...", "ko": "예시 질문 생성 중..."},
    # Buttons/labels in chat items
    "save_note": {"vi": "💾 Lưu ghi chú", "en": "💾 Save Note", "zh": "💾 保存笔记", "ja": "💾 メモを保存", "ko": "💾 노트 저장"},
    "saved": {"vi": "✅ Đã lưu", "en": "✅ Saved", "zh": "✅ 已保存", "ja": "✅ 保存しました", "ko": "✅ 저장됨"},
    "speak": {"vi": "🔊 Nghe", "en": "🔊 Speak", "zh": "🔊 朗读", "ja": "🔊 読み上げ", "ko": "🔊 듣기"},
    "listen_answer": {"vi": "Nghe câu trả lời này", "en": "Listen to this answer", "zh": "收听此答案", "ja": "この回答を聞く", "ko": "이 답변 듣기"},
    "audio_generated": {"vi": "🎵 Đã tạo âm thanh.", "en": "🎵 Audio generated.", "zh": "🎵 已生成音频。", "ja": "🎵 音声を生成しました。", "ko": "🎵 오디오가 생성되었습니다."},
    "audio_failed": {"vi": "❌ Tạo âm thanh thất bại", "en": "❌ Failed to generate audio", "zh": "❌ 生成音频失败", "ja": "❌ 音声の生成に失敗", "ko": "❌ 오디오 생성 실패"},
    "tts_not_available": {"vi": "❌ Dịch vụ TTS không khả dụng", "en": "❌ TTS service not available", "zh": "❌ TTS 服务不可用", "ja": "❌ TTS サービスは利用できません", "ko": "❌ TTS 서비스를 사용할 수 없습니다"},
    "audio_generating": {"vi": "🎵 Đang tạo âm thanh...", "en": "🎵 Generating audio...", "zh": "🎵 正在生成音频...", "ja": "🎵 音声を生成中...", "ko": "🎵 오디오 생성 중..."},
    "answer_truncated": {"vi": "⚠️ Câu trả lời đã bị cắt cho TTS (tối đa {n} ký tự)", "en": "⚠️ Answer was truncated for TTS (max {n} characters)", "zh": "⚠️ 为适配TTS已截断答案（最多 {n} 字符）", "ja": "⚠️ TTSのため回答を切り詰めました（最大 {n} 文字）", "ko": "⚠️ TTS를 위해 답변이 잘렸습니다(최대 {n}자)"},
    "error_generating_speech": {"vi": "❌ Lỗi tạo âm thanh", "en": "❌ Error generating speech", "zh": "❌ 生成语音时出错", "ja": "❌ 音声生成エラー", "ko": "❌ 음성 생성 오류"},
    # Notes
    "notes": {"vi": "📝 Ghi chú", "en": "📝 Notes", "zh": "📝 笔记", "ja": "📝 メモ", "ko": "📝 노트"},
    "no_saved_notes": {"vi": "Chưa có ghi chú nào.", "en": "No saved notes yet.", "zh": "暂无笔记。", "ja": "保存されたメモはまだありません。", "ko": "저장된 노트가 없습니다."},
    "add_to_source": {"vi": "📚 Thêm vào Nguồn", "en": "📚 Add to Source", "zh": "📚 添加到来源", "ja": "📚 ソースに追加", "ko": "📚 소스에 추가"},
    "added": {"vi": "✅ Đã thêm", "en": "✅ Added", "zh": "✅ 已添加", "ja": "✅ 追加しました", "ko": "✅ 추가됨"},
    "note_added_to_sources": {"vi": "✅ Đã thêm ghi chú vào Nguồn!", "en": "✅ Note added to sources!", "zh": "✅ 已将笔记添加到来源！", "ja": "✅ メモをソースに追加しました！", "ko": "✅ 노트를 소스에 추가했습니다!"},
    "delete": {"vi": "🗑️ Xóa", "en": "🗑️ Delete", "zh": "🗑️ 删除", "ja": "🗑️ 削除", "ko": "🗑️ 삭제"},
    "note_deleted": {"vi": "✅ Đã xóa ghi chú!", "en": "✅ Note deleted!", "zh": "✅ 已删除笔记！", "ja": "✅ メモを削除しました！", "ko": "✅ 노트를 삭제했습니다!"},
    "note_word": {"vi": "Ghi chú", "en": "Note", "zh": "笔记", "ja": "メモ", "ko": "노트"},
    "sources_label": {"vi": "Nguồn", "en": "Sources", "zh": "来源", "ja": "ソース", "ko": "소스"},
    # Studio/Settings inside Notebook
    "studio": {"vi": "🎥 Studio", "en": "🎥 Studio", "zh": "🎥 工作室", "ja": "🎥 スタジオ", "ko": "🎥 스튜디오"},
    "settings": {"vi": "⚙️ Cài đặt", "en": "⚙️ Settings", "zh": "⚙️ 设置", "ja": "⚙️ 設定", "ko": "⚙️ 설정"},
    "tab_notebook": {"vi": "📓 **Notebook**", "en": "📓 **Notebook**", "zh": "📓 **笔记本**", "ja": "📓 **ノートブック**", "ko": "📓 **노트북**"},
    "tab_studio": {"vi": "🎨 **Studio**", "en": "🎨 **Studio**", "zh": "🎨 **工作室**", "ja": "🎨 **スタジオ**", "ko": "🎨 **스튜디오**"},
    "tab_sources": {"vi": "📚 **Nguồn**", "en": "📚 **Source**", "zh": "📚 **来源**", "ja": "📚 **ソース**", "ko": "📚 **소스**"},
    "rename_notebook": {"vi": "Đổi tên notebook", "en": "Rename notebook", "zh": "重命名笔记本", "ja": "ノートブック名を変更", "ko": "노트북 이름 변경"},
    "edit_tags": {"vi": "Sửa tag của notebook", "en": "Edit notebook tags", "zh": "编辑笔记本标签", "ja": "ノートブックのタグを編集", "ko": "노트북 태그 편집"},
    "save_settings": {"vi": "Lưu cài đặt", "en": "Save settings", "zh": "保存设置", "ja": "設定を保存", "ko": "설정 저장"},
    "invalid_notebook_name": {"vi": "Vui lòng nhập tên notebook hợp lệ", "en": "Please enter a valid notebook name", "zh": "请输入有效的笔记本名称", "ja": "有効なノートブック名を入力してください", "ko": "유효한 노트북 이름을 입력하세요"},
    "notebook_settings_saved": {"vi": "Đã cập nhật cài đặt notebook", "en": "Notebook settings updated", "zh": "已更新笔记本设置", "ja": "ノートブックの設定を更新しました", "ko": "노트북 설정이 업데이트되었습니다"},
    "confirm_delete_title": {"vi": "Bạn có chắc muốn xóa notebook này?", "en": "Are you sure you want to delete this notebook?", "zh": "确定要删除此笔记本吗？", "ja": "このノートブックを削除してよろしいですか？", "ko": "이 노트북을 삭제하시겠습니까?"},
    "yes_delete": {"vi": "Đồng ý xóa", "en": "Yes, delete", "zh": "是的，删除", "ja": "はい、削除します", "ko": "예, 삭제"},
    "cancel": {"vi": "Hủy", "en": "Cancel", "zh": "取消", "ja": "キャンセル", "ko": "취소"},
    # Misc generic errors/warnings
    "error": {"vi": "❌ Lỗi", "en": "❌ Error", "zh": "❌ 错误", "ja": "❌ エラー", "ko": "❌ 오류"},
    "warning": {"vi": "⚠️ Cảnh báo", "en": "⚠️ Warning", "zh": "⚠️ 警告", "ja": "⚠️ 警告", "ko": "⚠️ 경고"},
}

# ===== Settings page labels (vi/en/zh/ja/ko) =====
SETTINGS_TEXTS = {
    "settings_title": {"vi": "⚙️ Cài đặt", "en": "⚙️ Settings", "zh": "⚙️ 设置", "ja": "⚙️ 設定", "ko": "⚙️ 설정"},
    "settings_subtitle": {"vi": "Cấu hình các thông số cho ứng dụng ElevateAI", "en": "Configure ElevateAI application", "zh": "配置 ElevateAI 应用", "ja": "ElevateAI アプリケーションを設定", "ko": "ElevateAI 애플리케이션 설정"},
    # Tabs
    "tab_model": {"vi": "🤖 Model", "en": "🤖 Model", "zh": "🤖 模型", "ja": "🤖 モデル", "ko": "🤖 모델"},
    "tab_search": {"vi": "🔍 Tìm kiếm", "en": "🔍 Search", "zh": "🔍 搜索", "ja": "🔍 検索", "ko": "🔍 검색"},
    "tab_audio": {"vi": "🔊 Âm thanh", "en": "🔊 Audio", "zh": "🔊 音频", "ja": "🔊 オーディオ", "ko": "🔊 오디오"},
    "tab_memory": {"vi": "🧠 Bộ nhớ", "en": "🧠 Memory", "zh": "🧠 记忆", "ja": "🧠 メモリ", "ko": "🧠 메모리"},
    "tab_interface": {"vi": "🎨 Giao diện", "en": "🎨 Interface", "zh": "🎨 界面", "ja": "🎨 インターフェース", "ko": "🎨 인터페이스"},
    "tab_advanced": {"vi": "⚙️ Nâng cao", "en": "⚙️ Advanced", "zh": "⚙️ 高级", "ja": "⚙️ 詳細設定", "ko": "⚙️ 고급"},
    # Model settings
    "temperature": {"vi": "Temperature", "en": "Temperature", "zh": "温度", "ja": "温度", "ko": "온도"},
    "temperature_help": {"vi": "Điều khiển độ ngẫu nhiên", "en": "Controls randomness", "zh": "控制随机性", "ja": "ランダム性を制御", "ko": "무작위성 제어"},
    "max_tokens": {"vi": "Max Tokens", "en": "Max Tokens", "zh": "最大Tokens", "ja": "最大トークン", "ko": "최대 토큰"},
    "max_tokens_help": {"vi": "Số token tối đa", "en": "Maximum number of tokens", "zh": "最大令牌数量", "ja": "最大トークン数", "ko": "최대 토큰 수"},
    "top_p": {"vi": "Top P", "en": "Top P", "zh": "Top P", "ja": "Top P", "ko": "Top P"},
    "top_p_help": {"vi": "Điều khiển đa dạng", "en": "Controls diversity", "zh": "控制多样性", "ja": "多様性を制御", "ko": "다양성 제어"},
    "frequency_penalty": {"vi": "Frequency Penalty", "en": "Frequency Penalty", "zh": "频率惩罚", "ja": "頻度ペナルティ", "ko": "빈도 패널티"},
    "presence_penalty": {"vi": "Presence Penalty", "en": "Presence Penalty", "zh": "出现惩罚", "ja": "出現ペナルティ", "ko": "존재 패널티"},
    "model": {"vi": "Model", "en": "Model", "zh": "模型", "ja": "モデル", "ko": "모델"},
    # Search settings
    "similarity_threshold": {"vi": "Ngưỡng tương đồng", "en": "Similarity Threshold", "zh": "相似度阈值", "ja": "類似度しきい値", "ko": "유사도 임계값"},
    "max_results_label": {"vi": "Số kết quả tối đa", "en": "Max Results", "zh": "最大结果数", "ja": "最大件数", "ko": "최대 결과 수"},
    "chunk_size": {"vi": "Kích thước chunk", "en": "Chunk Size", "zh": "分块大小", "ja": "チャンクサイズ", "ko": "청크 크기"},
    "chunk_overlap": {"vi": "Chồng lấp chunk", "en": "Chunk Overlap", "zh": "分块重叠", "ja": "チャンクの重なり", "ko": "청크 중첩"},
    "enable_web_search": {"vi": "Bật tìm kiếm web", "en": "Enable Web Search", "zh": "启用网页搜索", "ja": "ウェブ検索を有効化", "ko": "웹 검색 활성화"},
    "enable_function_calling": {"vi": "Bật Function Calling", "en": "Enable Function Calling", "zh": "启用函数调用", "ja": "関数呼び出しを有効化", "ko": "함수 호출 활성화"},
    # Audio settings
    "enable_tts": {"vi": "Bật Text-to-Speech", "en": "Enable Text-to-Speech", "zh": "启用文本转语音", "ja": "音声合成を有効化", "ko": "TTS 활성화"},
    "tts_voice": {"vi": "Giọng TTS", "en": "TTS Voice", "zh": "TTS 音色", "ja": "TTS ボイス", "ko": "TTS 음성"},
    "audio_sample_rate": {"vi": "Tần số lấy mẫu", "en": "Audio Sample Rate", "zh": "采样率", "ja": "サンプリング周波数", "ko": "샘플링 주파수"},
    "noise_reduction": {"vi": "Giảm tiếng ồn", "en": "Noise Reduction", "zh": "降噪", "ja": "ノイズ低減", "ko": "소음 감소"},
    "enable_vocal_separation": {"vi": "Tách giọng hát", "en": "Enable Vocal Separation", "zh": "启用人声分离", "ja": "ボーカル分離を有効化", "ko": "보컬 분리 활성화"},
    # Memory settings
    "enable_memory": {"vi": "Bật hệ thống ghi nhớ", "en": "Enable Memory System", "zh": "启用记忆系统", "ja": "メモリシステムを有効化", "ko": "메모리 시스템 활성화"},
    "max_memory_context": {"vi": "Số context tối đa", "en": "Max Memory Context", "zh": "最大记忆上下文", "ja": "最大メモリ文脈", "ko": "최대 메모리 컨텍스트"},
    "memory_consolidation_threshold": {"vi": "Ngưỡng củng cố bộ nhớ", "en": "Memory Consolidation Threshold", "zh": "记忆巩固阈值", "ja": "記憶統合しきい値", "ko": "메모리 통합 임계값"},
    "store_conversations": {"vi": "Lưu hội thoại", "en": "Store Conversations", "zh": "存储对话", "ja": "会話を保存", "ko": "대화 저장"},
    "memory_retention_days": {"vi": "Số ngày lưu giữ", "en": "Memory Retention (days)", "zh": "保留天数", "ja": "保持日数", "ko": "보존 기간(일)"},
    "auto_cleanup": {"vi": "Tự dọn bộ nhớ cũ", "en": "Auto Cleanup Old Memories", "zh": "自动清理旧记忆", "ja": "古い記憶を自動クリーンアップ", "ko": "오래된 메모리 자동 정리"},
    # Interface settings
    "theme": {"vi": "Giao diện", "en": "Theme", "zh": "主题", "ja": "テーマ", "ko": "테마"},
    "language": {"vi": "Ngôn ngữ", "en": "Language", "zh": "语言", "ja": "言語", "ko": "언어"},
    "auto_save": {"vi": "Tự động lưu", "en": "Auto Save Settings", "zh": "自动保存设置", "ja": "自動保存設定", "ko": "자동 저장 설정"},
    "show_processing_time": {"vi": "Hiển thị thời gian xử lý", "en": "Show Processing Time", "zh": "显示处理时间", "ja": "処理時間を表示", "ko": "처리 시간 표시"},
    "show_confidence_score": {"vi": "Hiển thị điểm tin cậy", "en": "Show Confidence Score", "zh": "显示置信分数", "ja": "信頼スコアを表示", "ko": "신뢰 점수 표시"},
    "enable_animations": {"vi": "Bật animation", "en": "Enable Animations", "zh": "启用动画", "ja": "アニメーションを有効化", "ko": "애니메이션 활성화"},
    # Advanced
    "max_file_size": {"vi": "Kích thước file tối đa (MB)", "en": "Max File Size (MB)", "zh": "最大文件大小（MB）", "ja": "最大ファイルサイズ（MB）", "ko": "최대 파일 크기(MB)"},
    "enable_debug_mode": {"vi": "Bật chế độ debug", "en": "Enable Debug Mode", "zh": "启用调试模式", "ja": "デバッグモードを有効化", "ko": "디버그 모드 활성화"},
    "enable_caching": {"vi": "Bật cache", "en": "Enable Caching", "zh": "启用缓存", "ja": "キャッシュを有効化", "ko": "캐시 활성화"},
    "log_level": {"vi": "Mức log", "en": "Log Level", "zh": "日志级别", "ja": "ログレベル", "ko": "로그 레벨"},
    "enable_metrics": {"vi": "Bật thu thập metrics", "en": "Enable Metrics Collection", "zh": "启用指标收集", "ja": "メトリクス収集を有効化", "ko": "메트릭 수집 활성화"},
    "backup_enabled": {"vi": "Bật tự động sao lưu", "en": "Enable Auto Backup", "zh": "启用自动备份", "ja": "自動バックアップを有効化", "ko": "자동 백업 활성화"},
    # Buttons and others
    "save_apply": {"vi": "💾 Lưu & Áp dụng", "en": "💾 Save & Apply", "zh": "💾 保存并应用", "ja": "💾 保存して適用", "ko": "💾 저장 및 적용"},
    "reset_defaults": {"vi": "🔄 Khôi phục mặc định", "en": "🔄 Reset to Defaults", "zh": "🔄 恢复默认", "ja": "🔄 既定値にリセット", "ko": "🔄 기본값으로 재설정"},
    "confirm_reset": {"vi": "⚠️ Xác nhận khôi phục", "en": "⚠️ Confirm Reset", "zh": "⚠️ 确认重置", "ja": "⚠️ リセットの確認", "ko": "⚠️ 재설정 확인"},
    "settings_summary": {"vi": "📊 Tổng quan cài đặt", "en": "📊 Settings Summary", "zh": "📊 设置概览", "ja": "📊 設定サマリー", "ko": "📊 설정 요약"},
}

def ts(key: str, lang: str = "vi") -> str:
    try:
        bundle = SETTINGS_TEXTS.get(key)
        if not bundle:
            return key
        # Prefer requested language, then English, then Vietnamese, then any
        return bundle.get(lang) or bundle.get("en") or bundle.get("vi") or next(iter(bundle.values()))
    except Exception:
        return key


def t(key: str, lang: str = "vi") -> str:
    """Translate helper with graceful fallback."""
    try:
        bundle = UI_TEXTS.get(key)
        if not bundle:
            return key
        if lang in bundle:
            return bundle[lang]
        # Prefer English, then Vietnamese, then any
        return bundle.get("en") or bundle.get("vi") or next(iter(bundle.values()))
    except Exception:
        return key


