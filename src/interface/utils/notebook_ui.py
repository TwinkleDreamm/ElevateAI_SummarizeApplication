from __future__ import annotations

from typing import List


class NotebookUI:
    @staticmethod
    def tabs_labels() -> List[str]:
        return ["📚 **Sources**", "📓 **Notebook**", "🎨 **Studio**"]

    @staticmethod
    def chat_style_css(max_height: int = 500) -> str:
        return f"""
        <style>
        .nb-chat-wrapper {{max-height: {max_height}px; overflow-y: auto; padding-right: 8px; border: 1px solid #e0e0e0; border-radius: 8px; padding: 16px; margin: 16px 0;}}
        .nb-chat-item {{margin: 10px 0; display: flex;}}
        .nb-chat-q {{justify-content: flex-start;}}
        .nb-chat-a {{justify-content: flex-end;}}
        .bubble {{max-width: 75%; padding: 10px 14px; border-radius: 12px;}}
        .q-bubble {{background:#eef2ff; border:1px solid #c7d2fe;}}
        .a-bubble {{background:#ecfdf5; border:1px solid #a7f3d0;}}
        </style>
        """

    @staticmethod
    def smooth_scroll_to_question_js() -> str:
        return """
        <script>
        setTimeout(function() {
            const questionSection = document.querySelector("h3");
            if (questionSection && questionSection.textContent.includes("Ask a Question")) {
                questionSection.scrollIntoView({ behavior: "smooth", block: "center" });
            }
        }, 100);
        </script>
        """

    @staticmethod
    def notebook_card_header_html(name: str, gradient_css: str) -> str:
        return f"""
        <div style="
            background: {gradient_css};
            border-radius: 12px 12px 0 0;
            padding: 15px;
            color: white;
            min-height: 90px;
            margin-bottom: 0;
        ">
            <h3 style="margin: 0; font-size: 25px; font-weight: bold; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">{name}</h3>
        </div>
        """

    @staticmethod
    def notebook_card_body_html(created_date: str, sources_count: int, tags_preview: str) -> str:
        return f"""
        <div style="
            background: white;
            border: 1px solid #e0e0e0;
            border-radius: 0 0 12px 12px;
            padding: 10px;
            margin-bottom: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            min-height: 80px;
            display: flex;
            flex-direction: column;
            gap: 6px;
        ">
            <div style="font-size: 14px; color: #999;">
                📅 {created_date} • 📄 {sources_count} sources
            </div>
            {tags_preview}
        </div>
        """
