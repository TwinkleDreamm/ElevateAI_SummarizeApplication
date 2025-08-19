from __future__ import annotations

import hashlib
from typing import List
from datetime import datetime

import streamlit as st

from src.interface.notebooks import store
from src.interface.app_context import get_context
from src.interface.utils.notebook_ui import NotebookUI
from src.interface.notebooks.ingest import ingest_uploaded_files, ingest_url


class NotebookHelper:
    @staticmethod
    def render_filters():
        with st.expander("🔍 Filter Notebooks", expanded=False):
            col1, col2, col3 = st.columns(3)

            with col1:
                q = st.text_input("Search by name/desc/tag", key="home_q", placeholder="Type to search...")
                favorite_only = st.checkbox("Favorites only", key="home_fav")

            with col2:
                date_from = st.date_input("From date", value=None, key="home_from")
                date_to = st.date_input("To date", value=None, key="home_to")

            with col3:
                st.write("")
                if st.button("Clear filters", key="clear_filters"):
                    st.session_state.pop("home_q", None)
                    st.session_state.pop("home_fav", None)
                    st.session_state.pop("home_from", None)
                    st.session_state.pop("home_to", None)
                    st.rerun()
        return q, favorite_only, date_from.isoformat() if date_from else None, date_to.isoformat() if date_to else None

    @staticmethod
    def render_notebook_card(nb: store.Notebook, container):
        """Render notebook card with performance optimizations."""
        with container:
            # Use container to isolate each notebook card
            with st.container():
                # Cache card content to avoid re-rendering
                card_key = f"card_{nb.id}_{hash(nb.name + str(nb.updated_at) if hasattr(nb, 'updated_at') else nb.created_at)}"
                
                if card_key not in st.session_state:
                    # Generate card content only once
                    st.session_state[card_key] = {
                        'name': nb.name,
                        'created_date': nb.created_at.split('T')[0] if nb.created_at else 'Unknown',
                        'sources_count': len(nb.sources) if nb.sources else 0,
                        'tags_preview': ", ".join(nb.tags[:3]) if nb.tags else "No tags",
                        'is_favorite': nb.is_favorite
                    }
                
                card_data = st.session_state[card_key]
                
                # Render card header with gradient
                palette = [
                    ("#667eea", "#764ba2"),
                    ("#f093fb", "#f5576c"),
                    ("#00c6ff", "#0072ff"),
                    ("#43e97b", "#38f9d7"),
                    ("#fa709a", "#fee140"),
                    ("#30cfd0", "#330867"),
                    ("#a18cd1", "#fbc2eb"),
                ]
                h = int(hashlib.md5(str(nb.id).encode()).hexdigest(), 16)
                c1, c2 = palette[h % len(palette)]
                gradient = f"linear-gradient(135deg, {c1} 0%, {c2} 100%)"

                st.markdown(NotebookUI.notebook_card_header_html(card_data['name'], gradient), unsafe_allow_html=True)

                # Render card body
                with st.container():
                    tags_preview = (
                        f'<div style="font-size: 14px; color: #999; margin-top: 5px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">🏷️ {card_data["tags_preview"]}</div>'
                        if card_data['tags_preview'] != "No tags" else ''
                    )
                    st.markdown(
                        NotebookUI.notebook_card_body_html(
                            created_date=card_data['created_date'],
                            sources_count=card_data['sources_count'],
                            tags_preview=tags_preview
                        ),
                        unsafe_allow_html=True
                    )

                # Action buttons
                fav_label = "★" if card_data['is_favorite'] else "☆"
                action_cols = st.columns(3)
                
                with action_cols[0]:
                    if st.button("📖", key=f"open_{nb.id}", help="Open notebook", use_container_width=True):
                        st.session_state["current_notebook_id"] = nb.id
                        st.query_params["view"] = "notebook"
                        st.rerun()
                
                with action_cols[1]:
                    if st.button(fav_label, key=f"fav_{nb.id}", help="Toggle favorite", use_container_width=True):
                        store.toggle_favorite(nb.id)
                        # Update cache after favorite toggle
                        if card_key in st.session_state:
                            st.session_state[card_key]['is_favorite'] = not card_data['is_favorite']
                        st.rerun()
                
                with action_cols[2]:
                    if st.button("🗑️", key=f"del_{nb.id}", help="Delete notebook", use_container_width=True):
                        st.session_state[f"confirm_del_{nb.id}"] = True

                # Confirmation dialog
                if st.session_state.get(f"confirm_del_{nb.id}"):
                    st.warning("Are you sure you want to delete this notebook?")
                    col_confirm1, col_confirm2 = st.columns(2)
                    with col_confirm1:
                        if st.button("Yes, delete", key=f"confirm_btn_{nb.id}", type="primary"):
                            store.delete_notebook(nb.id)
                            st.session_state.pop(f"confirm_del_{nb.id}", None)
                            # Clear card cache after deletion
                            if card_key in st.session_state:
                                del st.session_state[card_key]
                            st.rerun()
                    with col_confirm2:
                        if st.button("Cancel", key=f"cancel_btn_{nb.id}"):
                            st.session_state.pop(f"confirm_del_{nb.id}", None)
                            st.rerun()

    @staticmethod
    def generate_example_questions(nb: store.Notebook) -> List[str]:
        return [
                "Tóm tắt 5 ý chính của tài liệu này, kèm trích dẫn nguồn.",
                "Liệt kê mốc thời gian quan trọng và nguồn trích dẫn.",
                "So sánh hai quan điểm chính trong các nguồn, kèm trích dẫn."
        ]

        if not nb.sources:
            return [
                "Tóm tắt 5 ý chính của tài liệu này, kèm trích dẫn nguồn.",
                "Liệt kê mốc thời gian quan trọng và nguồn trích dẫn.",
                "So sánh hai quan điểm chính trong các nguồn, kèm trích dẫn."
            ]

        try:
            ctx = get_context()
            if ctx.get("llm_client") and ctx.get("search_engine"):
                search = ctx["search_engine"]
                try:
                    results = search.search("nội dung chính", k=3, threshold=0.0, filters={"notebook_id": nb.id})
                    if results:
                        source_info = f"Notebook có {len(nb.sources)} nguồn: " + ", ".join([f"{s.type} ({s.title})" for s in nb.sources[:3]])

                        from src.ai.prompt_engineer import PromptEngineer
                        pe = PromptEngineer()

                        prompt = f"""
                        Dựa trên thông tin sau về notebook, hãy tạo 3 câu hỏi ví dụ thông minh và hữu ích:

                        {source_info}

                        Yêu cầu:
                        1. Câu hỏi phải liên quan đến nội dung thực tế của notebook
                        2. Câu hỏi phải đa dạng về loại (tóm tắt, phân tích, so sánh, v.v.)
                        3. Câu hỏi phải bằng tiếng Việt
                        4. Mỗi câu hỏi phải ngắn gọn và rõ ràng

                        Trả về chỉ 3 câu hỏi, mỗi câu một dòng, không có đánh số.
                        """

                        messages = [{"role": "user", "content": prompt}]
                        response = ctx["llm_client"].generate_response(messages, use_memory=False)

                        if response and response.content:
                            lines = response.content.strip().split('\n')
                            questions = []
                            for line in lines:
                                line = line.strip()
                                if line and not line.startswith(('1.', '2.', '3.', '-', '*', '•')):
                                    questions.append(line)

                            if len(questions) >= 3:
                                return questions[:3]
                except Exception:
                    pass
        except Exception:
            pass

        source_types = [s.type for s in nb.sources]
        has_media = any(t in ['youtube', 'mp4', 'avi', 'mov', 'mkv', 'mp3', 'wav'] for t in source_types)
        has_docs = any(t in ['pdf', 'docx', 'txt'] for t in source_types)
        has_urls = any(t == 'url' for t in source_types)

        questions = []
        if has_media:
            questions.append("Tóm tắt nội dung chính của video/audio trong notebook này.")
        if has_docs:
            questions.append("Liệt kê các ý tưởng quan trọng từ các tài liệu, kèm trích dẫn nguồn.")
        if has_urls:
            questions.append("Phân tích thông tin từ các trang web và so sánh với nội dung khác.")
        if len(nb.sources) > 1:
            questions.append("So sánh và đối chiếu thông tin từ các nguồn khác nhau trong notebook.")
        questions.append("Tóm tắt 5 điểm quan trọng nhất từ tất cả nguồn, kèm trích dẫn.")
        while len(questions) < 3:
            questions.append("Đưa ra phân tích chi tiết về một chủ đề cụ thể trong notebook.")
        return questions[:3]
