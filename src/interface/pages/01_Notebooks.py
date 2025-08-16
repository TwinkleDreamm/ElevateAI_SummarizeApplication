from __future__ import annotations

import streamlit as st
from typing import List, Dict, Any
from datetime import datetime

from src.interface.notebooks import store
from src.interface.app_context import get_context
from src.interface.notebooks.ingest import ingest_uploaded_files, ingest_url
from src.interface.utils.notebook_helper import NotebookHelper
import json
import os
from pathlib import Path


def _get_notes_storage_path(notebook_id: str) -> Path:
    """Get the storage path for notes of a specific notebook."""
    data_dir = Path("data/notes")
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir / f"notebook_{notebook_id}_notes.json"


def _save_notes_to_storage(notebook_id: str, notes: list):
    """Save notes to persistent storage."""
    try:
        storage_path = _get_notes_storage_path(notebook_id)
        with open(storage_path, 'w', encoding='utf-8') as f:
            json.dump(notes, f, ensure_ascii=False, indent=2, default=str)
    except Exception as e:
        st.error(f"Failed to save notes to storage: {e}")


def _load_notes_from_storage(notebook_id: str) -> list:
    """Load notes from persistent storage."""
    try:
        storage_path = _get_notes_storage_path(notebook_id)
        if storage_path.exists():
            with open(storage_path, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        st.error(f"Failed to load notes from storage: {e}")
    return []


def _render_filters():
    return NotebookHelper.render_filters()


def _render_notebook_card(nb: store.Notebook, col):
    return NotebookHelper.render_notebook_card(nb, col)


def _render_sources_panel(nb: store.Notebook):
    """Render sources panel."""
    st.subheader("📚 Sources")
    
    # Sources list
    if not nb.sources:
        st.info("No sources yet. Add files or links.")
    else:
        for i, s in enumerate(nb.sources):
            col1, col2 = st.columns([4, 1])
            with col1:
                with st.expander(f"{s.title}", expanded=False):
                    st.caption(f"Type: {s.type} • Added: {s.added_at.split('T')[0]}")
                    if s.meta:
                        st.json(s.meta)
            with col2:
                if st.button("❌", key=f"del_source_{s.id}", help="Delete source"):
                    store.remove_source(nb.id, s.id)
                    st.rerun()

    st.markdown("---")
    st.subheader("Add sources")
    uploaded = st.file_uploader(
        "Upload files",
        type=["mp4","avi","mov","mkv","mp3","wav","pdf","docx","txt"],
        accept_multiple_files=True,
        key="nb_upload",
    )
    url = st.text_input("Or add a link", key="nb_url")
    if st.button("Add to notebook", key="nb_add_btn"):
        added = 0
        if uploaded:
            added += ingest_uploaded_files(nb.id, uploaded)
            for f in uploaded:
                store.add_source(nb.id, type="file", title=f.name, source_path_or_url=f.name)
        if url:
            added += ingest_url(nb.id, url)
            store.add_source(nb.id, type=("youtube" if ("youtube.com" in url or "youtu.be" in url) else "url"), title=url, source_path_or_url=url)
        st.success(f"Added {added} chunks.")


def _notebook_overview(nb: store.Notebook):
    with st.expander("📘 Overview & Example questions", expanded=False):
        st.markdown("### Overview")

        # Check if we need to regenerate overview and examples
        current_sources_count = len(nb.sources)
        cached_sources_count = st.session_state.get(f"sources_count_{nb.id}", 0)
        stored_overview = store.get_overview(nb.id)
        stored_examples = nb.examples
        
        # Only regenerate if sources count changed or no cached data exists
        needs_regeneration = (
            current_sources_count != cached_sources_count or 
            not stored_overview or 
            not stored_examples
        )
        
        if needs_regeneration:
            # Generate overview
            if not stored_overview or current_sources_count != cached_sources_count:
                with st.spinner("Đang tạo overview..."):
                    ctx = get_context()
                    search = ctx["search_engine"]
                try:
                    results = search.search("tóm tắt nội dung", k=5, threshold=0.0, filters={"notebook_id": nb.id})
                except Exception:
                    results = []

                overview = ""
                if ctx.get("summarizer") and results:
                    results_for_sum = [
                        {"text": r.text or r.metadata.get("text", ""), "score": r.score, "metadata": r.metadata}
                        for r in results
                    ]
                    try:
                        detected_langs = []
                        any_vi_chars = False
                        for r in results_for_sum:
                            meta = r.get("metadata", {}) or {}
                            lang = (meta.get("language") or meta.get("lang") or "").lower()
                            text_all = (r.get("text") or "")
                            vi_chars = sum(ch in "àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ" for ch in text_all.lower())
                            if vi_chars > 0:
                                any_vi_chars = True
                                lang = lang or "vi"
                            detected_langs.append(lang or "")
                        target_lang = "vi" if (any_vi_chars or any(l.startswith("vi") for l in detected_langs)) else ("en" if any(l.startswith("en") for l in detected_langs) else "vi")
                        if target_lang.startswith("vi"):
                            lang_instruction = "Hãy viết phần tổng quan bằng tiếng Việt, văn phong rõ ràng, mạch lạc."
                            role_instruction = "Bạn phải luôn trả lời bằng tiếng Việt."
                        else:
                            lang_instruction = "Write the overview in English with clear, concise language."
                            role_instruction = "You must always respond in English."
                        summary_result = ctx["summarizer"].summarize_chunks(
                            results_for_sum,
                            query="Provide a concise 5-10 sentence overview of key ideas.",
                            extra_instructions=lang_instruction,
                            role=role_instruction,
                        )
                        overview = summary_result.summary
                    except Exception:
                        overview = ""
                if not overview:
                    overview = nb.description or "This notebook contains your uploaded sources. Ask questions on the right."
                
                # Store the overview in database
                store.update_overview(nb.id, overview)
            else:
                overview = stored_overview
            
            # Generate examples
            if not stored_examples or current_sources_count != cached_sources_count:
                with st.spinner("Đang tạo câu hỏi ví dụ..."):
                    examples = _generate_example_questions(nb)
                    store.update_examples(nb.id, examples)
            else:
                examples = stored_examples
            
            # Update cached sources count
            st.session_state[f"sources_count_{nb.id}"] = current_sources_count
        else:
            # Use cached data
            overview = stored_overview
            examples = stored_examples
        
        st.write(overview)
        
        # Show cache status
        if not needs_regeneration:
            st.caption("📝 Overview và Examples đã được lưu cache")

        st.markdown("#### Example questions")
        cols = st.columns(len(examples))
        for i, ex in enumerate(examples[:3]):
            with cols[i]:
                if st.button(ex, key=f"ex_{i}"):
                    st.session_state["nb_chat_input"] = ex


def _generate_example_questions(nb: store.Notebook) -> List[str]:
    return NotebookHelper.generate_example_questions(nb)


def _render_chat(nb: store.Notebook):
    """Render chat interface with conversation view and save note per answer."""
    ctx = get_context()

    # Initialize chat history in session
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []  # list of {id, question, answer, timestamp, sources}
    
    # Initialize button counter to ensure unique keys
    if 'chat_button_counter' not in st.session_state:
        st.session_state.chat_button_counter = 0

    # Preference for including sources (used when rendering history)
    include_sources_pref = st.session_state.get('nb_include_sources', True)

    # Ensure Studio notes store exists for this notebook
    if 'studio_notes' not in st.session_state:
        st.session_state.studio_notes = {}
    if nb.id not in st.session_state.studio_notes:
        st.session_state.studio_notes[nb.id] = []

    # Status placeholder
    status_ph = st.empty()

    # Show searching status when searching
    if st.session_state.get('is_searching', False):
        status_ph.info("Searching and answering…")
    else:
        # Chat history display area - only show when not searching
        st.markdown("### Chat History")
        if st.session_state.chat_history:
            st.info(f"📝 Chat history has {len(st.session_state.chat_history)} messages")
            
            # Chat history styling
            from src.interface.utils.notebook_ui import NotebookUI
            chat_style = NotebookUI.chat_style_css(max_height=500)
            st.markdown(chat_style, unsafe_allow_html=True)
            st.markdown('<div class="nb-chat-wrapper">', unsafe_allow_html=True)
            
            for item in st.session_state.chat_history[-50:]:
                st.markdown(f'<div class="nb-chat-item nb-chat-q"><div class="bubble q-bubble">{item["question"]}</div></div>', unsafe_allow_html=True)
                st.markdown(f'<div class="nb-chat-item nb-chat-a"><div class="bubble a-bubble">{item["answer"]}</div></div>', unsafe_allow_html=True)
                
                # Action buttons for each message - now with 3 columns to accommodate Speak button
                bcols = st.columns([1,1,1,3])
                with bcols[0]:
                    # Stable key per chat message to avoid key drift across reruns
                    button_key = f"save_note_{item['id']}"
                    
                    # Check if this note was already saved
                    # Use per-notebook storage for studio notes
                    notes_for_this_nb = st.session_state.studio_notes.get(nb.id, [])
                    note_already_saved = any(n.get('original_chat_id') == item['id'] for n in notes_for_this_nb)
                    
                    if st.button("💾 Save Note" if not note_already_saved else "✅ Saved", 
                               key=button_key, 
                               disabled=note_already_saved):
                        # Ensure per-notebook notes store exists
                        if 'studio_notes' not in st.session_state:
                            st.session_state.studio_notes = {}
                        if nb.id not in st.session_state.studio_notes:
                            st.session_state.studio_notes[nb.id] = []
                        note = {
                            'id': f"note_{item['id']}",
                            'original_chat_id': item['id'],  # Track original chat message
                            'content': item['answer'],
                            'timestamp': datetime.now().isoformat(),
                            'sources': item.get('sources', []),
                            'question': item.get('question', ''),
                            'added_to_source': False
                        }
                        st.session_state.studio_notes[nb.id].append(note)
                        
                        # Save to persistent storage
                        _save_notes_to_storage(nb.id, st.session_state.studio_notes[nb.id])
                        
                        st.success("✅ Note saved to Studio!")
                        st.rerun()  # Rerun to update button state
                
                with bcols[1]:
                    # Speak button for text-to-speech
                    if st.button("🔊 Speak", key=f"speak_chat_{item['id']}", 
                               help="Listen to this answer", use_container_width=True):
                        try:
                            ctx = get_context()
                            if ctx.get("tts_client"):
                                with st.spinner("🎵 Generating audio..."):
                                    # Get answer content for TTS
                                    answer_text = item['answer']
                                    
                                    # Truncate text if too long (TTS has limits)
                                    max_length = 4000
                                    if len(answer_text) > max_length:
                                        answer_text = answer_text[:max_length] + "..."
                                        st.warning(f"⚠️ Answer was truncated for TTS (max {max_length} characters)")
                                    
                                    # Generate audio using TTS client
                                    audio_data = ctx["tts_client"].text_to_speech(
                                        text=answer_text,
                                        voice="alloy",  # Fixed voice
                                        model="tts-1",  # Fixed model - you can change this
                                        instructions="Speak clearly and at a moderate pace, suitable for answer reading."
                                    )
                                    
                                    if audio_data:
                                        # Create audio player with autoplay
                                        st.audio(audio_data, format="audio/mp3", start_time=0)
                                        st.success("🎵 Audio generated.")
                                    else:
                                        st.error("❌ Failed to generate audio")
                            else:
                                st.error("❌ TTS service not available")
                        except Exception as e:
                            st.error(f"❌ Error generating speech: {str(e)}")
                
                with bcols[2]:
                    if include_sources_pref and item.get('sources'):
                        with st.popover("Sources"):
                            for s in item['sources'][:5]:
                                st.markdown(f"- {s}")
            
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info("💬 No chat history yet. Ask a question to start chatting!")

    st.markdown("---")
    
    # Auto scroll to this section if flag is set
    if st.session_state.get('auto_scroll_to_question', False):
        st.session_state['auto_scroll_to_question'] = False
        # Add JavaScript for smooth scrolling
        from src.interface.utils.notebook_ui import NotebookUI
        st.markdown(NotebookUI.smooth_scroll_to_question_js(), unsafe_allow_html=True)
    
    st.markdown("### Ask a Question")

    # Input area with container for scroll target
    with st.container(key="ask_question"):
        # Clear the input on rerun if flagged (must happen BEFORE widget is instantiated)
        if st.session_state.get('clear_nb_input', False):
            st.session_state['nb_chat_input'] = ""
            st.session_state['clear_nb_input'] = False
        query = st.text_area("Your question", key="nb_chat_input", height=120, placeholder="Ask anything about the sources in this notebook…")
    col1, col2 = st.columns([1,1])
    with col1:
        use_cot = st.checkbox("Chain-of-thought", value=False)
    with col2:
        include_src = st.checkbox("Include sources", value=include_sources_pref)
        st.session_state['nb_include_sources'] = include_src

        if st.button("Ask", type="primary"):
            if not query.strip():
                st.warning("Please enter a question")
                return
            
            # Set searching flag to hide chat history and show status
            st.session_state['is_searching'] = True
            
            try:
                results = ctx["search_engine"].search(query, k=10, threshold=0.2, filters={"notebook_id": nb.id})
                if not results:
                    st.info("No relevant chunks found in this notebook.")
                    # Reset searching flag to show chat history when no results
                    st.session_state['is_searching'] = False
                    return
                
                rag_results = [
                    {"text": r.text or r.metadata.get("text", ""), "score": r.score, "metadata": r.metadata}
                    for r in results
                ]
                # Detect question language and enforce consistent answer language
                vi_char_set = "àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ"
                is_vietnamese = any(ch in vi_char_set for ch in query.lower())
                lang_instruction = (
                    "Hãy trả lời bằng tiếng Việt, văn phong rõ ràng, mạch lạc."
                    if is_vietnamese else
                    "Answer in English with clear and concise language."
                )
                role_instruction = (
                    "Bạn phải luôn trả lời bằng tiếng Việt."
                    if is_vietnamese else
                    "You must always respond in English."
                )
                summary_result = ctx["summarizer"].summarize_chunks(
                    rag_results,
                    query=query,
                    use_chain_of_thought=use_cot,
                    extra_instructions=lang_instruction,
                    role=role_instruction,
                )
                from src.ai.prompt_engineer import PromptEngineer
                pe = PromptEngineer()
                messages = pe.build_qa_prompt(context=summary_result.summary, question=query, use_chain_of_thought=use_cot)
                messages.insert(0, {"role": "system", "content": role_instruction})
                
                if ctx.get("llm_client"):
                    response = ctx["llm_client"].generate_response(messages, use_memory=True, store_in_memory=True, max_memory_context=3, context_sources=[r.metadata.get("source","unknown") for r in results])
                    answer = response.content
                else:
                    answer = summary_result.summary

                # Append new message to chat history
                new_message = {
                    'id': f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
                    'question': query,
                    'answer': answer,
                    'timestamp': datetime.now().isoformat(),
                    'sources': [r.metadata.get('source','unknown') for r in results[:5]]
                }
                st.session_state.chat_history.append(new_message)
                
                # Show success message
                st.success("✅ Answer generated! Scroll up to see the chat history.")
                
                # Reset searching flag to show chat history
                st.session_state['is_searching'] = False
                
                # Set flags for UI behavior on rerun
                st.session_state['auto_scroll_to_question'] = True
                st.session_state['clear_nb_input'] = True
                
                # Force rerun to show updated chat history
                st.rerun()
                
            except Exception as e:
                st.error(f"Error generating answer: {str(e)}")
                # Reset searching flag to show chat history after error
                st.session_state['is_searching'] = False


def _render_studio_panel(nb: store.Notebook):
    """Render studio panel with saved notes."""
    st.subheader("🎨 Studio")
    
    # Initialize per-notebook Studio notes store
    if 'studio_notes' not in st.session_state:
        st.session_state.studio_notes = {}
    if nb.id not in st.session_state.studio_notes:
        st.session_state.studio_notes[nb.id] = []
    
    # Load notes from persistent storage if session state is empty
    if not st.session_state.studio_notes[nb.id]:
        stored_notes = _load_notes_from_storage(nb.id)
        if stored_notes:
            st.session_state.studio_notes[nb.id] = stored_notes
            st.success(f"📚 Loaded {len(stored_notes)} notes from storage")

    # One-time migration from legacy saved_notes list (if present)
    if 'saved_notes' in st.session_state and st.session_state.saved_notes:
        st.session_state.studio_notes[nb.id].extend(st.session_state.saved_notes)
        st.session_state.saved_notes = []
        # Save migrated notes to storage
        _save_notes_to_storage(nb.id, st.session_state.studio_notes[nb.id])

    notes = st.session_state.studio_notes[nb.id]
    # Deduplicate notes by original_chat_id (fallback id)
    try:
        unique_notes_map = {}
        for n in notes:
            dedup_key = n.get('original_chat_id') or n.get('id')
            if dedup_key not in unique_notes_map:
                unique_notes_map[dedup_key] = n
        if len(unique_notes_map) != len(notes):
            st.session_state.studio_notes[nb.id] = list(unique_notes_map.values())
            notes = st.session_state.studio_notes[nb.id]
            # Save deduplicated notes to storage
            _save_notes_to_storage(nb.id, notes)
    except Exception:
        pass

    if not notes:
        st.info("No saved notes yet. Save notes from chat to see them here.")
        return
    
    # Display saved notes
    for i, note in enumerate(notes):
        with st.expander(f"📝 Note {i+1} - {note['timestamp'][:16]}", expanded=False):
            st.markdown(note['content'])
            st.caption(f"Sources: {', '.join(note['sources'][:2])}")
            
            # Action buttons - now with 3 columns to accommodate Speak button
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                add_label = "📚 Add to Source" if not note.get('added_to_source') else "✅ Added"
                if st.button(add_label, key=f"add_to_source_{note['id']}_{i}", 
                           help="Convert note to source", disabled=note.get('added_to_source', False)):
                    # Add note as a new source
                    store.add_source(
                        nb.id, 
                        type="note", 
                        title=f"Note from {note['timestamp'][:10]}", 
                        source_path_or_url=note['content'][:100] + "..."
                    )
                    # Mark note as added to sources but keep it in Studio
                    note['added_to_source'] = True
                    # Save updated notes to storage
                    _save_notes_to_storage(nb.id, st.session_state.studio_notes[nb.id])
                    st.success("✅ Note added to sources! (kept in Studio)")
                    st.rerun()
            
            with col2:
                # Speak button for text-to-speech
                if st.button("🔊 Speak", key=f"speak_note_{note['id']}_{i}", 
                           help="Listen to this note", use_container_width=True):
                    try:
                        # Get context and TTS client with fallback
                        ctx = get_context()
                        tts_client = ctx.get("tts_client")
                        
                        # If TTS client is None, try to re-initialize it
                        if tts_client is None:
                            st.info("🔄 Re-initializing TTS client...")
                            try:
                                from src.ai.tts_client import TTSClient
                                tts_client = TTSClient()
                                # Update context
                                ctx["tts_client"] = tts_client
                                st.success("✅ TTS client re-initialized!")
                            except Exception as init_error:
                                st.error(f"❌ Failed to re-initialize TTS client: {init_error}")
                                return
                        
                        if tts_client:
                            with st.spinner("🎵 Generating audio..."):
                                # Get note content for TTS
                                note_text = note['content']
                                    
                                # Truncate text if too long (TTS has limits)
                                max_length = 4000
                                if len(note_text) > max_length:
                                    note_text = note_text[:max_length] + "..."
                                    st.warning(f"⚠️ Note was truncated for TTS (max {max_length} characters)")
                                    
                                # Generate audio using TTS client with fixed model
                                audio_data = tts_client.text_to_speech(
                                    text=note_text,
                                    voice="alloy",  # Fixed voice
                                    model="tts-1",  # Fixed model - you can change this
                                    instructions="Speak clearly and at a moderate pace, suitable for note reading."
                                )
                                
                                if audio_data:
                                    # Create audio player with autoplay
                                    st.audio(audio_data, format="audio/mp3", start_time=0)
                                    st.success("🎵 Audio generated")
                                else:
                                    st.error("❌ Failed to generate audio")
                        else:
                            st.error("❌ TTS service not available after re-initialization")
                    except Exception as e:
                        st.error(f"❌ Error generating speech: {str(e)}")
                        st.info("💡 This might be due to API rate limits or network issues")
            
            with col3:
                if st.button("🗑️ Delete", key=f"delete_note_{note['id']}_{i}", 
                           help="Delete note"):
                    st.session_state.studio_notes[nb.id].pop(i)
                    # Save updated notes to storage
                    _save_notes_to_storage(nb.id, st.session_state.studio_notes[nb.id])
                    st.success("✅ Note deleted!")
                    st.rerun()


def _render_create_view():
    edit_notebook_id = st.session_state.get("edit_notebook_id")
    is_edit_mode = edit_notebook_id is not None

    if is_edit_mode:
        st.title("Edit notebook")
        notebook = store.get_notebook(edit_notebook_id)
        if not notebook:
            st.error("Notebook not found")
            st.query_params["view"] = "list"
            st.rerun()
            return
    else:
        st.title("Create a new notebook")

    with st.form("create_notebook_form"):
        name = st.text_input(
            "Notebook name",
            placeholder="e.g., Marketing Q4 Reports",
            value=notebook.name if is_edit_mode else ""
        )
        desc = st.text_area(
            "Description (optional)",
            value=notebook.description if is_edit_mode else ""
        )
        tags = st.text_input(
            "Tags (comma-separated)",
            value=", ".join(notebook.tags) if is_edit_mode and notebook.tags else ""
        )
        uploaded = st.file_uploader(
            "Upload files",
            type=["mp4","avi","mov","mkv","mp3","wav","pdf","docx","txt"],
            accept_multiple_files=True,
        )
        url = st.text_input("Add a link (web page or YouTube)")
        submitted = st.form_submit_button("Save" if is_edit_mode else "Create", type="primary")

    if submitted:
        if not name.strip():
            st.error("Please provide a notebook name")
            return

        if is_edit_mode:
            store.update_notebook(
                edit_notebook_id,
                name=name.strip(),
                description=desc.strip(),
                tags=[t.strip() for t in tags.split(',') if t.strip()]
            )
            st.success("Notebook updated successfully!")
            st.session_state.pop("edit_notebook_id", None)
            st.session_state["current_notebook_id"] = edit_notebook_id
            st.query_params["view"] = "notebook"
            st.rerun()
        else:
            nb = store.create_notebook(name=name.strip(), description=desc.strip(), tags=[t.strip() for t in tags.split(',') if t.strip()])
            added = 0
            if uploaded:
                added += ingest_uploaded_files(nb.id, uploaded)
                for f in uploaded:
                    store.add_source(nb.id, type="file", title=f.name, source_path_or_url=f.name)
            if url:
                added += ingest_url(nb.id, url)
                store.add_source(nb.id, type=("youtube" if ("youtube.com" in url or "youtu.be" in url) else "url"), title=url, source_path_or_url=url)

            st.success(f"Notebook created. Added {added} chunks to knowledge base.")
            st.session_state["current_notebook_id"] = nb.id
            st.query_params["view"] = "notebook"
            st.rerun()

    if st.button("← Back to Notebooks"):
        st.session_state.pop("edit_notebook_id", None)
        st.query_params["view"] = "list"
        st.rerun()


def main():
    st.set_page_config(page_title="Notebooks", page_icon="📓", layout="wide")
    
    # Load custom CSS
    with open("src/interface/styles/notebook_layout.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

    action = st.query_params.get("action", None)
    notebook_id = st.query_params.get("id", None)
    view = st.query_params.get("view", "list")

    if action == "open" and notebook_id:
        st.session_state["current_notebook_id"] = notebook_id
        st.query_params["view"] = "notebook"
        st.rerun()
    elif action == "favorite" and notebook_id:
        store.toggle_favorite(notebook_id)
        st.rerun()
    elif action == "edit" and notebook_id:
        st.session_state["edit_notebook_id"] = notebook_id
        st.query_params["view"] = "create"
        st.rerun()
    elif action == "delete" and notebook_id:
        st.session_state[f"confirm_del_{notebook_id}"] = True

    if view == "notebook":
        nb_id = st.session_state.get("current_notebook_id")
        if not nb_id:
            st.warning("No notebook selected. Returning to list.")
            st.query_params["view"] = "list"
            st.rerun()
            return
        nb = store.get_notebook(nb_id)
        if not nb:
            st.warning("Notebook not found. Returning to list.")
            st.query_params["view"] = "list"
            st.rerun()
            return

        # Header
        header_cols = st.columns([9,1])
        with header_cols[0]:
            st.title(nb.name)
            st.caption(nb.description or "")
            # Info chips: sources count • tags • created date
            created_date = (nb.created_at or "").split("T")[0]
            sources_count = len(nb.sources) if nb.sources else 0
            tags_text = ", ".join(nb.tags[:5]) if getattr(nb, 'tags', None) else "No tags"
            st.markdown(
                f"**📄 {sources_count} sources** • **🏷️ {tags_text}** • **📅 {created_date}**"
            )
        with header_cols[1]:
            if st.button("📓 Notebooks"):
                st.query_params["view"] = "list"
                st.rerun()

        # Three-tab layout: Sources | Notebook | Studio
        tab_sources, tab_notebook, tab_studio = st.tabs(["📚 **Sources**", "📓 **Notebook**", "🎨 **Studio**"])
        
        with tab_sources:
            _render_sources_panel(nb)
        
        with tab_notebook:
            _notebook_overview(nb)
            st.divider()
            _render_chat(nb)
        
        with tab_studio:
            _render_studio_panel(nb)
        
        return

    if view == "create":
        _render_create_view()
        return

    st.title("📓 ElevateAI Notebooks")
    st.markdown("Create, organize and chat with your knowledge notebooks.")

    create_col = st.container()
    with create_col:
        if st.button("Create New Notebook", type="primary"):
            st.query_params["view"] = "create"
            st.rerun()

    st.markdown("---")
    st.subheader("Your Notebooks")

    # Lazy load filters and notebooks
    with st.spinner("Loading notebooks..."):
        q, fav, dfrom, dto = _render_filters()
        
        # Cache notebooks in session state to avoid reloading
        cache_key = f"notebooks_cache_{hash(f'{q}_{fav}_{dfrom}_{dto}')}"
        if cache_key not in st.session_state:
            st.session_state[cache_key] = store.list_notebooks(q, fav, dfrom, dto)
        
        notebooks = st.session_state[cache_key]
        
        if not notebooks:
            st.info("No notebooks yet. Create your first notebook to get started!")
            return
        
        # Lazy render notebooks with pagination
        page_size = 8  # Show 8 notebooks per page
        current_page = st.session_state.get('notebooks_page', 0)
        total_pages = (len(notebooks) + page_size - 1) // page_size
        
        # Pagination controls
        if total_pages > 1:
            col1, col2, col3 = st.columns([1, 2, 1])
            with col1:
                if st.button("← Previous", disabled=current_page == 0):
                    st.session_state['notebooks_page'] = max(0, current_page - 1)
                    st.rerun()
            with col2:
                st.write(f"Page {current_page + 1} of {total_pages}")
            with col3:
                if st.button("Next →", disabled=current_page >= total_pages - 1):
                    st.session_state['notebooks_page'] = min(total_pages - 1, current_page + 1)
                    st.rerun()
        
        # Calculate start and end indices for current page
        start_idx = current_page * page_size
        end_idx = min(start_idx + page_size, len(notebooks))
        current_notebooks = notebooks[start_idx:end_idx]
        
        # Render notebooks in grid with lazy loading
        cols = st.columns(4)
        for i, nb in enumerate(current_notebooks):
            col_index = i % 4
            with cols[col_index]:
                # Use container to isolate each notebook card
                with st.container():
                    _render_notebook_card(nb, cols[col_index])
        
        # Show total count
        st.caption(f"Showing {start_idx + 1}-{end_idx} of {len(notebooks)} notebooks")
        
        # Clear cache button for debugging
        if st.button("🔄 Refresh Notebooks", help="Clear cache and reload notebooks"):
            # Clear all notebook caches
            for key in list(st.session_state.keys()):
                if key.startswith('notebooks_cache_'):
                    del st.session_state[key]
            st.session_state['notebooks_page'] = 0
            st.rerun()


if __name__ == "__main__":
    main()


