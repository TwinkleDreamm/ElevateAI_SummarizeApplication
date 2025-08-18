from __future__ import annotations

import streamlit as st
import hashlib
from typing import List, Dict, Any
from datetime import datetime

from src.interface.notebooks import store
from src.interface.app_context import get_context
from src.interface.notebooks.ingest import ingest_uploaded_files, ingest_url
from src.interface.pages.notebook_helper import NotebookHelper


def _calculate_sources_hash(sources: List[store.SourceRef]) -> str:
    """Calculate hash of all sources to detect content changes."""
    if not sources:
        return ""

    # Create a string representation of all sources
    sources_data = []
    for s in sources:
        sources_data.append(f"{s.id}|{s.type}|{s.title}|{s.source_path_or_url}|{s.added_at}")

    # Sort to ensure consistent hash regardless of order
    sources_data.sort()
    sources_string = "|".join(sources_data)

    # Calculate hash
    return hashlib.md5(sources_string.encode()).hexdigest()


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
                    # Reset sources hash to trigger overview regeneration
                    store.update_notebook_sources_hash(nb.id, "")
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

        if added > 0:
            # Reset sources hash to trigger overview regeneration
            store.update_notebook_sources_hash(nb.id, "")

        st.success(f"Added {added} chunks.")


def _notebook_overview(nb: store.Notebook) -> list:
    with st.expander("📘 Overview & Example questions", expanded=False):
        st.markdown("### Overview")

        # Check if we need to regenerate overview and examples
        current_sources_count = len(nb.sources)
        current_sources_hash = _calculate_sources_hash(nb.sources)
        stored_overview = store.get_overview(nb.id)
        stored_examples = nb.examples

        # Get last known sources hash from database (more accurate than count)
        last_known_sources_hash = getattr(nb, 'last_sources_hash', "")

        # Detect if sources content actually changed (not just count)
        sources_changed = current_sources_hash != last_known_sources_hash

        # Only regenerate if sources changed OR no stored data exists
        needs_regeneration = (
            sources_changed or
            (not stored_overview and current_sources_count > 0) or
            (not stored_examples and current_sources_count > 0)
        )

        # Debug info (remove in production)
        if st.session_state.get('debug_overview', False):
            st.info(f"Debug: sources_changed={sources_changed}, needs_regeneration={needs_regeneration}, stored_examples={bool(stored_examples)}")
        
        if needs_regeneration:
            # Generate overview
            if not stored_overview or sources_changed:
                with st.spinner("Đang tạo overview..."):
                    ctx = get_context()
                    search = ctx["search_engine"]
                    if not search:
                        st.error("Search engine not available")
                        return
                try:
                    results = search.search("tóm tắt nội dung", k=5, threshold=0.0, filters={"notebook_id": nb.id})
                except Exception:
                    results = []

                overview = ""
                if not ctx.get("summarizer"):
                    st.warning("Summarizer not available - using fallback overview")
                elif not results:
                    st.info("No search results found for overview generation")

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
                # Update sources hash in notebook to track changes
                store.update_notebook_sources_hash(nb.id, current_sources_hash)
            else:
                overview = stored_overview
            
            # Generate examples
            if not stored_examples or sources_changed:
                with st.spinner("Đang tạo câu hỏi ví dụ..."):
                    examples = _generate_example_questions(nb)
                    if examples:
                        store.update_examples(nb.id, examples)
                        # Update sources hash in notebook to track changes
                        store.update_notebook_sources_hash(nb.id, current_sources_hash)
                    else:
                        st.warning("Could not generate example questions")
            else:
                examples = stored_examples
            
            # Sources hash now tracked in database, no need for session state
        else:
            # Use cached data
            overview = stored_overview
            examples = stored_examples
        
        st.write(overview)

        # Return examples for use in chat interface
        return examples
        





def _generate_example_questions(nb: store.Notebook) -> List[str]:
    return NotebookHelper.generate_example_questions(nb)


def _render_chat(nb: store.Notebook, examples=None):
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
            from src.interface.pages.notebook_ui import NotebookUI
            chat_style = NotebookUI.chat_style_css(max_height=500)
            st.markdown(chat_style, unsafe_allow_html=True)
            st.markdown('<div class="nb-chat-wrapper">', unsafe_allow_html=True)
            
            for item in st.session_state.chat_history[-50:]:
                st.markdown(f'<div class="nb-chat-item nb-chat-q"><div class="bubble q-bubble">{item["question"]}</div></div>', unsafe_allow_html=True)
                st.markdown(f'<div class="nb-chat-item nb-chat-a"><div class="bubble a-bubble">{item["answer"]}</div></div>', unsafe_allow_html=True)
                
                # Action buttons for each message
                bcols = st.columns([1,1,6])
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
                        st.success("✅ Note saved to Studio!")
                        st.rerun()  # Rerun to update button state
                
                with bcols[1]:
                    if include_sources_pref and item.get('sources'):
                        with st.popover("Sources"):
                            # Remove duplicate sources
                            unique_sources = list(set(item['sources'][:5]))
                            for s in unique_sources:
                                st.markdown(f"- {s}")
            
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info("💬 No chat history yet. Ask a question to start chatting!")

    st.markdown("---")
    
    # Auto scroll to this section if flag is set
    if st.session_state.get('auto_scroll_to_question', False):
        st.session_state['auto_scroll_to_question'] = False
        # Add JavaScript for smooth scrolling
        from src.interface.pages.notebook_ui import NotebookUI
        st.markdown(NotebookUI.smooth_scroll_to_question_js(), unsafe_allow_html=True)
    
    st.markdown("### Ask a Question")

    # Input area with container for scroll target
    with st.container(key="ask_question"):
        # Handle pending example question (must happen BEFORE widget is instantiated)
        if st.session_state.get('pending_example_question'):
            st.session_state['nb_chat_input'] = st.session_state['pending_example_question']
            del st.session_state['pending_example_question']

        # Clear the input on rerun if flagged (must happen BEFORE widget is instantiated)
        if st.session_state.get('clear_nb_input', False):
            st.session_state['nb_chat_input'] = ""
            st.session_state['clear_nb_input'] = False

        # Question input with button on the right
        input_col, button_col = st.columns([5, 1])
        with input_col:
            query = st.text_area("Your question", key="nb_chat_input", height=120, placeholder="Ask anything about the sources in this notebook…")
        with button_col:
            # Add spacing to align with text area center
            st.markdown("<div style='height: 45px;'></div>", unsafe_allow_html=True)
            ask_button = st.button("🚀 Ask", type="primary", use_container_width=True)

        # Add JavaScript for Enter key support (Enter to submit, Shift+Enter for new line)
        st.markdown("""
        <script>
        (function() {
            // Function to setup keyboard handler
            function setupKeyboardHandler() {
                const textArea = document.querySelector('textarea[aria-label="Your question"]');
                if (textArea && !textArea.hasAttribute('data-keyboard-setup')) {
                    textArea.setAttribute('data-keyboard-setup', 'true');

                    textArea.addEventListener('keydown', function(e) {
                        if (e.key === 'Enter' && !e.shiftKey) {
                            e.preventDefault();

                            // Check if there's text to submit
                            if (textArea.value.trim()) {
                                // Find and click the Ask button
                                const askButton = document.querySelector('button[kind="primary"]');
                                if (askButton) {
                                    askButton.click();
                                }
                            }
                        }
                        // Shift+Enter allows new line (default behavior)
                    });

                    console.log('Keyboard handler setup for question input');
                }
            }

            // Setup initially and after DOM changes
            function init() {
                setupKeyboardHandler();

                // Watch for DOM changes (Streamlit reruns)
                if (window.keyboardObserver) {
                    window.keyboardObserver.disconnect();
                }

                window.keyboardObserver = new MutationObserver(function(mutations) {
                    setTimeout(setupKeyboardHandler, 100);
                });

                window.keyboardObserver.observe(document.body, {
                    childList: true,
                    subtree: true
                });
            }

            // Initialize when DOM is ready
            if (document.readyState === 'loading') {
                document.addEventListener('DOMContentLoaded', init);
            } else {
                init();
            }
        })();
        </script>
        """, unsafe_allow_html=True)

    # Options row - compact layout
    col1, col2, _ = st.columns([2, 2, 4])
    with col1:
        use_cot = st.checkbox("Chain-of-thought", value=False)
    with col2:
        include_src = st.checkbox("Include sources", value=include_sources_pref)
        st.session_state['nb_include_sources'] = include_src

    # Handle both button click and Enter key (through form submission)
    submit_query = ask_button or st.session_state.get('submit_on_enter', False)
    if st.session_state.get('submit_on_enter', False):
        st.session_state['submit_on_enter'] = False

    if submit_query:
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
                    # Create unique sources list for context
                    unique_context_sources = list(set([r.metadata.get("source","unknown") for r in results]))
                    response = ctx["llm_client"].generate_response(messages, use_memory=True, store_in_memory=True, max_memory_context=3, context_sources=unique_context_sources)
                    answer = response.content
                else:
                    answer = summary_result.summary

                # Append new message to chat history with unique sources
                unique_sources = list(set([r.metadata.get('source','unknown') for r in results[:5]]))
                new_message = {
                    'id': f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
                    'question': query,
                    'answer': answer,
                    'timestamp': datetime.now().isoformat(),
                    'sources': unique_sources
                }
                st.session_state.chat_history.append(new_message)
                
                # Show success message
                st.success("Answer generated! Scroll up to see the chat history.")
                
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

    # Example questions section - moved below Ask a Question
    if examples:
        st.markdown("---")
        st.markdown("#### 💡 Example questions")
        st.caption("Click on any question to use it:")

        # Display example questions in a more compact layout
        for i, ex in enumerate(examples[:3]):
            if st.button(f"📝 {ex}", key=f"ex_{i}", use_container_width=True):
                # Set flag to populate input on next rerun
                st.session_state["pending_example_question"] = ex
                st.rerun()


def _render_studio_panel(nb: store.Notebook):
    """Render studio panel with saved notes."""
    st.subheader("🎨 Studio")
    
    # Initialize per-notebook Studio notes store
    if 'studio_notes' not in st.session_state:
        st.session_state.studio_notes = {}
    if nb.id not in st.session_state.studio_notes:
        st.session_state.studio_notes[nb.id] = []

    # One-time migration from legacy saved_notes list (if present)
    if 'saved_notes' in st.session_state and st.session_state.saved_notes:
        st.session_state.studio_notes[nb.id].extend(st.session_state.saved_notes)
        st.session_state.saved_notes = []

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
    except Exception:
        pass

    if not notes:
        st.info("No saved notes yet. Save notes from chat to see them here.")
        return
    
    # Display saved notes
    for i, note in enumerate(notes):
        with st.expander(f"📝 Note {i+1} - {note['timestamp'][:16]}", expanded=False):
            st.markdown(note['content'])
            # Remove duplicate sources and show unique ones
            unique_note_sources = list(set(note['sources'][:2]))
            st.caption(f"Sources: {', '.join(unique_note_sources)}")
            
            # Action buttons
            col1, col2 = st.columns(2)
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
                    st.success("✅ Note added to sources! (kept in Studio)")
                    st.rerun()
            
            with col2:
                if st.button("🗑️ Delete", key=f"delete_note_{note['id']}_{i}", 
                           help="Delete note"):
                    st.session_state.studio_notes[nb.id].pop(i)
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
            # Generate/get overview and examples
            overview_examples = _notebook_overview(nb)
            st.divider()
            # Use examples from overview generation (cached or newly generated)
            examples = overview_examples if overview_examples else (nb.examples if hasattr(nb, 'examples') and nb.examples else [])
            _render_chat(nb, examples)
        
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

    q, fav, dfrom, dto = _render_filters()
    notebooks = store.list_notebooks(q, fav, dfrom, dto)
    
    if not notebooks:
        st.info("No notebooks yet. Create your first notebook to get started!")
        return
    
    cols = st.columns(4)
    for i, nb in enumerate(notebooks):
        col_index = i % 4
        _render_notebook_card(nb, cols[col_index])


if __name__ == "__main__":
    main()


