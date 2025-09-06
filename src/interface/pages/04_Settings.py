from __future__ import annotations

import streamlit as st
import json
import os
from pathlib import Path
from typing import Dict, Any
import toml

from src.interface.components import UIComponents
from src.interface.app_context import get_context
from src.utils.settings_manager import settings_manager, get_settings, set_setting
from config.settings import settings
from src.interface.utils.prompt_text import UI_TEXTS, SETTINGS_TEXTS, ts, t
from src.utils.logger import logger


def _get_lang() -> str:
    try:
        lang = st.session_state.get('language', settings.default_language)
        try:
            logger.info(f"[SettingsPage] _get_lang -> {lang}")
        except Exception:
            pass
        return lang
    except Exception as e:
        try:
            logger.warning(f"[SettingsPage] _get_lang failed: {e}; fallback 'vi'")
        except Exception:
            pass
        return 'vi'


def save_settings_to_session(settings_dict: Dict[str, Any]):
    """Lưu settings vào session state."""
    settings_manager.save_to_session(settings_dict)
    st.success("✅ Settings đã được lưu thành công!")


def save_settings_to_file(settings_dict: Dict[str, Any]):
    """Lưu settings vào file cấu hình."""
    if settings_manager.save_to_file(settings_dict):
        st.success("✅ Settings đã được lưu vào file cấu hình!")
    else:
        st.error("❌ Lỗi khi lưu file cấu hình!")


def validate_and_save_settings(settings_dict: Dict[str, Any]):
    """Validate và lưu settings."""
    # Validate settings
    errors = settings_manager.validate_settings(settings_dict)
    
    if errors:
        st.error("❌ Có lỗi trong cài đặt:")
        for field, error in errors.items():
            st.error(f"• {field}: {error}")
        return False
    
    # Save settings
    settings_manager.save_to_session(settings_dict)
    settings_manager.save_to_file(settings_dict)
    
    # Apply settings to application
    settings_manager.apply_settings_to_app()
    
    st.success("✅ Settings đã được lưu và áp dụng thành công!")
    try:
        st.rerun()
    except Exception:
        pass
    return True


def render_model_settings():
    lang = _get_lang()
    """Render phần cài đặt model."""
    st.subheader(ts("tab_model", lang))
    
    col1, col2 = st.columns(2)
    
    with col1:
        temperature = st.slider(
            ts("temperature", lang),
            min_value=0.0,
            max_value=2.0,
            value=st.session_state.get('openai_temperature', settings.openai_temperature),
            step=0.1,
            help=ts("temperature_help", lang)
        )
        
        max_tokens = st.number_input(
            ts("max_tokens", lang),
            min_value=100,
            max_value=8000,
            value=st.session_state.get('openai_max_tokens', settings.openai_max_tokens),
            step=100,
            help=ts("max_tokens_help", lang)
        )
        
        top_p = st.slider(
            ts("top_p", lang),
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.get('openai_top_p', settings.openai_top_p),
            step=0.1,
            help=ts("top_p_help", lang)
        )
    
    with col2:
        frequency_penalty = st.slider(
            ts("frequency_penalty", lang),
            min_value=-2.0,
            max_value=2.0,
            value=st.session_state.get('openai_frequency_penalty', settings.openai_frequency_penalty),
            step=0.1,
            help=""
        )
        
        presence_penalty = st.slider(
            ts("presence_penalty", lang),
            min_value=-2.0,
            max_value=2.0,
            value=st.session_state.get('openai_presence_penalty', settings.openai_presence_penalty),
            step=0.1,
            help=""
        )
        
        model_name = st.selectbox(
            ts("model", lang),
            options=["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo", "claude-3-opus", "claude-3-sonnet"],
            index=0,
            help=ts("model", lang)
        )
    
    return {
        'openai_temperature': temperature,
        'openai_max_tokens': max_tokens,
        'openai_top_p': top_p,
        'openai_frequency_penalty': frequency_penalty,
        'openai_presence_penalty': presence_penalty,
        'openai_chat_model': model_name
    }


def render_search_settings():
    lang = _get_lang()
    """Render phần cài đặt tìm kiếm."""
    st.subheader(ts("tab_search", lang))
    
    col1, col2 = st.columns(2)
    
    with col1:
        similarity_threshold = st.slider(
            ts("similarity_threshold", lang),
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.get('similarity_threshold', settings.similarity_threshold),
            step=0.05,
            help=ts("similarity_threshold", lang)
        )
        
        max_results = st.number_input(
            ts("max_results_label", lang),
            min_value=1,
            max_value=50,
            value=st.session_state.get('max_results', settings.max_results),
            step=1,
            help=ts("max_results_label", lang)
        )
        
        chunk_size = st.number_input(
            ts("chunk_size", lang),
            min_value=100,
            max_value=2000,
            value=st.session_state.get('chunk_size', settings.chunk_size),
            step=100,
            help=ts("chunk_size", lang)
        )
    
    with col2:
        chunk_overlap = st.number_input(
            ts("chunk_overlap", lang),
            min_value=0,
            max_value=500,
            value=st.session_state.get('chunk_overlap', settings.chunk_overlap),
            step=50,
            help=ts("chunk_overlap", lang)
        )
        
        enable_web_search = st.checkbox(
            ts("enable_web_search", lang),
            value=st.session_state.get('enable_web_search', settings.enable_web_search),
            help=ts("enable_web_search", lang)
        )
        
        enable_function_calling = st.checkbox(
            ts("enable_function_calling", lang),
            value=st.session_state.get('enable_function_calling', settings.enable_function_calling),
            help=ts("enable_function_calling", lang)
        )
    
    return {
        'similarity_threshold': similarity_threshold,
        'max_results': max_results,
        'chunk_size': chunk_size,
        'chunk_overlap': chunk_overlap,
        'enable_web_search': enable_web_search,
        'enable_function_calling': enable_function_calling
    }


def render_audio_settings():
    lang = _get_lang()
    """Render phần cài đặt âm thanh."""
    st.subheader(ts("tab_audio", lang))
    
    col1, col2 = st.columns(2)
    
    with col1:
        enable_tts = st.checkbox(
            ts("enable_tts", lang),
            value=st.session_state.get('enable_tts', settings.enable_tts),
            help=ts("enable_tts", lang)
        )
        
        tts_voice = st.selectbox(
            ts("tts_voice", lang),
            options=['alloy', 'echo', 'fable', 'onyx', 'nova', 'shimmer'],
            index=['alloy', 'echo', 'fable', 'onyx', 'nova', 'shimmer'].index(settings.tts_voice) if settings.tts_voice in ['alloy', 'echo', 'fable', 'onyx', 'nova', 'shimmer'] else 0,
            help=ts("tts_voice", lang)
        )
        
        audio_sample_rate = st.selectbox(
            ts("audio_sample_rate", lang),
            options=[8000, 16000, 22050, 44100],
            index=[8000, 16000, 22050, 44100].index(settings.audio_sample_rate) if settings.audio_sample_rate in [8000, 16000, 22050, 44100] else 1,
            help=ts("audio_sample_rate", lang)
        )
    
    with col2:
        noise_reduction = st.slider(
            ts("noise_reduction", lang),
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.get('noise_reduction', settings.noise_reduction_strength),
            step=0.1,
            help=ts("noise_reduction", lang)
        )
        
        enable_vocal_separation = st.checkbox(
            ts("enable_vocal_separation", lang),
            value=st.session_state.get('enable_vocal_separation', settings.enable_vocal_separation),
            help=ts("enable_vocal_separation", lang)
        )
    
    return {
        'enable_tts': enable_tts,
        'tts_voice': tts_voice,
        'audio_sample_rate': audio_sample_rate,
        'noise_reduction': noise_reduction,
        'enable_vocal_separation': enable_vocal_separation
    }


def render_memory_settings():
    lang = _get_lang()
    """Render phần cài đặt bộ nhớ."""
    st.subheader(ts("tab_memory", lang))
    
    col1, col2 = st.columns(2)
    
    with col1:
        enable_memory = st.checkbox(
            ts("enable_memory", lang),
            value=st.session_state.get('enable_memory', True),
            help=ts("enable_memory", lang)
        )
        
        max_memory_context = st.slider(
            ts("max_memory_context", lang),
            min_value=1,
            max_value=20,
            value=st.session_state.get('max_memory_context', 3),
            help=ts("max_memory_context", lang)
        )
        
        memory_consolidation_threshold = st.number_input(
            ts("memory_consolidation_threshold", lang),
            min_value=5,
            max_value=50,
            value=st.session_state.get('memory_consolidation_threshold', 10),
            help=ts("memory_consolidation_threshold", lang)
        )
    
    with col2:
        store_conversations = st.checkbox(
            ts("store_conversations", lang),
            value=st.session_state.get('store_conversations', True),
            help=ts("store_conversations", lang)
        )
        
        memory_retention_days = st.number_input(
            ts("memory_retention_days", lang),
            min_value=1,
            max_value=365,
            value=st.session_state.get('memory_retention_days', 30),
            help=ts("memory_retention_days", lang)
        )
        
        auto_cleanup = st.checkbox(
            ts("auto_cleanup", lang),
            value=st.session_state.get('auto_cleanup', True),
            help=ts("auto_cleanup", lang)
        )
    
    return {
        'enable_memory': enable_memory,
        'max_memory_context': max_memory_context,
        'memory_consolidation_threshold': memory_consolidation_threshold,
        'store_conversations': store_conversations,
        'memory_retention_days': memory_retention_days,
        'auto_cleanup': auto_cleanup
    }


def render_interface_settings():
    lang = _get_lang()
    st.subheader(ts("tab_interface", lang))
    
    col1, col2 = st.columns(2)
    
    with col1:
        current_lang = st.session_state.get('language', settings.default_language)
        language = st.selectbox(
            ts("language", lang),
            options=["vi", "en", "zh", "ja", "ko"],
            index=["vi", "en", "zh", "ja", "ko"].index(current_lang) if current_lang in ["vi", "en", "zh", "ja", "ko"] else 0,
            help=ts("language", lang),
            key='language'
        )
        
        auto_save = st.checkbox(
            ts("auto_save", lang),
            value=st.session_state.get('auto_save', settings.auto_save),
            help=ts("auto_save", lang)
        )
    
    with col2:
        # Removed: show_processing_time, show_confidence_score, enable_animations
        pass
    
    return {
        'language': language,
        'auto_save': auto_save,
    }


def render_advanced_settings():
    lang = _get_lang()
    """Render phần cài đặt nâng cao."""
    st.subheader(ts("tab_advanced", lang))
    
    col1, col2 = st.columns(2)
    
    with col1:
        max_file_size_mb = st.number_input(
            ts("max_file_size", lang),
            min_value=10,
            max_value=1000,
            value=st.session_state.get('max_file_size_mb', settings.max_file_size_mb),
            step=10,
            help=ts("max_file_size", lang)
        )
        
        enable_debug_mode = st.checkbox(
            ts("enable_debug_mode", lang),
            value=st.session_state.get('enable_debug_mode', settings.debug),
            help=ts("enable_debug_mode", lang)
        )
        
        cache_enabled = st.checkbox(
            ts("enable_caching", lang),
            value=st.session_state.get('cache_enabled', settings.cache_enabled),
            help=ts("enable_caching", lang)
        )
    
    with col2:
        log_level = st.selectbox(
            ts("log_level", lang),
            options=["DEBUG", "INFO", "WARNING", "ERROR"],
            index=["DEBUG", "INFO", "WARNING", "ERROR"].index(settings.log_level) if settings.log_level in ["DEBUG", "INFO", "WARNING", "ERROR"] else 1,
            help=ts("log_level", lang)
        )
        
        enable_metrics = st.checkbox(
            ts("enable_metrics", lang),
            value=st.session_state.get('enable_metrics', settings.enable_metrics),
            help=ts("enable_metrics", lang)
        )
        
        backup_enabled = st.checkbox(
            ts("backup_enabled", lang),
            value=st.session_state.get('backup_enabled', settings.backup_enabled),
            help=ts("backup_enabled", lang)
        )
    
    return {
        'max_file_size_mb': max_file_size_mb,
        'enable_debug_mode': enable_debug_mode,
        'cache_enabled': cache_enabled,
        'log_level': log_level,
        'enable_metrics': enable_metrics,
        'backup_enabled': backup_enabled
    }


def main():
    """Main function for Settings page."""
    st.set_page_config(page_title="Settings - Thunderbolts", page_icon="⚙️", layout="wide")
    
    # Header
    lang = st.session_state.get('language', settings.default_language)
    st.title(ts("settings_title", lang))
    st.markdown(ts("settings_subtitle", lang))
    st.markdown("---")
    
    # Load current settings
    current_settings = get_settings()
    
    # Initialize session state with current settings
    for key, value in current_settings.items():
        if key not in st.session_state:
            st.session_state[key] = value
    
    # Create tabs for different setting categories (Interface, Model, Search, Audio, Memory, Advanced)
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        ts("tab_interface", lang), ts("tab_model", lang), ts("tab_search", lang), ts("tab_audio", lang), ts("tab_memory", lang), ts("tab_advanced", lang)
    ])
    
    all_settings = {}
    
    with tab1:
        interface_settings = render_interface_settings()
        all_settings.update(interface_settings)
    
    with tab2:
        model_settings = render_model_settings()
        all_settings.update(model_settings)
    
    with tab3:
        search_settings = render_search_settings()
        all_settings.update(search_settings)
    
    with tab4:
        audio_settings = render_audio_settings()
        all_settings.update(audio_settings)
    
    with tab5:
        memory_settings = render_memory_settings()
        all_settings.update(memory_settings)
    
    with tab6:
        advanced_settings = render_advanced_settings()
        all_settings.update(advanced_settings)
    
    # Action buttons
    st.markdown("---")
    
    # Import/Export section
    with st.expander(ts("import_export_title", lang), expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"**{ts('import_settings_section', lang)}:**")
            uploaded_file = st.file_uploader(
                ts("choose_settings_file", lang),
                type=['json'],
                help=ts("upload_settings_help", lang)
            )
            
            if uploaded_file is not None:
                try:
                    settings_content = uploaded_file.read().decode('utf-8')
                    if st.button(ts("btn_import_settings", lang)):
                        if settings_manager.import_settings(settings_content):
                            st.success(ts("import_success", lang))
                            st.rerun()
                        else:
                            st.error(ts("import_failed", lang))
                except Exception as e:
                    st.error(f"{ts('file_read_error', lang)}: {e}")
        
        with col2:
            st.markdown(f"**{ts('export_settings_section', lang)}:**")
            settings_json = settings_manager.export_settings()
            st.download_button(
                label=ts("btn_download_current_settings", lang),
                data=settings_json,
                file_name="Thunderbolts_settings.json",
                mime="application/json",
                use_container_width=True
            )
    
    # Main action buttons
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button(ts("save_apply", lang), use_container_width=True, type="primary"):
            validate_and_save_settings(all_settings)
    
    with col2:
        if st.button(ts("reset_defaults", lang), use_container_width=True):
            if st.button(ts("confirm_reset", lang), key="confirm_reset"):
                settings_manager.reset_to_defaults()
                st.success("✅ Settings reset to defaults!")
                st.rerun()
    
    with col3:
        if st.button(ts("settings_summary", lang), use_container_width=True):
            summary = settings_manager.get_settings_summary()
            st.json(summary)

if __name__ == "__main__":
    main()
