from __future__ import annotations

import streamlit as st
import json
import os
from pathlib import Path
from typing import Dict, Any
import toml

from src.interface.components import UIComponents
from src.utils.settings_manager import settings_manager, get_settings, set_setting
from config.settings import settings


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
    return True


def render_model_settings():
    """Render phần cài đặt model."""
    st.subheader("🤖 Model Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=2.0,
            value=st.session_state.get('openai_temperature', settings.openai_temperature),
            step=0.1,
            help="Điều khiển độ ngẫu nhiên trong câu trả lời. Giá trị cao hơn = sáng tạo hơn"
        )
        
        max_tokens = st.number_input(
            "Max Tokens",
            min_value=100,
            max_value=8000,
            value=st.session_state.get('openai_max_tokens', settings.openai_max_tokens),
            step=100,
            help="Số lượng token tối đa trong câu trả lời"
        )
        
        top_p = st.slider(
            "Top P",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.get('openai_top_p', settings.openai_top_p),
            step=0.1,
            help="Điều khiển đa dạng của câu trả lời"
        )
    
    with col2:
        frequency_penalty = st.slider(
            "Frequency Penalty",
            min_value=-2.0,
            max_value=2.0,
            value=st.session_state.get('openai_frequency_penalty', settings.openai_frequency_penalty),
            step=0.1,
            help="Giảm lặp lại từ ngữ"
        )
        
        presence_penalty = st.slider(
            "Presence Penalty",
            min_value=-2.0,
            max_value=2.0,
            value=st.session_state.get('openai_presence_penalty', settings.openai_presence_penalty),
            step=0.1,
            help="Khuyến khích thảo luận về chủ đề mới"
        )
        
        model_name = st.selectbox(
            "Model",
            options=["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo", "claude-3-opus", "claude-3-sonnet"],
            index=0,
            help="Chọn model AI để sử dụng"
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
    """Render phần cài đặt tìm kiếm."""
    st.subheader("🔍 Search Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        similarity_threshold = st.slider(
            "Similarity Threshold",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.get('similarity_threshold', settings.similarity_threshold),
            step=0.05,
            help="Ngưỡng tương đồng tối thiểu cho kết quả tìm kiếm"
        )
        
        max_results = st.number_input(
            "Max Results",
            min_value=1,
            max_value=50,
            value=st.session_state.get('max_results', settings.max_results),
            step=1,
            help="Số lượng kết quả tìm kiếm tối đa"
        )
        
        chunk_size = st.number_input(
            "Chunk Size",
            min_value=100,
            max_value=2000,
            value=st.session_state.get('chunk_size', settings.chunk_size),
            step=100,
            help="Kích thước chunk khi xử lý tài liệu"
        )
    
    with col2:
        chunk_overlap = st.number_input(
            "Chunk Overlap",
            min_value=0,
            max_value=500,
            value=st.session_state.get('chunk_overlap', settings.chunk_overlap),
            step=50,
            help="Độ chồng lấp giữa các chunk"
        )
        
        enable_web_search = st.checkbox(
            "Enable Web Search",
            value=st.session_state.get('enable_web_search', settings.enable_web_search),
            help="Tìm kiếm web khi kết quả local không đủ"
        )
        
        enable_function_calling = st.checkbox(
            "Enable Function Calling",
            value=st.session_state.get('enable_function_calling', settings.enable_function_calling),
            help="Sử dụng function calling để phân tích nâng cao"
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
    """Render phần cài đặt âm thanh."""
    st.subheader("🔊 Audio Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        enable_tts = st.checkbox(
            "Enable Text-to-Speech",
            value=st.session_state.get('enable_tts', settings.enable_tts),
            help="Tạo âm thanh cho tóm tắt"
        )
        
        tts_voice = st.selectbox(
            "TTS Voice",
            options=['alloy', 'echo', 'fable', 'onyx', 'nova', 'shimmer'],
            index=['alloy', 'echo', 'fable', 'onyx', 'nova', 'shimmer'].index(settings.tts_voice) if settings.tts_voice in ['alloy', 'echo', 'fable', 'onyx', 'nova', 'shimmer'] else 0,
            help="Giọng nói cho text-to-speech"
        )
        
        audio_sample_rate = st.selectbox(
            "Audio Sample Rate",
            options=[8000, 16000, 22050, 44100],
            index=[8000, 16000, 22050, 44100].index(settings.audio_sample_rate) if settings.audio_sample_rate in [8000, 16000, 22050, 44100] else 1,
            help="Tần số lấy mẫu âm thanh"
        )
    
    with col2:
        noise_reduction = st.slider(
            "Noise Reduction",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.get('noise_reduction', settings.noise_reduction_strength),
            step=0.1,
            help="Độ mạnh giảm tiếng ồn"
        )
        
        enable_vocal_separation = st.checkbox(
            "Enable Vocal Separation",
            value=st.session_state.get('enable_vocal_separation', settings.enable_vocal_separation),
            help="Tách giọng nói khỏi nhạc nền"
        )
    
    return {
        'enable_tts': enable_tts,
        'tts_voice': tts_voice,
        'audio_sample_rate': audio_sample_rate,
        'noise_reduction': noise_reduction,
        'enable_vocal_separation': enable_vocal_separation
    }


def render_memory_settings():
    """Render phần cài đặt bộ nhớ."""
    st.subheader("🧠 Memory Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        enable_memory = st.checkbox(
            "Enable Memory System",
            value=st.session_state.get('enable_memory', True),
            help="Ghi nhớ ngữ cảnh và thông tin cuộc trò chuyện"
        )
        
        max_memory_context = st.slider(
            "Max Memory Context",
            min_value=1,
            max_value=20,
            value=st.session_state.get('max_memory_context', 3),
            help="Số lượng bản ghi bộ nhớ tối đa sử dụng cho ngữ cảnh"
        )
        
        memory_consolidation_threshold = st.number_input(
            "Memory Consolidation Threshold",
            min_value=5,
            max_value=50,
            value=st.session_state.get('memory_consolidation_threshold', 10),
            help="Ngưỡng để củng cố bộ nhớ ngắn hạn thành dài hạn"
        )
    
    with col2:
        store_conversations = st.checkbox(
            "Store Conversations",
            value=st.session_state.get('store_conversations', True),
            help="Lưu lịch sử cuộc trò chuyện"
        )
        
        memory_retention_days = st.number_input(
            "Memory Retention (days)",
            min_value=1,
            max_value=365,
            value=st.session_state.get('memory_retention_days', 30),
            help="Số ngày lưu trữ bộ nhớ"
        )
        
        auto_cleanup = st.checkbox(
            "Auto Cleanup Old Memories",
            value=st.session_state.get('auto_cleanup', True),
            help="Tự động dọn dẹp bộ nhớ cũ"
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
    """Render phần cài đặt giao diện."""
    st.subheader("🎨 Interface Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        theme = st.selectbox(
            "Theme",
            options=["light", "dark"],
            index=["light", "dark"].index(settings.theme) if settings.theme in ["light", "dark"] else 0,
            help="Chọn giao diện sáng hoặc tối"
        )
        
        language = st.selectbox(
            "Language",
            options=["vi", "en", "zh", "ja", "ko"],
            index=["vi", "en", "zh", "ja", "ko"].index(settings.default_language) if settings.default_language in ["vi", "en", "zh", "ja", "ko"] else 0,
            help="Ngôn ngữ giao diện"
        )
        
        auto_save = st.checkbox(
            "Auto Save Settings",
            value=st.session_state.get('auto_save', settings.auto_save),
            help="Tự động lưu cài đặt"
        )
    
    with col2:
        show_processing_time = st.checkbox(
            "Show Processing Time",
            value=st.session_state.get('show_processing_time', settings.show_processing_time),
            help="Hiển thị thời gian xử lý"
        )
        
        show_confidence_score = st.checkbox(
            "Show Confidence Score",
            value=st.session_state.get('show_confidence_score', settings.show_confidence_score),
            help="Hiển thị điểm tin cậy"
        )
        
        enable_animations = st.checkbox(
            "Enable Animations",
            value=st.session_state.get('enable_animations', settings.enable_animations),
            help="Bật hiệu ứng animation"
        )
    
    return {
        'theme': theme,
        'language': language,
        'auto_save': auto_save,
        'show_processing_time': show_processing_time,
        'show_confidence_score': show_confidence_score,
        'enable_animations': enable_animations
    }


def render_advanced_settings():
    """Render phần cài đặt nâng cao."""
    st.subheader("⚙️ Advanced Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        max_file_size_mb = st.number_input(
            "Max File Size (MB)",
            min_value=10,
            max_value=1000,
            value=st.session_state.get('max_file_size_mb', settings.max_file_size_mb),
            step=10,
            help="Kích thước file tối đa được phép upload"
        )
        
        enable_debug_mode = st.checkbox(
            "Enable Debug Mode",
            value=st.session_state.get('enable_debug_mode', settings.debug),
            help="Bật chế độ debug để xem thông tin chi tiết"
        )
        
        cache_enabled = st.checkbox(
            "Enable Caching",
            value=st.session_state.get('cache_enabled', settings.cache_enabled),
            help="Bật cache để tăng tốc độ"
        )
    
    with col2:
        log_level = st.selectbox(
            "Log Level",
            options=["DEBUG", "INFO", "WARNING", "ERROR"],
            index=["DEBUG", "INFO", "WARNING", "ERROR"].index(settings.log_level) if settings.log_level in ["DEBUG", "INFO", "WARNING", "ERROR"] else 1,
            help="Mức độ log"
        )
        
        enable_metrics = st.checkbox(
            "Enable Metrics Collection",
            value=st.session_state.get('enable_metrics', settings.enable_metrics),
            help="Thu thập metrics để cải thiện hiệu suất"
        )
        
        backup_enabled = st.checkbox(
            "Enable Auto Backup",
            value=st.session_state.get('backup_enabled', settings.backup_enabled),
            help="Tự động sao lưu dữ liệu"
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
    st.set_page_config(
        page_title="Settings - ElevateAI",
        page_icon="⚙️",
        layout="wide"
    )
    
    # Header
    st.title("⚙️ Settings")
    st.markdown("Cấu hình các thông số cho ứng dụng ElevateAI")
    st.markdown("---")
    
    # Load current settings
    current_settings = get_settings()
    
    # Initialize session state with current settings
    for key, value in current_settings.items():
        if key not in st.session_state:
            st.session_state[key] = value
    
    # Create tabs for different setting categories
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🤖 Model", "🔍 Search", "🔊 Audio", "🧠 Memory", "🎨 Interface", "⚙️ Advanced"
    ])
    
    all_settings = {}
    
    with tab1:
        model_settings = render_model_settings()
        all_settings.update(model_settings)
    
    with tab2:
        search_settings = render_search_settings()
        all_settings.update(search_settings)
    
    with tab3:
        audio_settings = render_audio_settings()
        all_settings.update(audio_settings)
    
    with tab4:
        memory_settings = render_memory_settings()
        all_settings.update(memory_settings)
    
    with tab5:
        interface_settings = render_interface_settings()
        all_settings.update(interface_settings)
    
    with tab6:
        advanced_settings = render_advanced_settings()
        all_settings.update(advanced_settings)
    
    # Action buttons
    st.markdown("---")
    
    # Import/Export section
    with st.expander("📁 Import/Export Settings", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Import Settings:**")
            uploaded_file = st.file_uploader(
                "Choose a settings file",
                type=['json'],
                help="Upload a previously exported settings file"
            )
            
            if uploaded_file is not None:
                try:
                    settings_content = uploaded_file.read().decode('utf-8')
                    if st.button("📥 Import Settings"):
                        if settings_manager.import_settings(settings_content):
                            st.success("✅ Settings imported successfully!")
                            st.rerun()
                        else:
                            st.error("❌ Failed to import settings")
                except Exception as e:
                    st.error(f"❌ Error reading file: {e}")
        
        with col2:
            st.markdown("**Export Settings:**")
            settings_json = settings_manager.export_settings()
            st.download_button(
                label="📤 Download Current Settings",
                data=settings_json,
                file_name="elevateai_settings.json",
                mime="application/json",
                use_container_width=True
            )
    
    # Main action buttons
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("💾 Save & Apply", use_container_width=True, type="primary"):
            validate_and_save_settings(all_settings)
    
    with col2:
        if st.button("🔄 Reset to Defaults", use_container_width=True):
            if st.button("⚠️ Confirm Reset", key="confirm_reset"):
                settings_manager.reset_to_defaults()
                st.success("✅ Settings reset to defaults!")
                st.rerun()
    
    with col3:
        if st.button("📊 Settings Summary", use_container_width=True):
            summary = settings_manager.get_settings_summary()
            st.json(summary)
    
    # Test AI Settings Update
    st.markdown("---")
    st.subheader("🧪 Test AI Settings Update")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔍 Check AI Settings Status", use_container_width=True):
            try:
                context = get_context()
                if context.get("llm_client"):
                    settings_check = context["llm_client"].check_settings_applied()
                    
                    if settings_check["settings_applied"]:
                        st.success("✅ All AI settings are properly applied!")
                    else:
                        st.warning("⚠️ Some AI settings are not applied:")
                        for key, mismatch in settings_check["mismatches"].items():
                            st.write(f"• {key}: current={mismatch['current']}, expected={mismatch['expected']}")
                    
                    # Show detailed info
                    with st.expander("📋 Detailed Settings Info"):
                        st.json(settings_check)
                else:
                    st.error("❌ LangchainLLMClient not available")
            except Exception as e:
                st.error(f"❌ Error checking AI settings: {e}")
    
    with col2:
        if st.button("🔄 Test Settings Update", use_container_width=True):
            try:
                context = get_context()
                if context.get("llm_client"):
                    # Test update with a different temperature
                    test_temp = 0.9
                    context["llm_client"].update_config({"temperature": test_temp})
                    
                    # Check if update was applied
                    settings_check = context["llm_client"].check_settings_applied()
                    current_temp = context["llm_client"].config.get("temperature")
                    
                    if current_temp == test_temp:
                        st.success(f"✅ Temperature updated successfully to {test_temp}")
                    else:
                        st.error(f"❌ Temperature update failed. Current: {current_temp}, Expected: {test_temp}")
                    
                    # Show updated config
                    with st.expander("📋 Updated Configuration"):
                        st.json(context["llm_client"].config)
                else:
                    st.error("❌ LangchainLLMClient not available")
            except Exception as e:
                st.error(f"❌ Error testing settings update: {e}")
    
    # Debug Configuration
    st.markdown("---")
    st.subheader("🐛 Debug Configuration")
    
    if st.button("🔍 Debug LangchainLLMClient Config", use_container_width=True):
        try:
            context = get_context()
            if context.get("llm_client"):
                debug_info = context["llm_client"].debug_config()
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Self Config:**")
                    st.json(debug_info['self_config'])
                    
                    st.markdown("**Settings Values:**")
                    st.json(debug_info['settings_values'])
                
                with col2:
                    st.markdown("**Default Config:**")
                    st.json(debug_info['default_config'])
                    
                    st.markdown("**LLM Config:**")
                    st.json(debug_info['llm_config'])
                
                # Show settings source
                st.markdown(f"**Settings Source:** {debug_info.get('settings_source', 'Unknown')}")
                
            else:
                st.error("❌ LangchainLLMClient not available")
        except Exception as e:
            st.error(f"❌ Error debugging config: {e}")
    
    # Settings preview and information
    col1, col2 = st.columns([2, 1])
    
    with col1:
        with st.expander("📋 Current Settings Preview", expanded=False):
            st.json(all_settings)
    
    with col2:
        st.markdown("### ℹ️ Information")
        st.markdown("""
        **Settings Priority:**
        1. File Settings (vĩnh viễn)
        2. Session Settings (tạm thời)
        3. Default Settings
        
        **Quick Tips:**
        - Use "Save & Apply" to apply changes immediately
        - Export settings for backup
        - Import settings to restore configuration
        """)
        
        # Show current settings summary
        st.markdown("### 📊 Current Summary")
        summary = settings_manager.get_settings_summary()
        
        for category, settings in summary.items():
            st.markdown(f"**{category.title()}:**")
            for key, value in settings.items():
                if isinstance(value, bool):
                    display_value = "✅" if value else "❌"
                else:
                    display_value = str(value)
                st.markdown(f"• {key}: {display_value}")
            st.markdown("")
    
    # Footer
    st.markdown("---")
    st.markdown("*Settings will be applied to the entire application. Some changes may require a restart to take full effect.*")


if __name__ == "__main__":
    main()
