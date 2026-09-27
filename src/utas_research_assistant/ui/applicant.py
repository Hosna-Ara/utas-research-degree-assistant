"""Compact, session-owned sidebar CV controls with explicit save semantics."""
import streamlit as st
from utas_research_assistant.applicant import extract_upload, extract_profile
from utas_research_assistant.deployment import deployment_mode


def clear_cv():
    version = st.session_state.get('cv_widget_version', 0) + 1
    for key in list(st.session_state):
        if key.startswith(('profile_edit_', 'cv_upload_')) or key in {
            'applicant_profile', 'cv_upload', 'cv_error', 'cv_use_qwen', 'cv_saved_filename',
        }:
            del st.session_state[key]
    st.session_state.cv_widget_version = version
    st.session_state.messages = []
    st.session_state.pop('match_context', None)
    st.session_state.pending_question = None
    st.session_state.conversation_id = None
    # Generic matching resumes, but the privacy latch continues to prevent
    # CV-derived follow-up text entering persistent history in this session.


def stage_upload():
    """Selecting a file does not extract or activate a profile."""
    st.session_state.session_private = True
    st.session_state.pop('cv_error', None)


def process_upload(upload_key='cv_upload'):
    """Save the selected CV in session memory; called only by Save CV."""
    upload = st.session_state.get(upload_key)
    if upload is None:
        return
    st.session_state.session_private = True
    st.session_state.pop('applicant_profile', None)
    st.session_state.pop('cv_saved_filename', None)
    st.session_state.pop('match_context', None)
    st.session_state.messages = []
    st.session_state.conversation_id = None
    st.session_state.pending_question = None
    st.session_state.pop('cv_error', None)
    try:
        with st.spinner('Preparing your session profile…'):
            text = extract_upload(upload.name, upload.getvalue())
            profile = extract_profile(text, use_qwen=st.session_state.get('cv_use_qwen', False), public=deployment_mode() == 'public')
        st.session_state.applicant_profile = profile
        st.session_state.cv_saved_filename = upload.name
    except ValueError as exc:
        st.session_state.cv_error = str(exc)
    # Extracted text is only a local variable. No file/history/cache writes.


def _queue_profile_action(question):
    st.session_state.view = 'chat'
    st.session_state.pending_question = question


def render_cv():
    """Render inside the caller's sidebar Workspace section."""
    with st.expander('CV / Profile', expanded=bool(st.session_state.get('applicant_profile'))):
        st.caption('PDF / DOCX · up to 5 MB. Save CV activates matching for this session only. Your CV is not stored permanently.')
        upload_key = f'cv_upload_{st.session_state.get("cv_widget_version", 0)}'
        upload = st.file_uploader('Upload CV / resume', type=['pdf', 'docx'], key=upload_key, on_change=stage_upload)
        if upload is not None:
            st.text('Selected: ' + upload.name)
            st.caption('Click Save CV to use this file. Any previously saved profile stays active until then.')
        if deployment_mode() != 'public':
            st.checkbox('Use local Qwen (optional)', key='cv_use_qwen')
        st.button('Save CV', key='save_cv', disabled=upload is None, type='primary', use_container_width=True,
                  on_click=process_upload, args=(upload_key,))
        if st.session_state.get('cv_error'):
            st.warning(st.session_state.cv_error)
        profile = st.session_state.get('applicant_profile')
        if profile:
            st.success('CV loaded · Profile extracted for this session')
            if st.session_state.get('cv_saved_filename'):
                st.text('Saved: ' + st.session_state.cv_saved_filename)
            st.caption('Personalised project and supervisor matching is ready.')
            context = st.session_state.get('match_context')
            gap_question = ('What gaps do I have for this project?' if context and context.active_project
                            else 'Find 3 projects aligned with my CV and explain potential gaps')
            actions = [
                ('Find projects according to my profile', 'Find the 3 UTAS research projects most aligned with my CV', 'match_cv'),
                ('Find supervisors according to my profile', 'Which supervisors are most aligned with my CV?', 'match_cv_supervisors'),
                ('What are my research gaps?', gap_question, 'match_cv_gaps'),
                ('Which UTAS project matches me best?', 'Which UTAS project is most suitable for my profile?', 'match_cv_best'),
            ]
            for label, question, key in actions:
                st.button(label, key=key, use_container_width=True, on_click=_queue_profile_action, args=(question,))
        if profile or upload is not None:
            st.button('Remove CV', on_click=clear_cv, key='clear_cv', use_container_width=True)
