import streamlit as st
from database_operations import DatabaseOperations, init_supabase
from datetime import datetime
import json
import time
from typing import Dict, Any

class NotificationManager:
    """Manages real-time notifications and alerts"""
    
    def __init__(self, db_manager=None):
        # Initialize notification state
        if 'notifications' not in st.session_state:
            st.session_state.notifications = []
        if 'last_notification_check' not in st.session_state:
            st.session_state.last_notification_check = time.time()
        
        # Auto-create DatabaseOperations if not provided
        self.db_manager = db_manager or DatabaseOperations(init_supabase())
            
    def add_notification(self, message: str, level: str = "info", data: Dict[str, Any] = None):
        """Add a new notification"""
        notification = {
            'id': len(st.session_state.notifications) + 1,
            'message': message,
            'level': level,
            'timestamp': datetime.now().isoformat(),
            'read': False,
            'data': data or {}
        }
        st.session_state.notifications.insert(0, notification)
        
        # Store in database
        if self.db_manager:
            try:
                db_notification = {
                    'notification_type': level,
                    'message': message,
                    'status': 'unread',
                    'data': json.dumps(data) if data else None
                }
                if data and 'exam_id' in data:
                    db_notification['exam_id'] = data['exam_id']
                self.db_manager.create_notification(db_notification)
            except Exception as e:
                st.warning(f"Failed to save notification to database: {e}")
        
    def mark_as_read(self, notification_id: int):
        """Mark a notification as read"""
        for notif in st.session_state.notifications:
            if notif['id'] == notification_id:
                notif['read'] = True
                # Optional: also mark in DB if implemented
                break
                
    def get_unread_count(self) -> int:
        """Get count of unread notifications"""
        return len([n for n in st.session_state.notifications if not n['read']])
    
    def render_notification_center(self):
        """Render the notification center UI"""
        unread = self.get_unread_count()
        
        with st.sidebar:
            st.markdown("---")
            st.markdown(f"### 🔔 Notifications ({unread})")
            
            if st.session_state.notifications:
                for notif in st.session_state.notifications[:5]:
                    with st.container():
                        if notif['level'] == 'error':
                            st.error(notif['message'])
                        elif notif['level'] == 'warning':
                            st.warning(notif['message'])
                        elif notif['level'] == 'success':
                            st.success(notif['message'])
                        else:
                            st.info(notif['message'])
                        
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.caption(f"🕒 {datetime.fromisoformat(notif['timestamp']).strftime('%H:%M:%S')}")
                        with col2:
                            if not notif['read']:
                                if st.button("✓", key=f"read_{notif['id']}"):
                                    self.mark_as_read(notif['id'])
                                    st.rerun()
            else:
                st.info("No new notifications")
