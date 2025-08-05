import streamlit as st
from datetime import datetime
import json
import os
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
        
        # Store database manager reference
        self.db_manager = db_manager
            
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
        
        # If database manager is available, also store in database
        if self.db_manager:
            try:
                db_notification = {
                    'notification_type': level,
                    'message': message,
                    'status': 'unread',
                    'data': json.dumps(data) if data else None
                }
                # Add exam_id if it's in the data
                if data and 'exam_id' in data:
                    db_notification['exam_id'] = data['exam_id']
                self.db_manager.create_notification(db_notification)
            except Exception as e:
                st.warning(f"Failed to save notification to database: {e}")
        
    def mark_as_read(self, notification_id: int):
        """Mark a notification as read"""
        # Update local state
        for notif in st.session_state.notifications:
            if notif['id'] == notification_id:
                notif['read'] = True
                # If database manager available, update in database too
                if self.db_manager:
                    try:
                        self.db_manager.mark_notification_as_read(str(notification_id))
                    except Exception as e:
                        st.warning(f"Failed to mark notification as read in database: {e}")
                break
                
    def get_unread_count(self) -> int:
        """Get count of unread notifications"""
        # First check local notifications
        local_count = len([n for n in st.session_state.notifications if not n['read']])
        
        # If database manager is available, also check database
        if self.db_manager:
            try:
                db_notifications = self.db_manager.get_unread_notifications()
                return max(local_count, len(db_notifications))
            except Exception as e:
                st.warning(f"Failed to get unread notifications from database: {e}")
        
        return local_count
    
    def render_notification_center(self):
        """Render the notification center UI"""
        unread = self.get_unread_count()
        
        with st.sidebar:
            st.markdown("---")
            st.markdown(f"### 🔔 Notifications ({unread})")
            
            # Update from database if available
            if self.db_manager:
                try:
                    db_notifications = self.db_manager.get_unread_notifications()
                    for db_notif in db_notifications:
                        # Add to local state if not already present
                        if not any(n['id'] == db_notif['id'] for n in st.session_state.notifications):
                            notification = {
                                'id': db_notif['id'],
                                'message': db_notif['message'],
                                'level': db_notif['notification_type'],
                                'timestamp': db_notif['created_at'],
                                'read': db_notif['status'] == 'read',
                                'data': json.loads(db_notif['data']) if db_notif['data'] else {}
                            }
                            st.session_state.notifications.insert(0, notification)
                except Exception as e:
                    st.warning(f"Failed to fetch notifications from database: {e}")
            
            if st.session_state.notifications:
                for notif in st.session_state.notifications[:5]:  # Show last 5 notifications
                    with st.container():
                        # Style based on notification level
                        if notif['level'] == 'error':
                            st.error(notif['message'])
                        elif notif['level'] == 'warning':
                            st.warning(notif['message'])
                        elif notif['level'] == 'success':
                            st.success(notif['message'])
                        else:
                            st.info(notif['message'])
                        
                        # Show timestamp and mark as read button
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.caption(f"🕒 {datetime.fromisoformat(notif['timestamp']).strftime('%H:%M:%S')}")
                        with col2:
                            if not notif['read']:
                                if st.button("✓", key=f"read_{notif['id']}"):
                                    self.mark_as_read(notif['id'])
                                    st.rerun()
