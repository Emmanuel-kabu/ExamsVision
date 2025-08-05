from supabase import create_client, Client
import streamlit as st
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# Google OAuth credentials (if needed)
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")

# Initialize Supabase client
SITE_URL = os.getenv("SITE_URL", "http://localhost:8501")
supabase: Client = create_client(
    supabase_url=SUPABASE_URL,
    supabase_key=SUPABASE_KEY
)

# Set redirect URL in Supabase project settings instead of code
# Dashboard -> Authentication -> URL Configuration
# Site URL: http://localhost:8501
# Redirect URLs: http://localhost:8501/auth/callback

# sign-in function
def sign_in_with_email(email: str, password: str):
    """Sign in to Supabase with email and password."""
    try:
        auth_response = supabase.auth.sign_in_with_password({
            'email': email,
            'password': password
        })
        
        user = auth_response.user
        if user:
            # Check email verification using user_metadata
            if not user.user_metadata.get('email_verified', False):
                st.error("Please verify your email address before signing in. Check your inbox for the verification link.")
                return None
                
            st.session_state.authentication_status = True
            st.session_state.name = email.split('@')[0]
            st.session_state.user = {
                'email': user.email,
                'id': user.id,
                'role': user.role
            }
            st.success("Successfully signed in!")
            return user
            
        return None
    except Exception as e:
        st.error(f"Error signing in: {str(e)}")
        return None

# sign-up function
def sign_up_with_email(email: str, password: str):
    """Sign up to Supabase with email and password."""
    try:
        auth_response = supabase.auth.sign_up({
            'email': email,
            'password': password
        })
        user = auth_response.user
        if user:
            st.success("Sign up successful! Please check your email for verification.")
            st.info("You will need to verify your email before you can sign in.")
            # Don't set authentication status - require email verification first
            return user
        return None
    except Exception as e:
        if "User already registered" in str(e):
            st.error("This email is already registered. Please sign in instead.")
        else:
            st.error(f"Error signing up: {str(e)}")
        return None
    


# forget password function
def forget_password(email: str):
    """Send a password reset email."""
    try:
        supabase.auth.reset_password_email_request(email)
        st.success("Password reset email sent successfully.")
    except Exception as e:
        st.error(f"Error sending reset email: {str(e)}")

# Authentication screen
@st.cache_data
def auth_screen():
    """Display authentication screen with sign-in and sign-up forms."""
    # Create three columns with the middle one containing the auth form
    left_col, center_col, right_col = st.columns([1, 2, 1])
    
    with center_col:
        # Add some vertical spacing
        st.markdown("<br><br>", unsafe_allow_html=True)
        
        # Create a container for the auth form
        with st.container():
            # Title with custom styling
            st.markdown(
                """
                <h1 style='text-align: center; color: #1f77b4;'>
                    EXAM VISIO PRO
                </h1>
                """, 
                unsafe_allow_html=True
            )
            
            # Create tabs for Sign In and Sign Up
            tab1, tab2 = st.tabs(["Sign In", "Sign Up"])
            
            with tab1:
                # Sign In Form
                st.markdown("<br>", unsafe_allow_html=True)
                email_login = st.text_input("Email", key="email_login")
                password_login = st.text_input("Password", type="password", key="password_login")
                
                # Center the sign in button
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    if st.button("Sign In", use_container_width=True):
                        user = sign_in_with_email(email_login, password_login)
                        if user:
                            return True
                
                # Forgot password link
                st.markdown("<br>", unsafe_allow_html=True)
                email_forgot = st.text_input("Email for password reset", key="email_forgot")
                if st.button("Forgot Password?", use_container_width=True):
                    forget_password(email_forgot)
            
            with tab2:
                # Sign Up Form
                st.markdown("<br>", unsafe_allow_html=True)
                email_signup = st.text_input("Email", key="email_signup")
                password_signup = st.text_input("Password", type="password", key="password_signup")
                
                # Center the sign up button
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    if st.button("Sign Up", use_container_width=True):
                        user = sign_up_with_email(email_signup, password_signup)
                        if user:
                            return True
        
        # Add some spacing at the bottom
        st.markdown("<br><br>", unsafe_allow_html=True)
    
    return False
# sign-out function



def sign_out():
    """Sign out from Supabase."""
    try:
        supabase.auth.sign_out()
        st.success("Signed out successfully.")
    except Exception as e:
        st.error(f"Error signing out: {e}")      