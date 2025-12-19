import streamlit as st

# Page config
st.set_page_config(
    page_title="SuperJamey Portfolio",
    page_icon="🚀",
    layout="wide"
)

# Header
st.title("🚀 Welcome to My Interactive Portfolio")
st.markdown("### Exploring Data, AI, and Python")

# Introduction
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("""
    Hi! I'm Jamey, a developer passionate about creating interactive applications 
    and exploring data through code. This Streamlit app showcases various projects 
    and demonstrations of my work with Python.
    
    **What you'll find here:**
    - 📊 Interactive data visualizations
    - 🤖 Machine learning demonstrations  
    - 🔗 API integration examples
    - 💡 Practical Python tools
    """)
    
with col2:
    st.info("👈 Use the sidebar to navigate between different demos!")

# Quick links
st.markdown("---")
st.subheader("🔗 Quick Links")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("[![GitHub](https://img.shields.io/badge/GitHub-Profile-blue?style=for-the-badge&logo=github)](https://github.com/SuperJamey)")
    
with col2:
    st.markdown("[![Website](https://img.shields.io/badge/Website-superjamey.com-green?style=for-the-badge)](https://superjamey.com)")
    
with col3:
    st.markdown("[![X](https://img.shields.io/badge/X-Follow-black?style=for-the-badge&logo=x)](https://x.com/_JLaMar)")
    
with col4:
    st.markdown("[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=for-the-badge&logo=linkedin)](https://linkedin.com/in/jameylamar4805)")

# Featured Projects
st.markdown("---")
st.subheader("🌟 Featured Projects")

proj_col1, proj_col2 = st.columns(2)

with proj_col1:
    with st.container():
        st.markdown("#### 🤖 AI-Powered X Bot")
        st.markdown("""
        Automated bot that generates and posts daily Python tips using Google's 
        Gemini AI via LangChain.
        
        **Tech Stack:** Python, LangChain, Google Gemini, Tweepy
        """)
        st.link_button("View on GitHub", "https://github.com/SuperJamey/ai-posting-on-x")

with proj_col2:
    with st.container():
        st.markdown("#### 🌐 Personal Website")
        st.markdown("""
        Modern personal portfolio showcasing my work in development, painting, 
        and outdoor adventures.
        
        **Tech Stack:** HTML, CSS, JavaScript, Netlify
        """)
        st.link_button("Visit Site", "https://superjamey.com")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p>Built with ❤️ using Streamlit | © 2025 SuperJamey</p>
</div>
""", unsafe_allow_html=True)