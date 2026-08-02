import streamlit as st

st.set_page_config(
    page_title="Mera AI Prediction Dashboard",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Mera AI Prediction Dashboard")

st.markdown("""
## Welcome

Choose an application from the sidebar.

### Available Applications

🏠 House Price Prediction

📧 SMS & Email Spam Detection
""")

st.info("Select an application from the left sidebar.")
# Footer
st.divider()
st.caption("Powered by Mera")
