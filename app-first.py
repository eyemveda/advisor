import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pinecone import Pinecone

st.set_page_config(page_title="Bachatt AI Advisor", layout="wide")

GEMINI_KEY = st.secrets["GEMINI_API_KEY"] if "GEMINI_API_KEY" in st.secrets else os.environ.get("GEMINI_API_KEY")
PINECONE_KEY = st.secrets["PINECONE_API_KEY"] if "PINECONE_API_KEY" in st.secrets else os.environ.get("PINECONE_API_KEY")

@st.cache_data
def get_user_data():
    start_date = datetime.now() - timedelta(days=30)
    dates = [start_date + timedelta(days=i) for i in range(30)]
    np.random.seed(42)
    daily_savings = [np.random.randint(51, 200) if np.random.rand() > 0.2 else 0 for _ in range(30)]
    df = pd.DataFrame({
        "Date": [d.strftime("%Y-%m-%d") for d in dates],
        "Amount Saved (₹)": daily_savings
    })
    profile = {
        "name": "Ramesh Kumar",
        "occupation": "Kirana Store Owner",
        "goal": "Daughter's Higher Education (₹50k target)",
        "fund": "Nippon India Small Cap Fund"
    }
    return df, profile

df, profile = get_user_data()

st.sidebar.title("💰 Bachatt Gullak")
st.sidebar.subheader(f"Namaste, {profile['name']}!")
st.sidebar.caption(f"💼 Business: {profile['occupation']}")
st.sidebar.markdown(f"🎯 **Goal:** {profile['goal']}")
st.sidebar.markdown(f"📈 **Active Scheme:** {profile['fund']}")
st.sidebar.divider()

total_saved = int(df["Amount Saved (₹)"].sum())
avg_saved = float(df["Amount Saved (₹)"].mean())

st.sidebar.metric(label="Total Saved (30 Days)", value=f"₹{total_saved:,}")
st.sidebar.metric(label="Avg Daily Saving", value=f"₹{avg_saved:.2f}")
st.sidebar.subheader("Daily Micro-SIP Patterns")
st.sidebar.line_chart(df.set_index("Date"))

st.title("🤖 Bachatt Mitra — Your Daily Micro-Savings Guide")
st.write("I analyze your daily sales patterns to keep your financial goals on track smoothly.")

if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant",
        "content": f"Namaste Ramesh Ji! You saved ₹{total_saved} this month. Ready to top up today?"
    }]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask Bachatt Mitra (e.g., 'Am I on track for my goal?')"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Looking up financial guidelines..."):
            try:
                pc = Pinecone(api_key=PINECONE_KEY)
                embeddings = GoogleGenerativeAIEmbeddings(
                    model="models/gemini-embedding-001",
                    google_api_key=GEMINI_KEY
                )
                index = pc.Index("bachatt-kb")
                query_vector = embeddings.embed_query(prompt)
                search_results = index.query(vector=query_vector, top_k=1, include_metadata=True)
                kb_context = search_results['matches'][0]['metadata']['text'] if search_results['matches'] else "Maintain high liquidity for micro-businesses."

                genai.configure(api_key=GEMINI_KEY)
                model = genai.GenerativeModel(model_name="models/gemini-2.5-flash")
                instructions = f"""
                Role: Friendly Indian Financial Advisor 'Bachatt Mitra'
                Customer: {profile['name']}, {profile['occupation']}
                Metrics: Saved ₹{total_saved} this month, Avg ₹{avg_saved:.2f}/day.
                Reference: {kb_context}
                Question: {prompt}
                Reply warmly in 2-3 simple, actionable sentences. No jargon.
                """
                response = model.generate_content(instructions)
                output_text = response.text
            except Exception as e:
                output_text = f"Apologies Ramesh Ji, facing a small issue! (Error: {str(e)[:60]}...)"

            st.markdown(output_text)
            st.session_state.messages.append({"role": "assistant", "content": output_text})
