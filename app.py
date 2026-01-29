import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import tempfile
import os
import zipfile

# --- CLASSIC IMPORTS (LangChain 0.0.350) ---
# These paths are guaranteed to exist in this version.
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# --- PAGE SETTINGS ---
st.set_page_config(page_title="UPS Pilot Assistant", page_icon="✈️", layout="wide")

# --- MEMORY ---
if "chat_history" not in st.session_state: st.session_state.chat_history = []
if "vector_store" not in st.session_state: st.session_state.vector_store = None

# --- SIDEBAR ---
with st.sidebar:
    st.header("🛠 Configuration")
    api_key = st.text_input("OpenAI API Key", type="password")
    
    st.divider()
    st.header("📚 Manuals & Contract")
    st.info("Upload PDF or ZIP files.")
    uploaded_files = st.file_uploader("Drop files here", accept_multiple_files=True, type=["pdf", "zip"])
    
    if uploaded_files and api_key and st.button("Process Documents"):
        with st.spinner("Unzipping and reading manuals..."):
            all_docs = []
            with tempfile.TemporaryDirectory() as temp_dir:
                for uploaded_file in uploaded_files:
                    # ZIP HANDLING
                    if uploaded_file.name.endswith(".zip"):
                        try:
                            with zipfile.ZipFile(uploaded_file, "r") as z:
                                z.extractall(temp_dir)
                                for root, dirs, files in os.walk(temp_dir):
                                    for file in files:
                                        if file.endswith(".pdf"):
                                            loader = PyPDFLoader(os.path.join(root, file))
                                            docs = loader.load()
                                            for d in docs: d.metadata["source"] = file
                                            all_docs.extend(docs)
                        except Exception as e:
                            st.error(f"Error reading zip: {e}")
                    # PDF HANDLING
                    elif uploaded_file.name.endswith(".pdf"):
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                            tmp.write(uploaded_file.getvalue())
                            tmp_path = tmp.name
                        loader = PyPDFLoader(tmp_path)
                        docs = loader.load()
                        for d in docs: d.metadata["source"] = uploaded_file.name
                        all_docs.extend(docs)
                        os.remove(tmp_path)
            
            # BUILD BRAIN
            if all_docs:
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
                chunks = text_splitter.split_documents(all_docs)
                embeddings = OpenAIEmbeddings(openai_api_key=api_key)
                st.session_state.vector_store = FAISS.from_documents(chunks, embeddings)
                st.success(f"Success! Analyzed {len(chunks)} sections.")
            else:
                st.warning("No PDF files found.")

# --- TABS ---
tab1, tab2 = st.tabs(["💬 Contract Chat", "📊 Fatigue Calculator"])

# --- TAB 1: CHATBOT ---
with tab1:
    st.title("UPS Contract & Systems Expert")
    if st.session_state.vector_store and api_key:
        llm = ChatOpenAI(model_name="gpt-4", temperature=0, openai_api_key=api_key)
        
        template = """
        You are an expert UPS Pilot Assistant. 
        Use the context (Contract, AOM, FOM) to answer the question.
        Cite your source (Article or Section).
        
        Context: {context}
        Question: {question}
        Answer:
        """
        QA_PROMPT = PromptTemplate(template=template, input_variables=["context", "question"])
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=st.session_state.vector_store.as_retriever(search_kwargs={"k": 4}),
            chain_type_kwargs={"prompt": QA_PROMPT}
        )

        for role, msg in st.session_state.chat_history:
            st.chat_message(role).write(msg)

        if prompt := st.chat_input("Ask a question..."):
            st.chat_message("user").write(prompt)
            st.session_state.chat_history.append(("user", prompt))
            with st.spinner("Checking manuals..."):
                try:
                    response = qa_chain.run(prompt)
                except Exception as e:
                    response = f"Error: {e}"
            st.chat_message("assistant").write(response)
            st.session_state.chat_history.append(("assistant", response))
    elif not api_key:
        st.info("👈 Enter API Key to start.")

# --- TAB 2: FATIGUE MODEL ---
with tab2:
    st.title("Fatigue Risk Analyzer")
    
    def calculate_fatigue_trip(report_dt, flights, is_intl):
        timeline = []
        reservoir = 2800 
        current_t = report_dt
        last_arrival = flights[-1]['arr']
        debrief_min = 30 if is_intl else 15
        duty_end = last_arrival + timedelta(minutes=debrief_min)
        
        while current_t <= duty_end:
            hour = current_t.hour
            circadian_pen = 0
            if 2 <= hour <= 6: circadian_pen = 20
            
            decay = 4 # Default ground
            activity = "Duty"
            for f in flights:
                if f['dept'] <= current_t <= f['arr']:
                    activity = "Fly"
                    block = (f['arr'] - f['dept']).total_seconds() / 3600
                    crew = 2
                    if is_intl:
                        if block >= 12.0: crew = 4
                        elif block > 7.75: crew = 3
                    if crew == 2: decay = 5
                    elif crew == 3: decay = 3.5
                    elif crew == 4: decay = 2.5
                    break
            
            reservoir -= decay
            eff_score = max(0, min(100, (reservoir / 2800) * 100 - circadian_pen))
            
            color = "🟢"
            if eff_score < 90: color = "🟡"
            if eff_score < 85: color = "🟠"
            if eff_score < 80: color = "🔴"
            if eff_score < 75: color = "🟣"
            
            timeline.append({
                "Time": current_t.strftime("%H:%M"),
                "Activity": activity,
                "Effectiveness": round(eff_score, 1),
                "Risk": color
            })
            current_t += timedelta(minutes=60)
            
        return pd.DataFrame(timeline)

    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Trip Start Date")
        report_time = st.time_input("Report Time (Local)", datetime.strptime("08:00", "%H:%M").time())
    with col2:
        block_hrs = st.number_input("Block Hours", 13.5)
        intl_rules = st.checkbox("International Rules?", value=True)

    if st.button("Run Analysis"):
        report_dt = datetime.combine(start_date, report_time)
        dept_buffer = 1.5 if intl_rules else 1.0
        dept_dt = report_dt + timedelta(hours=dept_buffer)
        arr_dt = dept_dt + timedelta(hours=block_hrs)
        flights = [{'dept': dept_dt, 'arr': arr_dt}]
        
        results = calculate_fatigue_trip(report_dt, flights, intl_rules)
        st.metric("Minimum Effectiveness", f"{results['Effectiveness'].min()}%")
        st.dataframe(results, use_container_width=True)
        st.line_chart(results.set_index("Time")["Effectiveness"])
