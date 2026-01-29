import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import tempfile
import os
import zipfile

# --- LIBRARIES FOR AI/PDF READING ---
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# --- PAGE SETTINGS ---
st.set_page_config(page_title="UPS Pilot Assistant", page_icon="✈️", layout="wide")

# --- MEMORY (SESSION STATE) ---
if "chat_history" not in st.session_state: st.session_state.chat_history = []
if "vector_store" not in st.session_state: st.session_state.vector_store = None

# --- SIDEBAR ---
with st.sidebar:
    st.header("🛠 Configuration")
    api_key = st.text_input("OpenAI API Key", type="password")
    
    st.divider()
    st.header("📚 Manuals & Contract")
    st.info("Upload PDF or ZIP files (Contract, AOM, FOM).")
    uploaded_files = st.file_uploader("Drop files here", accept_multiple_files=True, type=["pdf", "zip"])
    
    if uploaded_files and api_key and st.button("Process Documents"):
        with st.spinner("Unzipping and reading manuals... (This takes 1-2 mins)"):
            all_docs = []
            
            # Create a temporary workspace
            with tempfile.TemporaryDirectory() as temp_dir:
                for uploaded_file in uploaded_files:
                    
                    # LOGIC: HANDLE ZIP FILES
                    if uploaded_file.name.endswith(".zip"):
                        try:
                            with zipfile.ZipFile(uploaded_file, "r") as z:
                                z.extractall(temp_dir)
                                # Find all PDFs inside the zip
                                for root, dirs, files in os.walk(temp_dir):
                                    for file in files:
                                        if file.endswith(".pdf"):
                                            full_path = os.path.join(root, file)
                                            loader = PyPDFLoader(full_path)
                                            docs = loader.load()
                                            for d in docs: d.metadata["source"] = file # Tag source
                                            all_docs.extend(docs)
                        except:
                            st.error(f"Error reading zip: {uploaded_file.name}")

                    # LOGIC: HANDLE REGULAR PDF
                    elif uploaded_file.name.endswith(".pdf"):
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                            tmp.write(uploaded_file.getvalue())
                            tmp_path = tmp.name
                        loader = PyPDFLoader(tmp_path)
                        docs = loader.load()
                        for d in docs: d.metadata["source"] = uploaded_file.name
                        all_docs.extend(docs)
                        os.remove(tmp_path)
            
            # BUILD THE BRAIN
            if all_docs:
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
                chunks = text_splitter.split_documents(all_docs)
                
                embeddings = OpenAIEmbeddings(openai_api_key=api_key)
                st.session_state.vector_store = FAISS.from_documents(chunks, embeddings)
                st.success(f"Success! Analyzed {len(chunks)} sections of text.")
            else:
                st.warning("No PDF files found.")

# --- TABS ---
tab1, tab2 = st.tabs(["💬 Pilot Assistant (Chat)", "📊 Fatigue Calculator"])

# --- TAB 1: CHATBOT ---
with tab1:
    st.title("UPS Contract & Systems Expert")
    st.markdown("Ask about the **Contract** ('Can I drop this trip?') or **Aircraft** ('Max crosswind 747?')")
    
    if st.session_state.vector_store and api_key:
        llm = ChatOpenAI(model_name="gpt-4o", temperature=0, openai_api_key=api_key)
        
        # INSTRUCTIONS FOR THE AI
        template = """
        You are an expert UPS Pilot Assistant. 
        Use the following pieces of context (Contract, AOM, FOM) to answer the question.
        
        - If the answer is in the Contract/Ref Guide, cite the Article.
        - If the answer is in the AOM/FOM, cite the Page/Section.
        - If you don't know, say "I don't see that in the uploaded manuals."
        
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

        # CHAT INTERFACE
        for role, msg in st.session_state.chat_history:
            st.chat_message(role).write(msg)

        if prompt := st.chat_input("Ask a question..."):
            st.chat_message("user").write(prompt)
            st.session_state.chat_history.append(("user", prompt))
            
            with st.spinner("Checking manuals..."):
                response = qa_chain.run(prompt)
            
            st.chat_message("assistant").write(response)
            st.session_state.chat_history.append(("assistant", response))

    elif not api_key:
        st.info("👈 Enter your OpenAI API Key in the sidebar to start.")
    else:
        st.info("👈 Upload your manuals (PDF or ZIP) in the sidebar and click Process.")


# --- TAB 2: FATIGUE MODEL ---
with tab2:
    st.title("Fatigue Risk Analyzer")
    
    # FATIGUE LOGIC (Embedded here for simplicity)
    def calculate_fatigue(dept_time, block_hrs, is_intl):
        # Simplified SAFTE-style math
        timeline = []
        # Start full (100% capacity)
        reservoir = 100 
        current_t = dept_time
        
        # Loop hour by hour
        for i in range(int(block_hrs) + 4): # Preflight + Flight + Post
            hour = current_t.hour
            
            # Circadian Low (WOCL) 0200-0600 home base
            circadian_pen = 0
            if 2 <= hour <= 6: circadian_pen = 15
            
            # Depletion
            if i == 0: activity = "Report"; decay = 2
            elif i <= block_hrs + 1: activity = "Fly"; decay = 4
            else: activity = "Debrief"; decay = 2
            
            reservoir -= decay
            effectiveness = max(0, reservoir - circadian_pen)
            
            color = "🟢"
            if effectiveness < 90: color = "🟡"
            if effectiveness < 80: color = "🟠"
            if effectiveness < 75: color = "🔴"
            
            timeline.append({"Time": current_t.strftime("%H:%M"), "Eff": effectiveness, "Risk": color})
            current_t += timedelta(hours=1)
            
        return pd.DataFrame(timeline)

    # INPUTS
    col1, col2 = st.columns(2)
    with col1:
        f_date = st.date_input("Flight Date")
        f_time = st.time_input("Report Time", datetime.now().time())
    with col2:
        blk = st.number_input("Block Hours", 5.0)
        intl = st.checkbox("International Rules?")

    if st.button("Analyze Trip"):
        dt = datetime.combine(f_date, f_time)
        res = calculate_fatigue(dt, blk, intl)
        st.dataframe(res, use_container_width=True)
        st.line_chart(res.set_index("Time")["Eff"])
