import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import tempfile
import os
import zipfile

# --- MODERN IMPORTS (Matches the flexible requirements) ---
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

# --- PAGE SETTINGS ---
st.set_page_config(page_title="UPS Pilot Assistant", page_icon="✈️", layout="wide")

# --- MEMORY (SESSION STATE) ---
if "chat_history" not in st.session_state: st.session_state.chat_history = []
if "vector_store" not in st.session_state: st.session_state.vector_store = None

# --- SIDEBAR: CONFIG & UPLOADS ---
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
                        except Exception as e:
                            st.error(f"Error reading zip: {e}")

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
                try:
                    response = qa_chain.run(prompt)
                except Exception as e:
                    response = f"Error: {e}"
            
            st.chat_message("assistant").write(response)
            st.session_state.chat_history.append(("assistant", response))

    elif not api_key:
        st.info("👈 Enter your OpenAI API Key in the sidebar to start.")
    else:
        st.info("👈 Upload your manuals (PDF or ZIP) in the sidebar and click Process.")


# --- TAB 2: FATIGUE MODEL ---
with tab2:
    st.title("Fatigue Risk Analyzer")
    st.markdown("Model based on SAFTE-style decay and UPS Contract Rules.")
    
    # FATIGUE LOGIC
    def calculate_fatigue_trip(report_dt, flights, is_intl):
        timeline = []
        reservoir = 2800 # "Fuel" for alertness (approx minutes)
        
        current_t = report_dt
        
        # Determine Duty End (Debrief)
        last_arrival = flights[-1]['arr']
        debrief_min = 30 if is_intl else 15
        duty_end = last_arrival + timedelta(minutes=debrief_min)
        
        # Simulation Loop (15 min increments)
        while current_t <= duty_end:
            # 1. Circadian Factor (WOCL 0200-0600 Home Base)
            hour = current_t.hour
            circadian_pen = 0
            if 2 <= hour <= 6: circadian_pen = 20 # Heavy WOCL Hit
            
            # 2. Activity Factor
            decay = 0
            activity = "Duty"
            
            # Check if Flying
            for f in flights:
                if f['dept'] <= current_t <= f['arr']:
                    activity = "Fly"
                    block = (f['arr'] - f['dept']).total_seconds() / 3600
                    
                    # Augmentation Rules
                    crew = 2
                    if is_intl:
                        if block >= 12.0: crew = 4
                        elif block > 7.75: crew = 3
                    
                    # Decay Rates
                    if crew == 2: decay = 5
                    elif crew == 3: decay = 3.5
                    elif crew == 4: decay = 2.5
                    break
            
            if activity == "Duty": decay = 4 # Ground duty is tiring
            
            # Update Reservoir
            reservoir -= decay
            
            # Calculate Effectiveness (0-100%)
            # Max reservoir is 2800. 
            eff_score = (reservoir / 2800) * 100 - circadian_pen
            eff_score = max(0, min(100, eff_score))
            
            # Color Coding
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
            
            current_t += timedelta(minutes=60) # Step 1 hour for graph
            
        return pd.DataFrame(timeline)

    # INPUTS
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Trip Start Date")
        report_time = st.time_input("Report Time (Local)", datetime.strptime("08:00", "%H:%M").time())
        origin = st.text_input("Origin", "SDF")
        
    with col2:
        dest = st.text_input("Destination", "ICN")
        block_hrs = st.number_input("Block Hours", 13.5)
        intl_rules = st.checkbox("International Rules?", value=True)

    if st.button("Run Analysis"):
        # Construct Duty
        report_dt = datetime.combine(start_date, report_time)
        
        # Calculate Flight Times (Simplified for single leg demo)
        # In full app, user would add multiple legs
        dept_buffer = 1.5 if intl_rules else 1.0 # 1:30 prior vs 1:00 prior
        dept_dt = report_dt + timedelta(hours=dept_buffer)
        arr_dt = dept_dt + timedelta(hours=block_hrs)
        
        flights = [{'dept': dept_dt, 'arr': arr_dt}]
        
        # Run Model
        results = calculate_fatigue_trip(report_dt, flights, intl_rules)
        
        # Display Metrics
        min_eff = results['Effectiveness'].min()
        st.metric("Minimum Effectiveness", f"{min_eff}%")
        
        # Display Table
        st.dataframe(results, use_container_width=True)
        
        # Display Chart
        st.line_chart(results.set_index("Time")["Effectiveness"])
