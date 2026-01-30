import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import tempfile
import os
import zipfile
import base64
from io import BytesIO
from PIL import Image
import json

# --- IMPORTS ---
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import PromptTemplate
from openai import OpenAI

# --- PAGE CONFIG ---
st.set_page_config(page_title="UPS Pilot Assistant", page_icon="✈️", layout="wide")

# --- MEMORY ---
if "chat_history" not in st.session_state: st.session_state.chat_history = []
if "vector_store" not in st.session_state: st.session_state.vector_store = None

# --- HELPER: IMAGE ENCODER ---
def encode_image(image):
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

# --- SIDEBAR ---
with st.sidebar:
    st.header("🛠 Configuration")
    api_key = st.text_input("OpenAI API Key", type="password")
    
    st.divider()
    st.header("📚 Manuals & Contract")
    uploaded_files = st.file_uploader("Upload Manuals (PDF/ZIP)", accept_multiple_files=True, type=["pdf", "zip"])
    
    if uploaded_files and api_key and st.button("Process Documents"):
        with st.spinner("Processing manuals..."):
            all_docs = []
            with tempfile.TemporaryDirectory() as temp_dir:
                for uploaded_file in uploaded_files:
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
                        except: pass
                    elif uploaded_file.name.endswith(".pdf"):
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                            tmp.write(uploaded_file.getvalue())
                            tmp_path = tmp.name
                        loader = PyPDFLoader(tmp_path)
                        docs = loader.load()
                        for d in docs: d.metadata["source"] = uploaded_file.name
                        all_docs.extend(docs)
                        os.remove(tmp_path)
            
            if all_docs:
                try:
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
                    chunks = text_splitter.split_documents(all_docs)
                    
                    # UPDATED: Explicitly use the modern model
                    embeddings = OpenAIEmbeddings(openai_api_key=api_key, model="text-embedding-3-small")
                    st.session_state.vector_store = FAISS.from_documents(chunks, embeddings)
                    st.success(f"Success! Knowledge Base Updated.")
                except Exception as e:
                    st.error(f"OpenAI API Error: {e}")
                    st.info("Tip: If you just added funds, wait 10 mins and generate a NEW key.")

# --- TABS ---
tab1, tab2 = st.tabs(["💬 Contract Chat", "📊 Trip Fatigue Analyzer"])

# --- TAB 1: CHATBOT ---
with tab1:
    st.title("UPS Contract & Systems Expert")
    if st.session_state.vector_store and api_key:
        llm = ChatOpenAI(model="gpt-4o", temperature=0, openai_api_key=api_key)
        retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 4})

        for role, msg in st.session_state.chat_history:
            st.chat_message(role).write(msg)

        if prompt := st.chat_input("Ask a question..."):
            st.chat_message("user").write(prompt)
            st.session_state.chat_history.append(("user", prompt))
            with st.spinner("Checking manuals..."):
                try:
                    docs = retriever.invoke(prompt)
                    context_text = "\n\n".join([d.page_content for d in docs])
                    system_prompt = f"You are a UPS Pilot Assistant. Answer using ONLY context.\nContext:\n{context_text}\nQuestion: {prompt}"
                    response = llm.invoke(system_prompt).content
                except Exception as e: response = f"Error: {e}"
            st.chat_message("assistant").write(response)
            st.session_state.chat_history.append(("assistant", response))
    elif not api_key: st.info("Enter API Key to start.")

# --- TAB 2: FATIGUE MODEL (VISION UPGRADE) ---
with tab2:
    st.title("Full Trip Fatigue Analyzer")
    st.info("Upload a screenshot or PDF of your pairing to analyze the entire trip.")

    # 1. PARSING FUNCTION
    def parse_schedule(uploaded_file, key):
        client = OpenAI(api_key=key)
        
        # Prepare content based on file type
        if uploaded_file.type in ["image/png", "image/jpeg"]:
            image = Image.open(uploaded_file)
            base64_image = encode_image(image)
            content = [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}]
        else:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.getvalue())
                path = tmp.name
            loader = PyPDFLoader(path)
            pages = loader.load()
            text_content = "\n".join([p.page_content for p in pages])
            content = [{"type": "text", "text": f"Extract flight data from this text:\n{text_content}"}]
            os.remove(path)

        # Prompt GPT-4o to extract JSON
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": """
                Extract the flight schedule into a JSON list. 
                Each item must have: 'date' (YYYY-MM-DD), 'dept_time' (HH:MM 24h), 'arr_time' (HH:MM 24h), 'origin', 'dest'. 
                Assume current year if missing. Return ONLY raw JSON.
                """},
                {"role": "user", "content": content}
            ],
            max_tokens=1000
        )
        try:
            raw = response.choices[0].message.content.replace("```json", "").replace("```", "")
            return json.loads(raw)
        except: return None

    # 2. FATIGUE ENGINE (Multi-Leg)
    def calculate_multi_leg_fatigue(legs, is_intl):
        timeline = []
        reservoir = 2800 # Max Fuel
        current_time = datetime.strptime(f"{legs[0]['date']} {legs[0]['dept_time']}", "%Y-%m-%d %H:%M") - timedelta(hours=1.5) # Report
        
        sorted_legs = []
        for l in legs:
            dept = datetime.strptime(f"{l['date']} {l['dept_time']}", "%Y-%m-%d %H:%M")
            arr = datetime.strptime(f"{l['date']} {l['arr_time']}", "%Y-%m-%d %H:%M")
            if arr < dept: arr += timedelta(days=1)
            sorted_legs.append({'dept': dept, 'arr': arr, 'orig': l['origin'], 'dest': l['dest']})
            
        final_time = sorted_legs[-1]['arr'] + timedelta(minutes=30)
        
        while current_time <= final_time:
            in_flight = False
            for leg in sorted_legs:
                if leg['dept'] <= current_time <= leg['arr']:
                    in_flight = True
                    break
            
            if not in_flight:
                time_to_next = 999
                for leg in sorted_legs:
                    if leg['dept'] > current_time:
                        time_to_next = (leg['dept'] - current_time).total_seconds() / 3600
                        break
                if time_to_next > 10: reservoir = min(2800, reservoir + 200) 
                else: reservoir -= 2 
            else:
                decay = 4
                if is_intl: decay = 2.5 
                reservoir -= decay

            hour = current_time.hour
            circadian = 0
            if 2 <= hour <= 6: circadian = 20
            
            score = max(0, min(100, (reservoir/2800)*100 - circadian))
            color = "🟢"
            if score < 85: color = "🟡"
            if score < 75: color = "🟠"
            if score < 70: color = "🔴"
            if score < 60: color = "🟣"
            
            timeline.append({
                "Time": current_time.strftime("%d/%H:%M"),
                "Activity": "Fly" if in_flight else "Rest/Duty",
                "Eff": int(score),
                "Risk": color
            })
            current_time += timedelta(minutes=60)
        return pd.DataFrame(timeline)

    # 3. INTERFACE
    uploaded_pairing = st.file_uploader("Upload Pairing (Screenshot or PDF)", type=["png", "jpg", "jpeg", "pdf"])
    intl_check = st.checkbox("Apply International Augmentation Rules?", value=True)
    
    if uploaded_pairing and api_key and st.button("Analyze Pairing"):
        with st.spinner("AI is reading your schedule..."):
            schedule_data = parse_schedule(uploaded_pairing, api_key)
        
        if schedule_data:
            st.success("Schedule Extracted Successfully!")
            st.json(schedule_data) 
            results = calculate_multi_leg_fatigue(schedule_data, intl_check)
            st.divider()
            st.subheader("Fatigue Analysis")
            min_score = results['Eff'].min()
            st.metric("Lowest Effectiveness", f"{min_score}%")
            if min_score < 70: st.error("⚠️ HIGH FATIGUE RISK DETECTED")
            st.line_chart(results.set_index("Time")["Eff"])
            st.dataframe(results, use_container_width=True)
        else:
            st.error("Could not read schedule.")
