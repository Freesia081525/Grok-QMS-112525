import streamlit as st
import yaml
import os
#from utils.grok_client import grok_chat
import google.generativeai as genai
from openai import OpenAI
import os
from xai_sdk import Client
from xai_sdk.chat import user, system

def grok_chat(messages, model="grok-4", temperature=0.3, max_tokens=8000):
    api_key = os.getenv("XAI_API_KEY")
    if not api_key:
        raise ValueError("請設定 XAI_API_KEY")
    
    client = Client(api_key=api_key, timeout=3600)
    chat = client.chat.create(model=model)
    
    for msg in messages:
        if msg["role"] == "system":
            chat.append(system(msg["content"]))
        else:
            chat.append(user(msg["content"]))
    
    response = chat.sample(
        temperature=temperature,
        max_tokens=max_tokens
    )
    return response.content

# ================== 載入 agents ==================
with open("agents.yaml", encoding="utf-8") as f:
    AGENTS_CONFIG = yaml.safe_load(f)["agents"]

# ================== Sidebar API Key 輸入 ==================
st.sidebar.title("API 金鑰設定（不會顯示）")
gemini_key = st.sidebar.text_input("Gemini API Key", type="password")
openai_key = st.sidebar.text_input("OpenAI API Key", type="password")
xai_key = st.sidebar.text_input("xAI (Grok) API Key", type="password")

if gemini_key:
    genai.configure(api_key=gemini_key)
if openai_key:
    os.environ["OPENAI_API_KEY"] = openai_key
if xai_key:
    os.environ["XAI_API_KEY"] = xai_key

st.title("醫療器材稽核報告自動重組系統")
st.markdown("### 貼上原始觀察紀錄 + 模板 → 自動產出正式報告（100%保留原文）")

col1, col2 = st.columns(2)
with col1:
    observations = st.text_area("【原始觀察紀錄】貼上這裡（純文字或 Markdown）", height=400)
with col2:
    template = st.text_area("【模板報告】貼上 Article A 或其他報告範本", height=400, value=st.session_state.get("last_template", ""))

if st.button("開始執行 Agent 流程", type="primary"):
    if not observations or not template:
        st.error("請同時填入觀察紀錄與模板")
    else:
        # 儲存模板供後續使用
        st.session_state.last_template = template
        
        # 初始化歷史
        if "agent_outputs" not in st.session_state:
            st.session_state.agent_outputs = {}
        if "chain_history" not in st.session_state:
            st.session_state.chain_history = []

        # 逐個 Agent 執行
        previous_output = ""
        for idx, (key, agent) in enumerate(AGENTS_CONFIG.items(), 1):
            st.subheader(f"{idx}. {agent['name']}")
            
            with st.expander(f"調整參數 - {agent['name']}", expanded=False):
                selected_model = st.selectbox(f"選擇模型 - {idx}", agent["model_options"], 
                                            index=agent["model_options"].index(agent["default_model"]), key=f"model_{idx}")
                max_tokens = st.slider("Max Tokens", 1000, 16000, agent["max_tokens"], key=f"tokens_{idx}")
                temperature = st.slider("Temperature", 0.0, 1.0, agent["temperature"], 0.05, key=f"temp_{idx}")
                custom_prompt = st.text_area("自訂系統提示詞（可留空使用預設）", height=150, key=f"prompt_{idx}")

            if st.button(f"執行 {agent['name']}", key=f"run_{idx}"):
                with st.spinner(f"執行中… 使用 {selected_model}"):
                    # 建構訊息
                    messages = [
                        {"role": "system", "content": custom_prompt or f"你是極專業的{agent['description']}，請用繁體中文輸出。"},
                        {"role": "user", "content": f"""【任務】
{agent['description']}

【模板報告（請嚴格依照此結構）】
{template}

【原始觀察紀錄（請100%保留原始文字）】
{observations}

【前一個Agent輸出（若有）】
{previous_output}

請輸出完整的 Markdown 格式結果。"""}
                    ]

                    # 呼叫不同 API
                    try:
                        if "gemini" in selected_model:
                            model = genai.GenerativeModel(selected_model.replace("-latest", ""))
                            response = model.generate_content(
                                messages[1]["content"],
                                generation_config={"temperature": temperature, "max_output_tokens": max_tokens}
                            )
                            result = response.text
                        elif "gpt" in selected_model:
                            client = OpenAI()
                            resp = client.chat.completions.create(
                                model=selected_model,
                                messages=messages,
                                temperature=temperature,
                                max_tokens=max_tokens
                            )
                            result = resp.choices[0].message.content
                        else:  # Grok
                            result = grok_chat(messages, model=selected_model, temperature=temperature, max_tokens=max_tokens)
                        
                        st.session_state.agent_outputs[key] = result
                        previous_output = result
                        st.session_state.chain_history.append((agent['name'], result))
                        st.success("完成！")
                    except Exception as e:
                        st.error(f"API 錯誤：{str(e)}")
                
                # 顯示並允許編輯
                if key in st.session_state.agent_outputs:
                    edited = st.text_area("可編輯此步驟輸出（作為下一步輸入）", 
                                        value=st.session_state.agent_outputs[key], height=500, key=f"edit_{idx}")
                    st.session_state.agent_outputs[key] = edited
                    previous_output = edited
                    st.markdown("---")

        # 最終報告與 Follow-up Questions
        if len(st.session_state.chain_history) == len(AGENTS_CONFIG):
            st.success("所有 Agent 已完成！")
            final_report = st.session_state.agent_outputs[list(AGENTS_CONFIG.keys())[-2]]  # 倒數第二個是正式報告
            st.markdown("## 最終稽核報告")
            st.markdown(final_report)
            
            st.download_button("下載報告.md", final_report, "audit_report.md")

            # 最後一步：產生 Follow-up Questions
            st.markdown("## 後續追問問題（Follow-up Questions）")
            if st.button("產生 Follow-up Questions"):
                with st.spinner("產生中…"):
                    # 直接用最後一個 Agent
                    last_agent = list(AGENTS_CONFIG.values())[-1]
                    messages = [
                        {"role": "system", "content": "你是最專業的醫療器材稽核顧問，請產生8-12條高價值、可立即用於現場追問的問題。"},
                        {"role": "user", "content": f"以下是最終稽核報告，請產生後續追問問題：\n\n{final_report}"}
                    ]
                    questions = grok_chat(messages, model="grok-4", max_tokens=2500) if "XAI_API_KEY" in os.environ else "（請設定 Grok Key）"
                    st.markdown(questions)
