import streamlit as st
import chromadb
from chromadb.utils import embedding_functions
import google.generativeai as genai
import json
import os

# ==========================================
# 1. CẤU HÌNH & GIAO DIỆN (THEME NAVY-GREY)
# ==========================================
st.set_page_config(page_title="NABIN AI", layout="wide", page_icon="💖")

# --- CSS TÙY CHỈNH: NAVY - GREY THEME ---
st.markdown(f"""
    <style>
    /* 1. Màu nền chính (Xám nhạt) */
    .stApp {{ background-color: #F0F2F6; }}
    
    /* 2. Màu nền Sidebar (Xám đậm) */
    [data-testid="stSidebar"] {{ background-color: #2D2D2D; }}
    
    /* 3. Màu chữ trong Sidebar (Trắng) */
    [data-testid="stSidebar"] .stMarkdown p, [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] div {{
        color: white !important;
    }}
    
    /* 4. Tiêu đề chính (Xanh Navy) */
    h1 span {{ color: #001F3F; font-weight: 800; }}
    h3 {{ color: #001F3F; }}
    
    /* 5. Tùy chỉnh bong bóng chat */
    .stChatMessage {{ background-color: transparent; }}
    
    /* 6. Tùy chỉnh nút Link (Map) */
    a[href] {{
        text-decoration: none;
        color: #001F3F !important;
        font-weight: bold;
    }}
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 2. XỬ LÝ DỮ LIỆU & AI
# ==========================================

# --- A. Xử lý ChromaDB (Lưu trữ vĩnh viễn) ---
@st.cache_resource
def get_chroma_collection():
    # Tạo thư mục lưu DB để không phải index lại mỗi lần reload
    if not os.path.exists("nabin_db_data"):
        os.makedirs("nabin_db_data")
        
    client = chromadb.PersistentClient(path="nabin_db_data")
    
    # Dùng model embedding nhẹ
    emb_func = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="paraphrase-multilingual-MiniLM-L12-v2"
    )
    
    collection = client.get_or_create_collection(name="nabin_places", embedding_function=emb_func)
    return collection

collection = get_chroma_collection()

# --- B. Hàm nạp dữ liệu ---
def index_data():
    try:
        data = []
        # Kiểm tra file tồn tại không để tránh lỗi
        if os.path.exists("food.json"):
            with open("food.json", "r", encoding="utf-8") as f: data += json.load(f)
        if os.path.exists("drink.json"):
            with open("drink.json", "r", encoding="utf-8") as f: data += json.load(f)
            
        if not data: return 0, "Không tìm thấy file json!"

        ids = []
        documents = []
        metadatas = []

        for i, item in enumerate(data):
            # Tạo nội dung text để AI đọc
            content = f"Tên quán: {item['name']}. Địa chỉ: {item['address']}. Mood/Không gian: {item.get('mood', 'Không rõ')}. Ghi chú món: {item.get('notes', '')}"
            
            ids.append(f"place_{i}")
            documents.append(content)
            # Lưu link map vào metadata để truy xuất sau
            metadatas.append({
                "name": item['name'],
                "address": item['address'],
                "map": item.get("map_link", "https://maps.google.com")
            })

        # Thêm vào DB
        collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
        return len(data), "Thành công"
    except Exception as e:
        return 0, str(e)

# ==========================================
# 3. SIDEBAR (CÀI ĐẶT)
# ==========================================
with st.sidebar:
    st.title("⚙️ Cài đặt NABIN")
    st.markdown("---")
    
    # Ưu tiên lấy API Key từ Secrets, nếu không có thì hiện ô nhập
    if "GOOGLE_API_KEY" in st.secrets:
        api_key = st.secrets["GOOGLE_API_KEY"]
        st.success("✅ Đã kết nối API Key từ hệ thống")
    else:
        api_key = st.text_input("Nhập Gemini API Key", type="password")
    
    st.markdown("---")
    if st.button("🔄 Nạp dữ liệu Quán (Re-index)"):
        with st.spinner("Đang học dữ liệu mới..."):
            count, msg = index_data()
            if count > 0:
                st.success(f"Đã nạp {count} địa điểm!")
            else:
                st.error(f"Lỗi: {msg}")

    if st.button("🗑️ Xóa lịch sử chat"):
        st.session_state.messages = []
        st.session_state.pop('last_results', None)
        st.rerun()

# ==========================================
# 4. GIAO DIỆN CHÍNH (2 CỘT)
# ==========================================
st.title("💖 NABIN - Trợ lý của Thanh Huy")

col1, col2 = st.columns([2, 1])

# --- CỘT 1: CHATBOT ---
with col1:
    st.subheader("💬 Trò chuyện")
    
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hé lô Thanh Huy! Hôm nay anh muốn đi ăn hay đi uống nước nè? 💖"}
        ]

    # Hiển thị lịch sử
    for msg in st.session_state.messages:
        avatar = "🦸‍♂️" if msg["role"] == "user" else "🧚‍♀️"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])

    # Xử lý nhập liệu
    if prompt := st.chat_input("Gõ vào đây nha... (ví dụ: Tìm quán cafe yên tĩnh làm việc)"):
        if not api_key:
            st.warning("Vui lòng nhập API Key ở Sidebar trước nha!")
        else:
            # 1. Hiển thị User Message
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user", avatar="🦸‍♂️"):
                st.markdown(prompt)

            # 2. Xử lý RAG + AI
            with st.chat_message("assistant", avatar="🧚‍♀️"):
                with st.spinner("Nabin đang suy nghĩ..."):
                    genai.configure(api_key=api_key)
                    
                    # Tìm kiếm trong ChromaDB
                    results = collection.query(query_texts=[prompt], n_results=3)
                    
                    # Ghép context
                    context_text = ""
                    if results['documents'] and results['documents'][0]:
                        context_text = "\n".join(results['documents'][0])
                        # Lưu kết quả tìm kiếm để hiển thị bên Cột 2
                        st.session_state.last_results = results
                    else:
                        st.session_state.last_results = None

                    # Prompt cho Gemini
                    sys_instruction = f"""Bạn là NABIN, trợ lý người yêu ảo cực kỳ dễ thương của Thanh Huy.
                    Nhiệm vụ: Tư vấn địa điểm ăn uống dựa trên danh sách sau đây.
                    
                    Danh sách quán tìm được:
                    {context_text}
                    
                    Yêu cầu:
                    - Trả lời giọng điệu cute, quan tâm (gọi là 'anh', xưng 'em' hoặc 'Nabin').
                    - Nếu tìm thấy quán, hãy tóm tắt tại sao quán đó phù hợp.
                    - Nếu không thấy quán phù hợp trong danh sách, hãy gợi ý dựa trên kiến thức chung nhưng nói rõ là "Em không thấy trong danh sách quán quen, nhưng em biết chỗ này...".
                    """
                    
                    try:
                        model = genai
                        
