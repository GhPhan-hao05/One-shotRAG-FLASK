from flask import Flask, render_template, request, jsonify, session
import os
from llama_index.core import VectorStoreIndex
from llama_index.llms.openai import OpenAI
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.chat_engine  import SimpleChatEngine
from pinecone import Pinecone
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core import Document
from llama_index.core import StorageContext

import uuid
import tempfile
os.environ["OPENAI_API_KEY"]= '[YOUR OPENAI API KEY]'  # Replace YOUR_API_KEY with your actual API key

app = Flask(__name__)
app.secret_key = os.urandom(24)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize global variables to store components
global_retriever = None
global_chat_engine = None
global_memory = None


def initialize_rag_components():
    """Initialize or reinitialize the RAG components"""
    global global_retriever, cls_engine, oneshot_engine, chat_engine, global_memory
    file_path = "RAG_data.txt"
    documents = []
    with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.read().splitlines()
    documents = [Document(text=line) for line in lines if line.strip()]
    pc = Pinecone(api_key='[YOUR PINECONE API KEY]')
    dataset_name = "nttu"
    pinecone_index = pc.Index(dataset_name)
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context
    )
    query_engine = index.as_query_engine(similarity_top_k=4)
    global_retriever = query_engine.retriever
    system_prompt_1 = (
        """Bạn là một chuyên gia phân loại câu hỏi/truy vấn RAG hoặc NO-RAG.
    Quyết định xem tin nhắn sau có cần tạo tăng cường truy xuất (RAG) hay không.
    Những câu hỏi về nhưng thông tin riêng, chuyên biệt của các tổ chức hoặc có liên quan đến các câu hỏi cần RAG trong bộ nhớ thì sẽ dùng RAG ngược lại là NO-RAG.
    Trả lời chỉ trả lời với duy nhất 1 trong 2 từ sau: "RAG" hoặc "NO-RAG".
    Ví dụ:
    "Giải thích LSTM" : NO_RAG
    "Luật đất đai của Việt Nam như thế nào?" hoặc "Cách thức nộp hồ sơ sinh viên" (không nói tên trường nhưng ngầm hiểu nó thuộc về một trường cụ thể nào đó): RAG
    "Các hồ sơ có cần phải công chứng không ?"(Câu hỏi liên quan với câu hỏi trên): RAG
    QUAN TRỌNG: chỉ phân loại câu hỏi, không trả lời nó!!!
    """
    )

    system_prompt_2 = (
    """khi bạn nhận 1 câu hỏi, hãy mô tả sơ lược về câu hỏi bạn nhận được, nhưng đừng mô tả quá dài
    ví dụ:
    input: cách thức tính điểm của 1 môn học phần?
    output: cách thức tính điểm của 1 môn học phần? Giải thích: giải thích rõ các điểm thành phần, phần trăm mỗi điểm thành phần, áp dụng công thức, và xác định điểm cuối cùng của môn học

    input: Quy trình cấp giấy xác nhận sinh viên như thế nào?
    output: Quy trình cấp giấy xác nhận sinh viên như thế nào? Giải thích: giải thích các bước cần làm, Thủ tục nộp đơn xin xác nhận, điền thông tin, Thời gian xử lý, Và nơi nhận kết quả """
    )
    # Initialize LLM
    llm = OpenAI()
    
    # Initialize memory
    global_memory = ChatMemoryBuffer.from_defaults(token_limit=3900)
    
    # Check if there are documents in the upload folder
    cls_engine = SimpleChatEngine.from_defaults(
    llm=llm,
    memory=global_memory,  # Both engines use the same memory instance
    system_prompt=system_prompt_1,
    )

    oneshot_engine = SimpleChatEngine.from_defaults(
        llm=llm,
        memory=global_memory,  # Both engines use the same memory instance
        system_prompt=system_prompt_2
    )
    chat_engine = SimpleChatEngine.from_defaults(
        llm=llm,
        memory=global_memory,  # Both engines use the same memory instance
    )

def classification_query (query):
  global cls_engine
  cls = cls_engine.chat(query)
  return cls.response
def get_more_inf(query):
  global oneshot_engine
  oneshot = oneshot_engine.chat(query)
  return oneshot.response

@app.route('/')
def home():
    # Generate a session ID if one doesn't exist
    if 'user_id' not in session:
        session['user_id'] = str(uuid.uuid4())
    
    return render_template('index.html')


@app.route('/chat', methods=['POST'])
def chat():
    global global_retriever, cls_engine, oneshot_engine, chat_engine, global_memory
    
    # Ensure RAG components are initialized
    if [cls_engine, oneshot_engine, chat_engine] is None:
        initialize_rag_components()
    
    user_input = request.json.get('message', '')
    
    try:
        if classification_query(user_input) == "RAG":
            template_str = """
            Bạn là một trợ lý thông minh. trả lời câu hỏi dựa vào thông tin truy xuất được.
            Nếu không có thông tin, hãy nói "Tôi không chắc dựa trên thông tin đã cho."
            Thông tin:
            {c}
            Câu hỏi:
            {q}
            Trả lời:
            """
            query_w_inf = get_more_inf(user_input)
            retrieved_nodes = global_retriever.retrieve(query_w_inf)
            content = [node.get_content() for node in retrieved_nodes]
            print(content)
            print('1')
            text_with_prompt_template = template_str.format(c=content, q = user_input)
            print(text_with_prompt_template)
            response = chat_engine.chat(text_with_prompt_template)
        else:
            # LLM-only mode
            print('2')
            response = chat_engine.chat(user_input)
        response_text = response.response
        
        return jsonify({'response': response_text})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'document' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['document']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file:
        # Save uploaded file
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)
        
        # Reinitialize RAG components with new document
        initialize_rag_components()
        
        return jsonify({'success': f'File {file.filename} uploaded successfully'})
    
    return jsonify({'error': 'File upload failed'}), 500


@app.route('/clear', methods=['POST'])
def clear_memory():
    global global_memory
    
    if global_memory:
        global_memory.reset()
    
    return jsonify({'success': 'Chat memory cleared'})


if __name__ == '__main__':
    # Initialize RAG components on startup
    initialize_rag_components()
    app.run(debug=True)