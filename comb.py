import os
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.llms import HuggingFacePipeline
from langchain.chains import LLMChain, ConversationalRetrievalChain
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, BitsAndBytesConfig
import torch

app = Flask(__name__)
ALLOWED_EXTENSIONS = {'pdf', 'txt', 'csv'}
uploaded_files = []
embeddings = None
retriever = None
qa_chain = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def index():
    global embeddings, retriever, qa_chain

    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)

        file = request.files['file']

        if file.filename == '':
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join('uploads', filename)
            file.save(file_path)
            uploaded_files.append(file_path)

            if len(uploaded_files) == 1:
                initialize_rag_chatbot()

            return redirect(url_for('index'))

    return render_template('index.html', uploaded_files=uploaded_files)

def initialize_rag_chatbot():
    global embeddings, retriever, qa_chain

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_name = 'mistralai/Mistral-7B-Instruct-v0.2'

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    use_4bit = True
    bnb_4bit_compute_dtype = "float16"
    bnb_4bit_quant_type = "nf4"
    use_nested_quant = False
    compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=use_4bit,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=use_nested_quant,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
    )

    text_generation_pipeline = transformers.pipeline(
        model=model,
        tokenizer=tokenizer,
        task="text-generation",
        temperature=0.1,
        repetition_penalty=1.1,
        return_full_text=True,
        max_new_tokens=3500,
        top_p=0.95,
        top_k=5,
        do_sample=True,
    )

    mistral_llm = HuggingFacePipeline(pipeline=text_generation_pipeline)

    all_documents = []
    for file_path in uploaded_files:
        file_extension = file_path.rsplit('.', 1)[1].lower()
        if file_extension == 'pdf':
            documents = PyPDFLoader(file_path).load()
        elif file_extension == 'txt':
            documents = TextLoader(file_path).load()
        elif file_extension == 'csv':
            loader = CSVLoader(file_path=file_path)
            data = loader.load()
            text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
            documents = text_splitter.split_documents(data)
        else:
            continue

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=250)
        chunked_documents = text_splitter.split_documents(documents)
        all_documents.extend(chunked_documents)

    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/gtr-t5-large')
    db = FAISS.from_documents(all_documents, embeddings)
    retriever = db.as_retriever(search_kwargs={'k': 13})

    prompt_template = """
    ### [INST] Instruction: Answer the question based on the information from the uploaded files. Here is context to help:

    {context}

    ### QUESTION:
    {question} [/INST]
    """

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=prompt_template,
    )

    llm_chain = LLMChain(llm=mistral_llm, prompt=prompt)

    qa_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | llm_chain
    )

@app.route('/chat', methods=['POST'])
def chat():
    global qa_chain

    if qa_chain is None:
        return "RAG chatbot not initialized. Please upload files first."

    question = request.form.get('question')
    result = qa_chain.invoke(question)
    return result['answer']

if __name__ == '__main__':
    app.run(debug=True)


pip install -q -U torch datasets transformers tensorflow langchain playwright html2text sentence_transformers faiss-cpu
pip install -q accelerate==0.21.0 peft==0.4.0 bitsandbytes==0.40.2 trl==0.4.7