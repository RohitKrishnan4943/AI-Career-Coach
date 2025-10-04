from flask import Flask, request, render_template, redirect, url_for
import os
from werkzeug.utils import secure_filename
import PyPDF2
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# File upload configuration
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create required directories
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

if not os.path.exists('vector_index'):
    os.makedirs('vector_index')

# Initialize text splitter (moved outside to avoid recreation)
text_splitter = CharacterTextSplitter(
    separator='\n',
    chunk_size=2000,
    chunk_overlap=200,
    length_function=len,
)

# Initialize embeddings (lazy loading)
_embeddings = None
def get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
    return _embeddings

# Initialize LLM (lazy loading)
_llm = None
def get_llm():
    global _llm
    if _llm is None:
        _llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0,
            google_api_key=os.environ.get("GOOGLE_API_KEY"),
        )
    return _llm


def extract_text_from_pdf(pdf_path):
    """Extract text from PDF file"""
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page_num in range(len(reader.pages)):
            text += reader.pages[page_num].extract_text()
    return text


def perform_qa(query):
    """Perform Q&A using vector database"""
    try:
        embeddings = get_embeddings()
        db = FAISS.load_local("vector_index", embeddings, allow_dangerous_deserialization=True)
        retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 4})
        llm = get_llm()
        rqa = RetrievalQA.from_chain_type(
            llm=llm, 
            chain_type="stuff", 
            retriever=retriever, 
            return_source_documents=True
        )
        result = rqa.invoke(query)
        return result['result']
    except Exception as e:
        return f"Error processing query: {str(e)}"


# Resume analysis prompt template
resume_summary_template = """
Role: You are an AI Career Coach.

Task: Given the candidate's resume, provide a comprehensive summary that includes the following key aspects:

- Career Objective
- Skills and Expertise
- Professional Experience
- Educational Background
- Notable Achievements

Instructions:
Provide a concise summary of the resume, focusing on the candidate's skills, experience, and career trajectory. Ensure the summary is well-structured, clear, and highlights the candidate's strengths in alignment with industry standards.

Requirements:
{resume}
"""

resume_prompt = PromptTemplate(
    input_variables=["resume"],
    template=resume_summary_template,
)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(url_for('index'))
    
    file = request.files['file']
    
    if file.filename == '':
        return redirect(url_for('index'))
    
    if file and file.filename.endswith('.pdf'):
        try:
            # Save the uploaded file
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            # Extract text from the PDF
            resume_text = extract_text_from_pdf(file_path)
            
            # Split text and create vector store
            splitted_text = text_splitter.split_text(resume_text)
            embeddings = get_embeddings()
            vectorstore = FAISS.from_texts(splitted_text, embeddings)
            vectorstore.save_local("vector_index")
            
            # Run resume analysis using the LLM chain
            llm = get_llm()
            resume_analysis_chain = LLMChain(llm=llm, prompt=resume_prompt)
            resume_analysis = resume_analysis_chain.invoke({"resume": resume_text})
            
            # Extract text from response
            if isinstance(resume_analysis, dict):
                resume_analysis = resume_analysis.get('text', resume_analysis)
            
            # Clean up uploaded file to save space
            os.remove(file_path)
            
            return render_template('results.html', resume_analysis=resume_analysis)
        except Exception as e:
            return f"Error processing file: {str(e)}", 500
    
    return redirect(url_for('index'))


@app.route('/ask', methods=['GET', 'POST'])
def ask_query():
    if request.method == 'POST':
        query = request.form.get('query', '')
        if query:
            result = perform_qa(query)
            return render_template('qa_results.html', query=query, result=result)
    return render_template('ask.html')


@app.route('/health')
def health():
    """Health check endpoint for Render"""
    return {'status': 'healthy'}, 200


if __name__ == "__main__":
    # For local development
    app.run(debug=True)