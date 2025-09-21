from flask import Flask, request, render_template, redirect, url_for
import os
import tempfile
from werkzeug.utils import secure_filename
import PyPDF2
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter
from dotenv import load_dotenv  

load_dotenv()  

# Global variable to store vector database path
vector_store_path = None

text_splitter = CharacterTextSplitter(
    separator='\n',
    chunk_size=2000,
    chunk_overlap=200,
    length_function=len,
)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def perform_qa(query):
    global vector_store_path
    try:
        if vector_store_path and os.path.exists(vector_store_path):
            db = FAISS.load_local(vector_store_path, embeddings, allow_dangerous_deserialization=True)
            retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 4})
            rqa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)
            result = rqa.invoke(query)
            return result['result']
        else:
            return "Please upload a resume first before asking questions."
    except Exception as e:
        return f"Error processing query: {str(e)}"

app = Flask(__name__)

def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page_num in range(len(reader.pages)):
            text += reader.pages[page_num].extract_text()
    return text

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash-latest",
    temperature=0,
    google_api_key=os.environ.get("GOOGLE_API_KEY")
)

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

resume_analysis_chain = LLMChain(
    llm=llm,
    prompt=resume_prompt,
)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    global vector_store_path
    
    if 'file' not in request.files:
        return redirect(url_for('index'))
    
    file = request.files['file']
    
    if file.filename == '':
        return redirect(url_for('index'))
    
    if file:
        # Use temporary file instead of permanent storage
        temp_dir = tempfile.gettempdir()
        filename = secure_filename(file.filename)
        file_path = os.path.join(temp_dir, filename)
        file.save(file_path)
        
        try:
            # Extract text from the PDF
            resume_text = extract_text_from_pdf(file_path)
            splitted_text = text_splitter.split_text(resume_text)
            
            # Create vector store in temp directory
            vectorstore = FAISS.from_texts(splitted_text, embeddings)
            vector_store_path = os.path.join(temp_dir, "vector_index")
            vectorstore.save_local(vector_store_path)
            
            # Run resume analysis
            resume_analysis = resume_analysis_chain.run(resume=resume_text)
            
            return render_template('results.html', resume_analysis=resume_analysis)
            
        except Exception as e:
            print(f"Error processing file: {e}")
            return redirect(url_for('index'))
        finally:
            # Clean up uploaded file
            if os.path.exists(file_path):
                os.remove(file_path)

@app.route('/ask', methods=['GET', 'POST'])
def ask_query():
    if request.method == 'POST':
        query = request.form['query']
        result = perform_qa(query)
        return render_template('qa_results.html', query=query, result=result)
    return render_template('ask.html')

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)