import os
import streamlit as st
from langchain_community.chat_models import AzureChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
import fitz  # PyMuPDF for PDF text extraction

# Function to extract text from a PDF file buffer using PyMuPDF
def extract_text_from_pdf(file) -> str:
    
    try:
        doc = fitz.open(stream=file.read(), filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
        return text
    except Exception as e:
        st.error(f"Error reading PDF file: {str(e)}")
        return ""

# Function to score a single resume text against the job description text
def score_resume(llm: AzureChatOpenAI, jd_text: str, resume_text: str, resume_name: str) -> float:
    system_prompt = (
        "You are a helpful assistant that scores how well a resume matches a job description. "
        "Score on a scale of 0 to 100, where 0 means no match and 100 means perfect fit. "
        "Provide only the score as a number."
    )
    query = (
        f"Job Description:\n---\n{jd_text}\n---\n"
        f"Resume ({resume_name}):\n---\n{resume_text}\n---\n"
        "On a scale from 0 to 100, how well does this resume match the job description? "
        "Reply only with the numeric score."
    )
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=query)
    ]

    try:
        response = llm(messages)
        score_text = response.content.strip()
        score = float(score_text)
        return max(0, min(100, score))  # Clamp between 0–100
    except Exception as e:
        st.warning(f"Could not score resume '{resume_name}': {e}")
        return -1

# Streamlit app main
def main():
    st.set_page_config(page_title="Resume Ranking by Job Description", layout="centered")
    st.title("Resume Ranking by Job Description")
    st.markdown(
        """
        Upload a **Job Description** text file and up to **10 Resume PDFs**.
        The app will extract content and use Azure OpenAI to score how well each resume matches the job description.
        Resumes will be ranked by score.
        """
    )

    # Upload job description file (single text file)
    jd_file = st.file_uploader("Upload Job Description (Text file, .txt)", type=["txt"])

    # Upload resumes (PDFs)
    resume_files = st.file_uploader(
        "Upload up to 10 Resumes (PDF)", 
        type=["pdf"], 
        accept_multiple_files=True, 
        help="You can upload up to 10 PDF resumes."
    )

    if jd_file and resume_files:
        if len(resume_files) > 10:
            st.warning("Please upload no more than 10 resumes.")
            return

        # Read JD text content
        try:
            jd_text = jd_file.read().decode("utf-8")
        except Exception as e:
            st.error(f"Could not read Job Description file: {e}")
            return
        
        # Extract text from each resume pdf 
        resumes_texts = {}
        with st.spinner("Extracting text from resumes..."):
            for file in resume_files:
                text = extract_text_from_pdf(file)
                if text.strip() == "":
                    st.warning(f"No text extracted from resume: {file.name}")
                resumes_texts[file.name] = text
        
        if len(resumes_texts) == 0:
            st.error("No resume texts available to score.")
            return

        # Initialize AzureChatOpenAI LLM
        AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
        AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
        AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
        AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2023-05-15")

        if not all([AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_DEPLOYMENT]):
            st.error("Azure OpenAI environment variables are not set. Please set AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, and AZURE_OPENAI_DEPLOYMENT_NAME.")
            return

        with st.spinner("Initializing language model..."):
            llm = AzureChatOpenAI(
                azure_deployment=AZURE_OPENAI_DEPLOYMENT,
                azure_endpoint=AZURE_OPENAI_ENDPOINT,
                openai_api_key=AZURE_OPENAI_API_KEY,
                openai_api_version=AZURE_OPENAI_API_VERSION,
                temperature=0,
                max_tokens=100,
)


        # Score all resumes
        scores = []
        progress = st.progress(0)
        total = len(resumes_texts)
        for idx, (name, text) in enumerate(resumes_texts.items()):
            score = score_resume(llm, jd_text, text, name)
            scores.append({"name": name, "score": score})
            progress.progress((idx + 1) / total)

        # Sort by score descending
        scores = sorted(scores, key=lambda x: x["score"], reverse=True)

        st.subheader("Ranked Resumes")
        for i, item in enumerate(scores, start=1):
            score_display = "N/A" if item["score"] == -1 else f"{item['score']:.2f}"
            st.markdown(f"**{i}. {item['name']}** — Score: {score_display}")

if __name__ == "__main__":
    main()
