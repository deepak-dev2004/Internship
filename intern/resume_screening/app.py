import streamlit as st
from utils import extract_text_from_pdf, clean_text, calculate_similarity, extract_skills

# Load skills
with open("skills.txt", "r") as f:
    skills = [line.strip() for line in f.readlines()]

st.title("📄 Resume Screening System")

# Upload resume
uploaded_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])

# Job description
job_desc = st.text_area("Enter Job Description")

if st.button("Analyze Resume"):

    if uploaded_file and job_desc:

        # Extract resume text
        resume_text = extract_text_from_pdf(uploaded_file)

        # Clean text
        resume_clean = clean_text(resume_text)
        job_clean = clean_text(job_desc)

        # Similarity score
        score = calculate_similarity(resume_clean, job_clean)

        # Extract skills
        resume_skills = extract_skills(resume_clean, skills)

        # Output
        st.subheader("📊 Match Score")
        st.success(f"{score}% Match")

        st.subheader("🛠 Extracted Skills")
        st.write(resume_skills)

        # Decision
        if score > 60:
            st.success("✅ Good Match - Shortlist Candidate")
        else:
            st.warning("❌ Low Match - Not Suitable")

    else:
        st.error("Please upload resume and enter job description")