from pdfminer.high_level import extract_text
import docx2txt


import google.generativeai as genai
import json
from stqdm import stqdm

SYSTEM_PROMPT = "You have to respond in JSON only.You are a smart assistant to career advisors at the Harvard Extension School. You will reply with JSON only."

CV_TEXT_PLACEHOLDER = "<CV_TEXT>"

SYSTEM_TAILORING = """
You are a smart assistant to career advisors at the Harvard Extension School. Your take is to rewrite
resumes to be more brief and convincing according to the Resumes and Cover Letters guide.
"""

TAILORING_PROMPT = """
Consider the following CV:
<CV_TEXT>

Your task is to rewrite the given CV. Follow these guidelines:
- Make sure the resume starts with the candidates name and role
- Be truthful and objective to the experience listed in the CV
- Be specific rather than general
- Rewrite job highlight items using STAR methodology (but do not mention STAR explicitly)
- Fix spelling and grammar errors
- Writte to express not impress
- Articulate and don't be flowery
- Prefer active voice over passive voice
- Do not include a summary about the candidate
Analyze the CV or job description provided and rewrite the resume content. Ensure the following rules are followed:

---


"""

RESUME_TEMPLATE = """
-Single A4 page with all sections fitting within the page.
-Make sure the resume starts with name and personal details section
-Make sure the second section contains details of skills and technologies
-Make sure the third section contains details of previous projects
-Make sure the fourth section contains details of previous/current experiences
-Make sure the fifth section contains details of education of the candidate
-Make sure the details are generated for a single page A4 pdf
-Make sure the resume meets the ATS quality
"""

BASICS_PROMPT = """
You are going to write a JSON resume section for an applicant applying for job posts.

Consider the following CV:
<CV_TEXT>

Now consider the following TypeScript Interface for the JSON schema:

interface Basics {
    name: string;
    email: string;
    phone: string;
    website: string;
    address: string;
}

Write the basics section according to the Basic schema. On the response, include only the JSON.
"""

EDUCATION_PROMPT = """
You are going to write a JSON resume section for an applicant applying for job posts.

Consider the following CV:
<CV_TEXT>

Now consider the following TypeScript Interface for the JSON schema:

interface EducationItem {
    institution: string;
    area: string;
    additionalAreas: string[];
    studyType: string;
    startDate: string;
    endDate: string;
    score: string;
    location: string;
}

interface Education {
    education: EducationItem[];
}


Write the education section according to the Education schema. On the response, include only the JSON.
"""

AWARDS_PROMPT = """
You are going to write a JSON resume section for an applicant applying for job posts.

Consider the following CV:
<CV_TEXT>

Now consider the following TypeScript Interface for the JSON schema:

interface AwardItem {
    title: string;
    date: string;
    awarder: string;
    summary: string;
}

interface Awards {
    awards: AwardItem[];
}

Write the awards section according to the Awards schema. Include only the awards section. On the response, include only the JSON.
"""

PROJECTS_PROMPT = """
You are going to write a JSON resume section for an applicant applying for job posts.

Analyze the provided CV or job description to generate detailed, professionally formatted Projects and Work Experience sections for a resume. Follow these rules:

---

###rules for project section
1.*Word Limit*:
   - Each bullet point must be between *80–100 words*, including spaces.
2. *Actionable Language*:
    - Start each bullet point with an actionable verb (Developed, Designed, Led, Optimized, Analyzed, etc.).
    - Use professional language to describe contributions and outcomes.

3. *Highlight(make the word or mesurable as boldKey )Information*:
    - Some points should include quantifiable data (e.g., "Reduced processing time by 50%", "Increased revenue by 15%").
    - Other points can focus on general responsibilities or tasks, such as tools or methodologies used.

4. *Structure*:
    - Each project entry should have a *title* and a *list of bullets* (maximum 4–5 points per project).


5. *Quantifiable Data*:
    - Some points should include measurable results example:Decreased operational costs by 25%"
    - Other points should focus on general contributions (e.g., "Collaborated with a team to deliver a key project").


Consider the following CV:
<CV_TEXT>

Now consider the following TypeScript Interface for the JSON schema:

interface ProjectItem {
    name: string;
    startDate: string;
    endDate: string;
    description: string;
    keywords: string[];
    url: string;
}

interface Projects {
    projects: ProjectItem[];
}

Write the projects section according to the Projects schema. Include all projects, but only the ones present in the CV. On the response, include only the JSON.
"""

SKILLS_PROMPT = """ Your task is to extract relevant technologies, programming languages, frameworks, tools, and other technical skills from the CV provided (<CV_TEXT>). Ensure that the JSON output is professional, specific, and formatted for inclusion in a resume. Use the following rules:

### *Rules for Extraction*

1. *Categories*:
    - Languages: Include individual programming languages, listed specifically (e.g., "Python", "C++").
    - Frameworks: Include specific frameworks or libraries mentioned in the CV (e.g., "ReactJS", "Flask").
    - Technologies: Include specific technologies (e.g., "Azure", "AWS") and avoid grouping them under broad terms like "Cloud Computing."
    - Tools: Include specific tools and software used (e.g., "JIRA", "Jupyter Notebook").

2. *Formatting Rules*:
    - Do not group items into generic categories like "Cloud Computing" or "Profiling Tools."
    - Separate individual technologies (e.g., list "Azure" and "AWS" instead of "Cloud Computing (Azure, AWS)").
    - The output must be clear, concise, and structured for professional use.
    - Include a maximum of *6 items per category* to maintain brevity.

3. *Response Format*:
    Output must be in the following JSON format:

    json
    {
        "Languages": ["<Language1>", "<Language2>", ...],
        "Frameworks": ["<Framework1>", "<Framework2>", ...],
        "Technologies": ["<Technology1>", "<Technology2>", ...],
        "Tools": ["<Tool1>", "<Tool2>", ...]
    }
    

4. *Additional Guidance*:
    - Ensure the output is professional and polished.
    - Avoid redundancy (e.g., listing "Azure" and "Azure Services").
    - Use title case for proper nouns (e.g., "ReactJS" instead of "reactjs").

---

"""

WORK_PROMPT = """
You are going to write a JSON resume section for an applicant applying for job posts.
Analyze the provided CV or job description to generate detailed, professionally formatted Projects and Work Experience sections for a resume. Follow these rules:

---

### *Rules for works Section*:
1.*Word Limit*:
   - Each bullet point must be between *80–100 words*, including spaces.
2. *Actionable Language*:
    - Start each bullet point with an actionable verb (Developed, Designed, Led, Optimized, Analyzed, etc.).
    - 

3. *highlight(make it as bold)Key Information*:
    -Highlight key contributions and measurable outcomes. hInclude specific, critical details (e.g., tools, technologies, methodologies, outcomes).
4. *Add *quantifiable data* (e.g., "Reduced processing time by 50%", "Increased sales by 20%") for at least *50% of points*.



Consider the following CV:
<CV_TEXT>

Now consider the following TypeScript Interface for the JSON schema:

interface WorkItem {
    company: string;
    position: string;
    startDate: string;
    endDate: string;
    location: string;
    highlights: string[];
}

interface Work {
    work: WorkItem[];
}

Write a work section for the candidate according to the Work schema. Include only the work experience and not the project experience. For each work experience, provide  a company name, position name, start and end date, and bullet point for the highlights. Follow the Harvard Extension School Resume guidelines and phrase the highlights with the STAR methodology
"""

def extract_text_from_pdf(file):
    return extract_text(file)


def extract_text_from_docx(file):
    return docx2txt.process(file)


def extract_text_from_upload(file):
    if file.endswith(".pdf"):
        text = extract_text_from_pdf(file)
        return text
    elif (
        file.endswith(".document")
        
    ):
        text = extract_text_from_docx(file)
        return text
    elif file.endswith("json"):
        return file.getvalue().decode("utf-8")
    else:
        return file.getvalue().decode("utf-8")

def generate_section_content(section_name, extracted_text, template, job_description, model,api_key):
    """
    Generate tailored LaTeX content for a section using the LLM.
    """
    prompt = f"""
    Generate a LaTeX formatted resume section for {section_name} based on the following:
    - Extracted content: {extracted_text}
    - Section template: {template}
    - Job description: {job_description}
    - Each point should not exceed 100 letters
    -Single A4 page with all sections fitting within the page.
    - Make sure that the the details filled in the template without changing any syntax and the number of bullet points should always be same
    - The details should be modified enhanced to match 100 percent to the job description with little bluffing and adding additional content
    - Each data should not be precisley within a single line
    The output should only include LaTeX code for the section, formatted professionally.
    """
    genai.configure(api_key=api_key)
    model_instance = genai.GenerativeModel(model)
    response = model_instance.generate_content(prompt)
    return response.text if response else ""



def tailor_resume(cv_text, api_key, model,job_description):
    """Tailor a resume using Google's Generative AI"""
    try:
        # Configure the generative AI with the provided API key
        genai.configure(api_key=api_key)
        model_instance = genai.GenerativeModel(model)

        # Prepare the tailored prompt
        combined_prompt = f"""
        - Single A4 page with all sections fitting within the page.
        Using the provided job description and resume content, generate a revised resume that is fully optimized for the specific role. The tailored resume should:
        1. First, display personal details (name, contact, LinkedIn) followed by a "Technologies" section showcasing the relevant skills. This should appear first.
        2. Then, present the "Experience" section, followed by any projects listed in the CV.
        3. Ensure that sections such as Skills and Experience are formatted correctly in LaTeX.
        4. Highlight skills, achievements, and experiences directly relevant to the job description.
        5. Use language aligned with the keywords present in the job description to improve ATS compatibility.
        6. Include hypothetical or inferred project examples that showcase the application of required skills and technologies.
        7. Avoid providing templates or generic instructions. Ensure professional formatting and accurate grammar.
        8. Ensure the final output is concise, polished, and immediately usable for the job application.
        

        Highlight skills, achievements, and experiences directly relevant to the job description.
        Use language aligned with the keywords present in the job description to improve ATS compatibility.
        Include hypothetical or inferred project examples that showcase the application of required skills and technologies.
        Require no manual corrections, ensuring professional formatting and accurate grammar.
        Avoid providing templates or generic instructions.
        Ensure the final output is concise, polished, and immediately usable for the job application.
        Job Description and required qualifications:
        {job_description}

        Resume Template:
        {RESUME_TEMPLATE}

        Original Resume Content:
        {cv_text}
    ### *General Rules:*
1. *Ideal Resume Length*:
    - Limit the resume to 450–850 words.

2. *Avoid Buzzwords*:
    - Eliminate generic buzzwords like "Hardworking," "Team Player," "Strategic Thinker," or "Dynamic."
    - Replace these with measurable achievements and concrete, action-driven statements.

3. *Incorporate Professional Action Verbs*:
    - Use precise, impactful action verbs for specific skills or achievements, categorized as follows:
    
    - *Leadership Words*:
      Empowered, Directed, Guided, Championed, Led, Inspired, Mobilized, Spearheaded.

    - *Communication Words*:
      Conveyed, Facilitated, Negotiated, Persuaded, Presented.

    - *Problem-Solving Words*:
      Analyzed, Diagnosed, Executed, Assessed, Evaluated, Investigated, Resolved, Formulated, Identified.

    - *Drive and Initiative Words*:
      Spearheaded, Pioneered, Innovated, Revitalized, Drove, Galvanized.

    - *Adaptability and Flexibility Words*:
      Adapted, Embraced, Pivoted, Navigated, Balanced, Integrated.

    - *Time and Project Management Words*:
      Coordinated, Prioritized, Scheduled, Streamlined, Managed, Delegated.

        """

        # Generate tailored resume content
        response = model_instance.generate_content(combined_prompt)
        return response.text
    except Exception as e:
        print(f"Failed to tailor resume: {e}")
        return cv_text

