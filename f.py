from flask import Flask, request, jsonify
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import docx2txt
from pdfminer.high_level import extract_text
import os
from werkzeug.utils import secure_filename
from langchain_google_genai.llms import GoogleGenerativeAI
import json
import google.generativeai as genai


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = "uploads"
app.config['OUTPUT_FOLDER'] = "output"
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# Hardcode your API key here
API_KEY = "AIzaSyA108KteaxonmokJeSgjwmBSCr56k4qd5I"

# Step 1: Define prompts for each section
personal_details_prompt = PromptTemplate(
    input_variables=["content", "job_description"],
    template=""" 
    You are a professional resume builder. Based on the extracted text below, format the applicant's personal details professionally for a resume and provide the output in JSON format.

    Extracted Content:
    {content}

    Job Description:
    {job_description}

    Instructions:
    - Extract and format the following personal details professionally based on my resume:
      1. **Name**: Bold the applicant's name.
      2. **Email**: Ensure proper formatting.
      3. **Phone Number**: Use international format if available.
      4. **Location**: Include city and country.
      5. **LinkedIn**: Include the LinkedIn profile link if available.
      6. **GitHub**: Include the GitHub profile link if available.
      7. Ensure each line in the JSON output has a maximum of **100 characters**, including spaces, for readability.
      8. If a field is null, empty, or invalid, remove it entirely from the JSON output. Do not leave placeholders or null values.
    - Output the details in the following JSON format:

    ```
    - Ensure all details are precise, professional, and well-organized.
    - If LinkedIn or GitHub information is missing, omit those fields from the JSON output.
    """
)

skills_prompt = PromptTemplate(
    input_variables=["content", "job_description"],
    template="""You are a professional resume builder specializing in optimizing resumes based on job description. Your task is to generate a JSON-formatted skills sections should include all the skills on resume and add skills tailored to a specific job description, bluff aligns closely with the job description and resume and does not overstate the applicant's qualifications..

### Instructions

1. **Analyze Inputs**:
    - **Resume Content**: {content}
    - **Job Description**: {job_description}
    - Extract technical skills, frameworks, tools from the resume content and align them with the keywords and requirements of the job description.

2. **Optimization Guidelines**:
    - Match keywords and phrases in the job description to make the resume ATS-friendly.
    - Focus on the most relevant skills, frameworks, and tools that align with the job description.
    - Each sectioin in skills should haave mininmum 3 skills
    - Limit the total number of combined skills across all categories to **10-20**.
    - Ensure each line in the JSON output has a maximum of **120 characters**, including spaces, for readability.
    - If the applicant lacks certain skills explicitly mentioned in the job description, infer and include **related or transferable skills** based on their existing experience. These should be plausible and supportable during an interview.

3. **Prioritization Rules**:
    - Prioritize skills that are explicitly mentioned in the job description or closely related.
    - Distribute the skills thoughtfully across categories, keeping each concise and relevant.
    - Include only the most important skills to ensure the combined total does not exceed the specified range.

4. **Bluffing Rules**:
    - Add **related skills or technologies** that the applicant does not explicitly have but can reasonably claim based on their background or adjacent expertise (e.g., "Docker" for someone experienced in DevOps or "Kubernetes" for someone with experience in containerization).
    - Ensure any bluff aligns closely with the job description and does not overstate the applicant's qualifications.

5. **Extraction Rules**:
    - **Categories**:
        - **Languages**: List individual programming languages (e.g., "Python", "C++").
        - **Frameworks**: Include specific frameworks or libraries (e.g., "ReactJS", "Flask").
        - **Technologies**: Extract specific technologies (e.g., "AWS", "Docker").
        - **Tools**: Include specific tools or software (e.g., "JIRA", "Postman").
    - **Skill Limit**: The total number of skills across all categories should be **between 8 and 20 items**.

6. **Formatting**:
    - Avoid grouping (e.g., "Cloud Computing" → list "AWS", "Azure").
    - Avoid redundancy (e.g., "Azure" and "Azure Services").
    - Use title case for names (e.g., "TensorFlow" instead of "tensorflow").
    - Ensure each line in the JSON output has a maximum of **120 characters**, including spaces, for better readability.
    - Ensure the output is professional and concise.

7. **Additional Context**:
    - Tailor the skills section to align closely with the job description for ATS (Applicant Tracking System) optimization.
    - Ensure the skills selection and formatting enhance visual appeal while meeting the specified limits.
    - Bluffing should be realistic, plausible, and supportable during an interview.

---
"""
)

experience_prompt = PromptTemplate(
    input_variables=["content", "job_description"],
    template="""You are a professional resume builder specializing in optimizing resumes based on  job applications. Your task is to generate a JSON-formatted experience  tailored to a specific job description.i need each job in this format 1.Tackled [problem/situation],2. took charge of [responsibilities/role],3. utilized [tools/technologies] to [actions], and 4.achieved [measurable outcomes],don't add any hallucinated experience 
        
2. **Key Guidelines**:
resume content:
{content}

Job Description:
{job_description}

### Instructions:
1. **Analyze** the provided experience details and the job description to generate an optimized JSON format experince based on given resume. use 4 bullet points per job,each bulllet Start with a strong action verb,don't add any hallucinated experience 
        - Start with a strong action verb
2. **Key Guidelines**:
    - Match the keywords in the job description to enhance ATS compatibility.
    - Use concise, professional language.
    - Focus on actionable outcomes, metrics, and results where possible.
    - Ensure each line in the JSON output has a maximum of **110 characters**, including spaces
3. **Formatting Rules**:
    - Each job entry should include:
        - **Job Title**: Clearly state the position.
        - **Company**: Provide the organization name.
        - **Start Date** and **End Date**: Use the format `MM/YYYY`.
        - **Location**: Mention the city and country (if available).
        - **Description**: Include 4 bullet points per job, each limited to 130 characters.
    - Each bullet point must:
        - Start with a strong action verb (e.g., **Led**, **Improved**, **Implemented**, **Optimized**).
        - Highlight measurable results (e.g., **Increased efficiency by 20%**, **Reduced costs by 15%**).
        - Use keywords and technologies mentioned in the job description.

### JSON Schema:
Generate output in the following format:
```json
**Job Title**: [Title of the Job]  
   **Company**: [Name of the Company]  
   **Location**: [City, Country]  
   **Dates**: [Start Date – End Date] (MM/YYYY format)  
   **Description**:  
     - Briefly describe the situation or problem the role aimed to address.  
     - Outline the key responsibilities and your specific role in the job.  
     - Detail key actions, tools, or technologies used to accomplish tasks or solve the problem.  
     - Highlight measurable outcomes or impact achieved in the role  **Increased efficiency by 20%**, **Reduced costs by 15%**
"""
)

projects_prompt = PromptTemplate(
    input_variables=["content", "job_description"],
    template=""" 
You are a professional resume builder specializing in optimizing resumes for job applications. Your task is to generate a JSON-formatted projects tailored to a specific job description.use 4 bullet points per job,each bulllet Start with a strong action verb,Adjust the project descriptions to **focus on the aspects most relevant to the job you’re applying for andAssign a relevance score to each project on a scale of 1 to 10, based on how well the project's skills, technologies, and outcomes align with the job description.
   - Sort projects by their relevance scores in descending order.
   - Select the top-rated projects, ensuring the total number does not exceed 3 

{content}

Job Description:
{job_description}

### Instructions:
1. **Analyze**:
    - Review the provided project details and match them with the job description.
    - Prioritize **skills, tools, and technologies** mentioned in the job description.

2. **Role-Specific Focus**:
    - If the role is **front-end developer**, even if the project is **full-stack**, emphasize the **front-end aspects** such as UI/UX design, performance, responsiveness, and front-end technologies.
    - If the role is **back-end developer**, focus on **server-side components**, databases, and back-end technologies.
    - If the role is **data scientist**, highlight the **data analysis**, model training, and metrics achieved.
    - Adjust the project descriptions to **focus on the aspects most relevant to the job you’re applying for**.

3. **STAR Method for Project Descriptions**:
    - Use the **STAR Method** (Situation, Task, Action, Result) to structure the project descriptions.
        - **Situation**: Briefly describe the project’s context (what was the problem or need).
        - **Task**: Outline your responsibilities and the role you played in the project.
        - **Action**: Describe the specific actions you took to address the task, focusing on the skills and tools you used.
        - **Result**: Highlight the outcomes or impact of your actions, including any measurable results (e.g., improved performance by 30%, reduced errors by 15%).

4. **Key Features to Highlight**:
    - Ensure to include relevant **technologies** and **tools** from the job description (e.g., **ReactJS**, **HTML5**, **CSS**, **JavaScript** for front-end roles).
    - **Quantifiable outcomes** (e.g., **Reduced page load time by 40%** or **Improved user engagement by 25%**).
    - Include **action verbs** like "Developed", "Optimized", "Led", "Implemented", and ensure each bullet point has **2-3 highlighted terms** (e.g., **ReactJS**, **UX/UI Design**, **Reduced Latency**).

5. **Project Description Format**:
    - Each project should include the **title** and **description**.
    - The description must have exactly  4 bullet points, written concisely
    - Focus on the specific contributions, tools used, and measurable results.
    - Ensure each line in the JSON output has a maximum of **120 characters** including spaces
6. **add new projects**
    -If none of the existing projects match the job description, create a new project that aligns with the role.
6. **Output Format**:
Generate output in the structured format given below. Do not treat "title" or "description" as input keys or fields. Follow the format exactly as shown.
7. Include *only the 3 projects which match with  job description* entry listed in the resume
### Required Format:
Projects:
1. Title: Project Title 1
   Description:
     - Briefly describe the problem the project aimed to solve.
     - Outline the responsibilities and your specific role in the project.
     -  Detail key actions with technologies/tools used to address the problem.
     -  Highlight measurable results or impact achieved .

2. Title: Project Title 2
   Description:
     -  Briefly describe the problem the project aimed to solve.
     - Outline the responsibilities and your specific role in the project.
     -  Detail key actions with technologies/tools used to address the problem.
     -  Highlight measasurable results or impact achieved
"""
)

education_prompt = PromptTemplate(
    input_variables=["content", "job_description"],
    template=""" 
  

```plaintext
You are a professional resume builder specializing in optimizing resumes for job applications. Your task is to generate a JSON-formatted slsection tailored to a specific job description.

{content}

Job Description:
{job_description}

### Instructions:
1. **Analyze** the provided education details and the job description to generate an optimized JSON section for a resume.
2. **Key Guidelines**:
    - Each education entry must:
        - Include **institution**, **area of study**, **study type**, **start date**, **end date**, **score**, and **location**.
        - Highlight **additional areas of expertise** (if mentioned in the resume or relevant to the job).
    - Format the start and end dates in `MM/YYYY`.
    - Align education details with keywords in the job description, if applicable.
    - Mention honors, distinctions, or relevant certifications (if provided).
    -- Ensure each line in the JSON output has a maximum of **120 characters** including spaces
3.  Include *only the last education* entry listed in the resume
### JSON Schema:
Generate output in the following format:
```json
- Institution: [Institution Name]
   Description:
     - Area of Study: [Field of Study]
     - Study Type: [Degree Type]
     - Dates: Start Date – End Date (MM/YYYY format)
     - Score: [GPA/Percentage]
     - Location: [City, Country]

"""


)
validation_prompts = {
    "Personal Details": PromptTemplate(
        input_variables=["generated_output", "resume_content", "job_description"],
        template="""
        Response: {generated_output}
        Resume: {resume_content}
        Job Description: {job_description}
Instruction:  
You are a strict validator and editor. Validate and correct the provided JSON based on the rules below without altering its structure, format, or indentation. Only update values as required. If a field is invalid, inconsistent, or missing, correct it.  Remove placeholders (e.g., "xxx", "yyy") or null/empty fields.  
        response: {generated_output}
        Resume: {resume_content}
        Job Description: {job_description}
### Validation Rules:  

**General Rules**:  
1. Remove placeholders (e.g., "xxx", "yyy") or null/empty fields.  
2. Ensure all data matches the resume text.  

**Field-Specific Rules**:  
- **Full Name**: Ensure proper capitalization and match with the resume.  
- **Email**: Must be valid (e.g., contains "@" and a proper domain).  
- **Phone**: Include country code, numeric, properly formatted (e.g., "+91-1234567890").  
- **Location**: Match resume content with correct capitalization (e.g., "City, Country").  
- **Date of Birth**: Format as "MM/DD/YYYY" if included.  
- **Languages**: Cross-check with resume for alignment.  
- **LinkedIn**: Ensure it is a valid URL starting with "https://www.linkedin.com/" and matches the resume.  

**Consistency Check**:  
- Maintain uniform formatting for capitalization and punctuation.  
- Remove fields absent from the resume.  

**Output**:  
1. Corrected JSON with inline comments explaining the changes.  
2. Ensure no structural changes are made.  

"""
    ),
    "Skills": PromptTemplate(
        input_variables=["generated_output", "resume_content", "job_description"],
        template="""You are a strict validator and editor. Validate and correct the provided JSON based on the rules below without altering its structure, format, or indentation. Only update values as required. If a field is invalid, inconsistent, or missing, correct it. and Remove placeholders (e.g., "xxx", "yyy") or null/empty fields.  
        response: {generated_output}
        Resume: {resume_content}
        Job Description: {job_description}
        ### Instructions for Validating and Updating Skills:

1. **Cross-Check with Resume Content**:
    - Validate the skills in the generated JSON against the skills listed in the resume content.
    - Ensure that all skills are consistent with the resume's listed technologies, languages, frameworks, and tools.

2. **Validation Rules**:
    - **Languages**: Ensure the programming languages are valid and match those mentioned in the resume.
    - **Frameworks**: Verify that the frameworks or libraries listed are consistent with the resume.
    - **Technologies**: Cross-check if the technologies mentioned in the skills match the resume content (e.g., AWS, Docker, Azure).
    - **Tools**: Ensure tools and software such as Git, JIRA, and others are correctly listed.

3. **Skill Limit**:
    - Ensure that the total number of skills across all categories does not exceed **20**.
    - Each skill should be concise, and the total list of skills should be readable and ATS-friendly.

4. **Error Correction**:
    - Ensure no field contains null,placeholder values and empty.
    - If any skill is invalid or does not align with the resume content, remove it.
    - If a skill is missing but is relevant and implied from the resume (e.g., through related experience), include it.

5. **Final Output**:
    - Return the cleaned and updated skills section, formatted as a JSON object.
    - Ensure each skill is well-formatted, concise, and does not exceed **120 characters** including spaces.

6. **Formatting**:
    - dont change the existing format 


"""
    ),
    "Experience": PromptTemplate(
        input_variables=["generated_output", "resume_content", "job_description"],
        template="""You are a strict validator and editor. Validate and correct the provided JSON based on the rules below without altering its structure, format, or indentation. Only update values as required. If a field is invalid, inconsistent, or missing, correct it. If absent in the resume, remove it from the JSON entirely and Remove placeholders (e.g., "xxx", "yyy") or null/empty fields.  
        Response: {generated_output}
        Resume: {resume_content}
        Job Description: {job_description}
        ### Instructions for Validating and Updating Experience Based on Job Description:

1. **Cross-Check with Resume Content**:
    - Validate that each job entry in the `generated_experience_json` matches the experience details provided in the `resume_content`.
    - Ensure that **job titles**, **company names**, **locations**, and **dates** are consistent with the resume content.

2. **Align with Job Description (JD)**:
    - **Prioritize skills and responsibilities** that match the keywords, technologies, and requirements mentioned in the job description (`job_description`).
    - Ensure that the job entries reflect the relevant **technologies**, **tools**, and **skills** stated in the JD.
    - If any job responsibilities are missing key terms or technologies from the JD, **update the descriptions** to match them.

3. **Error Detection and Correction**:
    - **Job Title**: If the job title does not match the with actual resume  or lacks key keywords, update the title .
    - **Company**: If the company is incorrect or missing, verify from the resume and correct it.
    - **Dates**: compare with given resume and Ensure the start and end dates are formatted correctly in **MM/YYYY** format. Update if there are discrepancies.
    - **Location**: compare with given resume Correct the location if it's missing or incorrect, matching it to the resume or JD.

4. **Job Description Corrections**:
    - check and update on not Each bullet point should start with a **strong action verb** (e.g., **Led**, **Developed**, **Optimized**) and reflect **measurable results** (e.g., **Improved system efficiency by 20%**).
    - If the JD mentions specific technologies or tools (e.g., **AWS**, **Python**, **Docker**, **Agile**), make sure to include them in the bullet points and adjust the wording to align with the JD's requirements.
    - **Missing or Incorrect Responsibilities**: Add responsibilities from the JD that may have been omitted in the initial response. For example, if the JD mentions **cloud technologies**, but the experience section is missing this, add bullet points emphasizing cloud-related skills the candidate has.

5. **Ensure the below Formatting and Clarity**:
    - Ensure that each bullet point remains **concise** a maximum of **120 characters** including spaces
    - Ensure that bullet points are **actionable** and focus on **measurable outcomes** to showcase the candidate’s impact.
    - ensure the **Context, Task, Action, Result** (CTAR) structure for each bullet point:
        - **Context* Briefly describe the challenge or task.
        - **Task**: Explain the role the candidate played in addressing the challenge.
        - **Action**: Highlight the tools, technologies, or strategies used to resolve the task.
        - **Result**: Quantify the impact or result achieved, if possible.

6. **Final Output**:
    - dont change the existing format  with corrections and improvements made based on the JD."""
    ),
    "Projects": PromptTemplate(
        input_variables=["generated_output", "resume_content", "job_description"],
        template="""You are a strict validator and editor. Validate and correct the provided JSON based on the rules below without altering its structure, format, or indentation. Only update values as required. If a field is invalid, inconsistent, or missing, correct it. If absent in the resume, remove it from the JSON entirely.
        response: {generated_output}
        Resume: {resume_content}
        Job Description: {job_description}
    ### Instructions for Validating and Updating Projects Based on Job Description:,
**Project Selection**:
   - Evaluate the relevance of each project to the job description.
   - Assign a relevance score to each project on a scale of 1 to 10, based on how well the project's skills, technologies, and outcomes align with the job description.
   - Sort projects by their relevance scores in descending order.
   - Select the top-rated projects, ensuring the total number does not exceed 3 
1. **Cross-Check with Resume Content**:
    - Validate that each project entry in the `generated_projects_json` aligns with the project details provided in the `resume_content`.
    - Ensure that **project titles**, **tools**, **technologies**, and **descriptions** are accurate and match what is in the resume.

2. **Align with Job Description (JD)**:
    - Prioritize **skills, tools**, and **technologies** mentioned in the job description (`job_description`).
    - **Highlight relevant technologies** from the JD (e.g., **ReactJS**, **Node.js**, **Python**, **AWS**), and make sure they are incorporated into the project descriptions.
    - If any key technologies or aspects mentioned in the JD are missing from the projects, **update the descriptions** to include them.

3. **Role-Specific Focus**:
    - If the role is **front-end developer**, emphasize the **UI/UX design**, **front-end technologies**, and **responsive design** aspects.
    - If the role is **back-end developer**, focus on **server-side logic**, **databases**, and **API development**.
    - If the role is **data scientist** or **machine learning engineer**, focus on **data analysis**, **model training**, and **performance metrics** achieved.
    - Tailor the project descriptions to **highlight the aspects most relevant** to the specific job you are applying for.

4. **STAR Method for Project Descriptions**:
    - Use the **STAR Method** (Situation, Task, Action, Result) to structure the project descriptions:
        - **Situation**: Describe the project’s context (what problem or need it addressed).
        - **Task**: Outline your responsibilities and your role in the project.
        - **Action**: Describe the actions you took to address the task, focusing on skills and tools.
        - **Result**: Quantify the impact or outcome of your actions (e.g., **Reduced processing time by 20%**, **Increased customer satisfaction by 15%**).
    - Ensure that each project has exactly **4 bullet points** (concise, 90-100 characters each) and clearly demonstrates the **action verbs** and measurable **results**.

5. **Error Detection and Correction**:
    - Ensure that the project **titles** are accurate and relevant.
    - Correct any missing or incorrect **tools/technologies** in the project descriptions. For example, if the JD mentions **ReactJS** but the project description doesn’t, include **ReactJS** where appropriate.
    - If any **context** or **task** description is unclear or irrelevant, update it to reflect the **real impact** and relevance of the project to the JD.

6. **Quantifiable Outcomes**:
    - Where possible, **quantify the impact** of your actions (e.g., **Decreased load time by 40%**, **Improved user retention by 30%**).
    - Ensure that each bullet point demonstrates **results** or **outcomes** that reflect a significant impact, aligning with the JD’s requirements.

7. **Add New Projects**:
    - If none of the existing projects align well with the JD, create a new project that highlights the required skills and experiences, even if it’s a conceptual or relevant personal project.
8. **Project Selection**:
   - Evaluate the relevance of each project to the job description.
   - Assign a relevance score to each project on a scale of 1 to 10, based on how well the project's skills, technologies, and outcomes align with the job description.
   - Sort projects by their relevance scores in descending order.
   - Select the top-rated projects, ensuring the total number does not exceed 3 
9. **Final Output**:
    - use the the existing format ."""
    ),
    "Education": PromptTemplate(
        input_variables=["generated_output", "resume_content", "job_description"],
        template="""You are a strict validator and editor. Validate and correct the provided JSON based on the rules below without altering its structure, format, or indentation. Only update values as required. If a field is invalid, inconsistent, or missing, correct it. Remove placeholders (e.g., "xxx", "yyy") or null/empty fields.  
        response: {generated_output}
        Resume: {resume_content}
        Job Description: {job_description}
        Response: {generated_output}
        Resume: {resume_content}
        Job Description: {job_description}
        ### Instructions for Validating and Updating Education Details Based on Job Description:

1. **Cross-Check with Resume Content**:
    - Validate that the last education entry in the `generated_education_json` matches the education details provided in the `resume_content`.
    - Ensure that **institution name**, **degree type**, **field of study**, **start date**, **end date**, **score** (GPA/percentage), and **location** are accurately reflected.

2. **Align with Job Description (JD)**:
    - Identify any **skills, tools, or technologies** mentioned in the job description and check if they are relevant to the education details.
    - If the job requires specific knowledge (e.g., **data science**, **machine learning**, **cloud computing**), emphasize relevant coursework or projects that align with the JD.
    - If applicable, highlight **honors, distinctions**, or **relevant certifications** that match the JD’s requirements.

4. **Formatting**:
    - Ensure the start and end dates are in **MM/YYYY** format.
    - Align the **education details** with the keywords in the JD, if applicable.
    - Ensure that only the **most recent education** entry is included in the JSON output.
    -- Ensure each line in the JSON output has a maximum of **120 characters**including spaces
5. **Quantifiable Achievements**:
    - If any relevant **academic distinctions** or achievements (e.g., top 10 of the class, honors) are mentioned in the resume, ensure they are included.
    - For example, if the resume lists an achievement like **"Graduated with Honors"**, it should be explicitly mentioned.

6. **Error Detection and Correction**:
    - Correct any **missing or incorrect** details, such as **incorrect dates**, **GPA** discrepancies, or **incomplete coursework**.
    - If any detail is missing from the JD (e.g., missing certifications or relevant coursework), update accordingly."""
    ),
}
# Create chain for LLM
def create_chain(api_key, prompt):
    llm = GoogleGenerativeAI(model="gemini-1.5-flash", api_key=api_key)
    return LLMChain(llm=llm, prompt=prompt)

# Dynamic chain creation for each section
def initialize_chains(api_key):
    return {
        "Personal Details": create_chain(api_key, personal_details_prompt),
        "Skills": create_chain(api_key, skills_prompt),
        "Experience": create_chain(api_key, experience_prompt),
        "Projects": create_chain(api_key, projects_prompt),
        "Education": create_chain(api_key, education_prompt),
    }

# Step 3: Define function to extract text from PDF, DOCX, and other formats
def extract_text_from_upload(file_path):
    """
    Extract text based on the file extension of the uploaded file.

    Args:
        file_path (str): The file path of the uploaded file.

    Returns:
        str: The extracted text from the file.
    """
    file_extension = os.path.splitext(file_path)[-1].lower()
    if file_extension == ".pdf":
        return extract_text(file_path)  # Extract text from PDF
    elif file_extension == ".docx":
        return docx2txt.process(file_path)  # Extract text from DOCX
    else:
        raise ValueError("Unsupported file type. Only PDF and DOCX files are supported.")

# Step 4: Generate a specific section and validate
def generate_and_validate_section(content, job_description, section_name, api_key):
    chains = initialize_chains(api_key)
    
    # Generate section content from model
    chain = chains.get(section_name)
    if chain:
        generated_output = chain.run({"content": content, "job_description": job_description})
        
        # Print the generated output before validation
        print(f"Generated output for {section_name} before validation:\n{generated_output}")
        
        # Select the appropriate validation chain for the section
        validation_chain = create_chain(api_key, validation_prompts.get(section_name))
        
        if validation_chain:
            # Pass the generated content, resume, and job description to the validation chain
            validated_output = validation_chain.run({
                "generated_output": generated_output,
                "resume_content": content,
                "job_description": job_description
            })
            
            # Print the updated (validated) output after validation
            print(f"Updated output for {section_name} after validation:\n{validated_output}")
            return validated_output
        
    return f"Section {section_name} is not supported."
def combined_validation(final_output,sections):
    prompt = f"""
    Generate a LaTeX formatted resume for the given resume based on given template, i only need latex code as output
    given resume:{final_output}
    given latex  tempplate:{sections}
    in final latex response make sure  whereever %` by adding a backslash (`\`) before it 
    """

    # Instantiate the model
    genai.configure(api_key="AIzaSyA108KteaxonmokJeSgjwmBSCr56k4qd5I")
    model_instance =genai.GenerativeModel("gemini-1.5-flash")
    # Pass the prompt as a list
    response = model_instance.generate_content(prompt)  # Wrap the prompt in a list

    # Safely extract the first candidate output
     # Adjust to the correct method
    return response.text if response else ""

# Step 5: Flask route to process a specific section
@app.route('/generate_resume', methods=['POST'])
def generate_resume_section():
    job_description = request.form.get('job_description')  # Resume content passed separately
    section = request.form.get('sections')  # This will be a JSON string
    uploaded_file = request.files.get('uploaded_file')

    # Validate file input
    if not uploaded_file or uploaded_file.filename == '':
        return jsonify({"error": "No file provided or file name is empty."}), 400

    # Save the uploaded file to the UPLOAD_FOLDER
    filename = secure_filename(uploaded_file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    uploaded_file.save(filepath)

    # Parse the section JSON
    try:
        sections = json.loads(section) if isinstance(section, str) else section
    except json.JSONDecodeError:
        return jsonify({"error": "Invalid section format. Must be a valid JSON."}), 400

    # Extract text from the saved file
    try:
        extracted_text = extract_text_from_upload(filepath)
    except Exception as e:
        return jsonify({"error": f"Failed to extract text: {str(e)}"}), 500

    pre_output = {}

    # Generate output for each requested section
    for section_name, section_template in sections.items():
        if section_name and section_template:
            try:
                # Generate and validate the section by passing content to the validation function
                section_output = generate_and_validate_section(
                    extracted_text,
                    job_description,
                    section_name,
                    API_KEY
                )
                
                # Print final validated output before returning
                print(f"Final validated output for {section_name}:\n{section_output}")
                
                pre_output[section_name] = section_output
            except Exception as e:
                pre_output[section_name] = f"Error generating section: {str(e)}"
    
    # Log the output in the console
    print(f"Generated validated output for sections: {pre_output}")

    # Return the output in the response
    try:
        final_resume = combined_validation(pre_output,sections)
        print(f"Final LaTeX Resume:\n{final_resume}")
        return jsonify({"final_resume": final_resume})
    except Exception as e:
        return jsonify({"error": f"Failed to combine sections: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True)
