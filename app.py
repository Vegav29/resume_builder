from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename
import os
import google.generativeai as genai
from utils import extract_text_from_upload, generate_section_content
import json
app = Flask(__name__)

# Configure upload folder
app.config['UPLOAD_FOLDER'] = "uploads"
app.config['OUTPUT_FOLDER'] = "output"
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

@app.route('/generate_resume', methods=['POST'])
def generate_resume():
    try:
        # Parse form-data and JSON
        api_key = request.form.get('api_key')
        job_description = request.form.get('job_description')
        genai_model = request.form.get('genai_model', "gemini-1.5-pro")
        sections = request.form.get('sections')  # This will be a JSON string
        uploaded_file = request.files.get('resume_file')

        if not api_key:
            return jsonify({"error": "API key is required"}), 400
        if not uploaded_file:
            return jsonify({"error": "Resume file is required"}), 400
        if not sections:
            return jsonify({"error": "Sections and templates are required"}), 400

        # Convert sections JSON string to dictionary
        print("-------")
        try:
            sections = json.loads(sections) if isinstance(sections, str) else sections
        except json.JSONDecodeError as e:
            return jsonify({"error": f"Invalid JSON for sections: {str(e)}"}), 400
        print("------")
        # Save the uploaded file
        filename = secure_filename(uploaded_file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        uploaded_file.save(filepath)
        print("======")
        # Configure Google GenAI
        genai.configure(api_key=api_key)

        # Extract text from the uploaded resume
        extracted_text = extract_text_from_upload(filepath)
        print("-----------------")
        # Generate LaTeX content for all sections
        final_tex = ""
        for section, template in sections.items():
            if template:
                generated_tex = generate_section_content(
                    section,
                    extracted_text,
                    template,
                    job_description,
                    genai_model,
                    api_key
                )
                final_tex += generated_tex + "\n"
        print("----------------")
        # Save the LaTeX content to a file
        tex_filename = "generated_resume.tex"
        tex_filepath = os.path.join(app.config['OUTPUT_FOLDER'], tex_filename)
        with open(tex_filepath, "w", encoding="utf-8") as tex_file:
            tex_file.write(final_tex)

        # Send the LaTeX file as a response
        return send_file(tex_filepath, as_attachment=True)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
