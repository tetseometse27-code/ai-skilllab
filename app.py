import streamlit as st
import os
import json
import pdfplumber
import sqlite3
from dotenv import load_dotenv
from openai import OpenAI
from io import BytesIO
import fitz
import datetime

# OCR disabled for deployment stability
from PIL import Image

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm

# pytesseract.pytesseract.tesseract_cmd = r"D:\Tesseract OCR\tesseract.exe"
# POPPLER_PATH = r"D:\Release-25.12.0-0\poppler-25.12.0\Library\bin"

# Load environment variables
load_dotenv()

# ---- MSFC CURRICULUM STRUCTURE ----

MSFC_CURRICULUM = {

    9: {

        "Unit 1: Workshop & Engineering Techniques": [
            "Session 1: Introduction and safe practice of tools in engineering workshop",
            "Session 2: Measurement – measuring various physical quantities",
            "Session 3: Carpentry",
            "Session 4: Soldering and fabrication (making jobs from sheet metal)",
            "Session 5: Drilling, tapping and threading",
            "Session 6: Welding",
            "Session 7: Study of construction material",
            "Session 8: Piping and plumbing (simple pipeline work)"
        ],

        "Unit 2: Energy & Environment": [
            "Session 1: Introduction to electric appliances, tools and symbols",
            "Session 2: Types of wire, cables & switch",
            "Session 3: Joints of electrical conductor wires",
            "Session 4: Simple wiring",
            "Session 5: Types of wiring – staircase wiring, godown wiring",
            "Session 6: Earthing",
            "Session 7: Types and function of fuse",
            "Session 8: Miniature circuit breaker (MCB)",
            "Session 9: Soldering",
            "Session 10: Maintenance of lead acid battery",
            "Session 11: Maintenance and application of various types of stoves",
            "Session 12: Types of light",
            "Session 13: Generate electricity bill and measures to save electricity",
            "Session 14: Soak pit / septic tank – purpose and operating system",
            "Session 15: Waste management and regeneration"
        ],

        "Unit 3: Gardening, Nursery and Agricultural Techniques": [
            "Session 1: Machines and equipments for agriculture",
            "Session 2: Land cultivation, crop plantation, fertilizer application, mulching",
            "Session 3: Seed plantation and seed treatment",
            "Session 4: Prepare vermi compost and vermiwash",
            "Session 5: Prepare organic pesticide",
            "Session 6: Methods of determining the weight and age of animals",
            "Session 7: Type of the animal feed",
            "Session 8: Domesticated animals, diseases & care",
            "Session 9: Innovative gardening"
        ],

        "Unit 4: Food Processing Techniques": [
            "Session 1: Utensils and equipment used in cooking",
            "Session 2: Characteristics of raw food material",
            "Session 3: Food processing methods",
            "Session 4: Food preservation methods",
            "Session 5: Costing, packing and labelling of food products",
            "Session 6: Food and nutrition requirements of adolescent boys and girls",
            "Session 7: Methods of identifying food adulteration",
            "Session 8: Flow chart"
        ]
    },

    10: {

        "Unit 1: Workshop & Engineering Techniques": [
            "Session 1: Introduction of Engineering drawing instruments",
            "Session 2: Engineering drawing (Orthographic & Isometric projection)",
            "Session 3: Safety precautions in Engineering workshop",
            "Session 4: Types of GI pipe fittings",
            "Session 5: Welding technique & Welding joint test",
            "Session 6: Basic techniques in building construction",
            "Session 7: Making of RCC column",
            "Session 8: Costing of Construction",
            "Session 9: Plastering and Painting"
        ],

        "Unit 2: Energy & Environment": [
            "Session 1: Introduction to Electrical techniques and practices",
            "Session 2: Introduction of Electric Pump, DOL starter, and Inverter",
            "Session 3: Solar energy",
            "Session 4: Demonstrate the functioning of Petrol or Diesel engine",
            "Session 5: Bio gas concept and use",
            "Session 6: Rain water harvesting",
            "Session 7: Rainfall measurement method"
        ],

        "Unit 3: Gardening, Nursery and Agricultural Techniques": [
            "Session 1: Nursery technique",
            "Session 2: Irrigation & Water conservation methods",
            "Session 3: Interpreting result of Soil testing",
            "Session 4: Artificial Insemination",
            "Session 5: Prepare fodder for animals"
        ],

        "Unit 4: Personal Health and Hygiene": [
            "Session 1: Balanced diet",
            "Session 2: Personal health & hygiene and Community health",
            "Session 3: Communicable & Non-communicable diseases",
            "Session 4: Blood & blood group-basic information",
            "Session 5: Community health & Environment care",
            "Session 6: Pollution – Sources, Effects and Solutions",
            "Session 7: Handling of food products"
        ]
    }
}

# ---- PAGE CONFIG ----
st.set_page_config(page_title="AI SkillLab", page_icon="🤖", layout="wide")

# ---- APP NAVIGATION ----

page = st.sidebar.radio(
    "Navigation",
    ["Create Lesson", "Teacher Dashboard"]
)

# ---- API KEY ----
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

# ---- DATABASE SETUP ----

def init_db():
    conn = sqlite3.connect("lesson_history.db")
    c = conn.cursor()

    c.execute("""
    CREATE TABLE IF NOT EXISTS lessons (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        date TEXT,
        class_level INTEGER,
        subject TEXT,
        unit TEXT,
        session TEXT,
        objective TEXT,
        lesson_json TEXT,
        created_at TEXT
    )
    """)

    conn.commit()
    conn.close()

init_db()

# ---- PDF GENERATOR FUNCTION ----
def generate_pdf(data):

    doc = fitz.open("lesson_template_fillable.pdf")

    for page in doc:
        widgets = page.widgets()

        if widgets:
            for widget in widgets:
                field_name = widget.field_name

                if field_name in data:
                    widget.field_value = str(data[field_name])
                    widget.update()

    # -------- FLATTEN PDF --------
    flattened_doc = fitz.open()

    for page in doc:
        pix = page.get_pixmap(matrix=fitz.Matrix(3, 3))  # high quality
        new_page = flattened_doc.new_page(
            width=page.rect.width,
            height=page.rect.height
        )
        new_page.insert_image(page.rect, pixmap=pix)

    buffer = BytesIO()
    flattened_doc.save(buffer)
    buffer.seek(0)

    return buffer

# ---- WORKSHEET PDF GENERATOR FUNCTION ----
def generate_worksheet_pdf(text):

    buffer = BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=10*mm,
        rightMargin=10*mm,
        topMargin=10*mm,
        bottomMargin=10*mm
    )

    styles = getSampleStyleSheet()

    # Smaller font
    normal_style = ParagraphStyle(
        'CustomNormal',
        parent=styles['Normal'],
        fontSize=9,
        leading=11
    )

    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Title'],
        fontSize=14,
        leading=16
    )

    content = []

    # Title
    content.append(Paragraph("<b>Student Worksheet</b>", title_style))
    content.append(Spacer(1, 6))

    # Content
    for line in text.split("\n"):
        if line.strip():
            content.append(Paragraph(line, normal_style))
            content.append(Spacer(1, 4))

    doc.build(content)

    buffer.seek(0)
    return buffer

# ---- TEXT EXTRACTION WITH OCR ----
def extract_text_from_file(uploaded_file):

    extracted_text = ""

    file_name = uploaded_file.name.lower()

    if file_name.endswith(".pdf"):

        with pdfplumber.open(uploaded_file) as pdf:

            for page_number, page in enumerate(pdf.pages):

                text = page.extract_text()

                if text and len(text.strip()) > 30:
                    extracted_text += text + "\n"
                
    elif file_name.endswith((".png", ".jpg", ".jpeg")):
        extracted_text = "Image text extraction not supported in online version."

    return extracted_text

# ---- SMART SECTION DETECTION ----
def split_text_into_sections(text):

    import re

    sections = []
    current_section = ""

    lines = text.split("\n")

    for line in lines:

        line = line.strip()

        if not line:
            continue

        if len(line) < 80 and line.isupper():

            if current_section:
                sections.append(current_section.strip())

            current_section = line + "\n"

        else:
            current_section += line + "\n"

    if current_section:
        sections.append(current_section.strip())

    return sections


# ---- TEXT CHUNKING FUNCTION (PARAGRAPH BASED) ----
def split_text_into_chunks(text, max_chunk_chars=1200):

    import re

    paragraphs = re.split(r"\n\s*\n", text)

    chunks = []
    current_chunk = ""

    for para in paragraphs:

        para = para.strip()

        if not para:
            continue

        if len(current_chunk) + len(para) < max_chunk_chars:
            current_chunk += para + "\n\n"

        else:
            chunks.append(current_chunk.strip())
            current_chunk = para + "\n\n"

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


# ---- SMART RETRIEVAL FUNCTION (MINI RAG) ----
def retrieve_relevant_chunks(topic, text_chunks, top_k=5):

    if not text_chunks:
        return ""

    documents = text_chunks + [topic]

    vectorizer = TfidfVectorizer().fit_transform(documents)
    vectors = vectorizer.toarray()

    topic_vector = vectors[-1]
    chunk_vectors = vectors[:-1]

    similarities = cosine_similarity([topic_vector], chunk_vectors)[0]

    top_indices = similarities.argsort()[-top_k:][::-1]

    relevant_chunks = [text_chunks[i] for i in top_indices]

    return "\n".join(relevant_chunks)

# ---- DETECT UNITS AND SESSIONS ----
import re


def detect_units_and_sessions(text):

    lines = text.split("\n")

    structure = {}
    current_unit = None
    current_session = None

    for line in lines:

        line = line.strip()

        if not line:
            continue

        unit_match = re.match(r"UNIT\s+\d+", line, re.IGNORECASE)

        if unit_match:
            current_unit = line
            structure[current_unit] = {}
            current_session = None
            continue

        session_match = re.match(r"SESSION\s+\d+", line, re.IGNORECASE)

        if session_match and current_unit:
            current_session = line
            structure[current_unit][current_session] = ""
            continue

        if current_unit and current_session:
            structure[current_unit][current_session] += line + " "

    return structure

# ---- SAVE LESSON FUNCTION ----

def save_lesson(date, class_level, subject, unit, session, objective, lesson_data):

    conn = sqlite3.connect("lesson_history.db")
    c = conn.cursor()

    c.execute("""
    INSERT INTO lessons
    (date, class_level, subject, unit, session, objective, lesson_json, created_at)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        str(date),
        class_level,
        subject,
        unit,
        session,
        objective,
        json.dumps(lesson_data),
        str(datetime.datetime.now())
    ))

    conn.commit()
    conn.close()

if page == "Create Lesson":

    # EVERYTHING BELOW MUST BE INSIDE THIS BLOCK
    st.title("🤖 AI SkillLab – The AI Co-Teacher for Vocational Education")
    st.write("Generate lesson plans, worksheets, and quizzes instantly using AI.")

    # ---- CLASS SELECTOR (OUTSIDE FORM) ----

    with st.container(border=True):

        st.markdown("## Step 1 — Class & Session Information")

        col1, col2 = st.columns([1,1])

        with col1:
            lesson_date = st.date_input(
                "Date of Lesson",
                value=datetime.date.today(),
                format="DD/MM/YYYY"
            )

        with col2:
            class_level = st.radio(
                "Select Class",
                [9,10],
                horizontal=True
            )

        col3, col4, col5 = st.columns(3)

        with col3:
            duration = st.number_input(
                "Duration of session (minutes)",
                min_value=30,
                max_value=180,
                step=5
            )

        with col4:
            boys = st.number_input(
                "Number of Boys",
                min_value=0,
                step=1
            )

        with col5:
            girls = st.number_input(
                "Number of Girls",
                min_value=0,
                step=1
            )

        # ---- RESET SUBJECT & UNIT WHEN CLASS CHANGES ----

        if "previous_class" not in st.session_state:
            st.session_state.previous_class = class_level

        if st.session_state.previous_class != class_level:
            st.session_state.subject = "Select Subject"
            st.session_state.previous_class = class_level
        generate = False

        # ---- SUBJECT / UNIT / SESSION SELECTOR ----

        st.markdown("## Step 2 — Select Subject and Session")

        with st.container(border=True):

            subject = st.selectbox(
                "Select Vocational Subject",
                [
                    "Select Subject",
                    "Multi Skill Foundation Course (MSFC)",
                    "Tourism & Hospitality",
                    "Retail",
                    "Electronics",
                    "Agriculture",
                    "Music",
                    "Automotive",
                    "Finance",
                    "IT / ITES"
                ],
                key="subject"
            )

            units = ["Select Unit"] + list(MSFC_CURRICULUM[class_level].keys())

            selected_unit = st.selectbox(
                "Select Unit",
                units,
                key="unit_selectbox"
            )

            st.session_state["selected_unit"] = selected_unit

            # ---- SESSION DROPDOWN BASED ON UNIT ----

            if selected_unit == "Select Unit":
                sessions = ["Select Session"]
            else:
                sessions = ["Select Session"] + MSFC_CURRICULUM[class_level][selected_unit]

            session = st.selectbox(
                "Select Session",
                sessions,
                key="session_selectbox"
            )

            st.session_state["selected_session"] = session

            include_employability = st.checkbox(
                "Integrate Employability Skills (Communication, ICT, Entrepreneurship, Green Skills)",
                value=True
            )

            objective = st.text_area(
                "Skill Objective",
                placeholder="What should students be able to do by the end of this session?"
            )

            uploaded_file = st.file_uploader(
                "Upload Textbook Chapter (PDF / Image)",
                type=["pdf", "png", "jpg", "jpeg"]
            )

            # ---- PROCESS TEXTBOOK ----

            textbook_text = ""
            textbook_structure = {}

            if uploaded_file is not None:

                textbook_text = extract_text_from_file(uploaded_file)

                textbook_structure = detect_units_and_sessions(textbook_text)

        # ---- AUTO DETECT PREVIOUS SESSION ----

        auto_prev_unit = "Select Previous Unit"
        auto_prev_session = "Select Previous Session"

        # ✅ USE SESSION STATE VALUES (FIXED)
        current_unit = st.session_state.get("selected_unit", "Select Unit")
        current_session = st.session_state.get("selected_session", "Select Session")

        # ✅ TRACK LAST SESSION
        if "last_session" not in st.session_state:
            st.session_state["last_session"] = None

        if current_unit != "Select Unit" and current_session != "Select Session":

            session_changed = st.session_state["last_session"] != current_session

            unit_keys = list(MSFC_CURRICULUM[class_level].keys())
            current_unit_index = unit_keys.index(current_unit)

            session_list = MSFC_CURRICULUM[class_level][current_unit]
            current_session_index = session_list.index(current_session)

            # Case 1: Not first session
            if current_session_index > 0:
                auto_prev_unit = current_unit
                auto_prev_session = session_list[current_session_index - 1]

            # Case 2
            elif current_unit_index > 0:
                prev_unit = unit_keys[current_unit_index - 1]
                prev_sessions = MSFC_CURRICULUM[class_level][prev_unit]

                auto_prev_unit = prev_unit
                auto_prev_session = prev_sessions[-1]

            # ✅ APPLY ONLY IF SESSION CHANGED
            if session_changed:
                st.session_state["previous_unit"] = auto_prev_unit
                st.session_state["previous_session"] = auto_prev_session
                st.session_state["last_session"] = current_session

        # ---- PREVIOUS CLASS (REVISION) ----

        st.markdown("### Step 3 — Previous Class (for Revision)")

        with st.container(border=True):

            prev_units = ["Select Previous Unit"] + list(MSFC_CURRICULUM[class_level].keys())

            previous_unit = st.selectbox(
                "Previous Unit",
                prev_units,
                key="previous_unit"
            )

            if previous_unit == "Select Previous Unit":
                prev_sessions = ["Select Previous Session"]
            else:
                prev_sessions = ["Select Previous Session"] + MSFC_CURRICULUM[class_level][previous_unit]

            previous_session = st.selectbox(
                "Previous Session",
                prev_sessions,
                key="previous_session"
            )

        # ---- FORM STARTS ----

        with st.form("lesson_form"):

            generate = st.form_submit_button("Generate Teaching Materials")

        st.session_state["generate"] = generate

    # ---- UNIT / SESSION SELECTOR (DISABLED - USING CURRICULUM STRUCTURE INSTEAD) ----
    # This section is disabled because Unit and Session are now controlled
    # by the MSFC_CURRICULUM dictionary above.

    # if textbook_structure:
    #
    #     units = list(textbook_structure.keys())
    #
    #     selected_unit = st.selectbox("Select Unit", units)
    #
    #     sessions = list(textbook_structure[selected_unit].keys())
    #
    #     selected_session = st.selectbox("Select Session", sessions)
    #
    #     textbook_text = textbook_structure[selected_unit][selected_session]
        # ---- DEBUG PREVIEW PANEL ----
        with st.expander("Preview Extracted Text (Debug)"):

            st.write("Character count:", len(textbook_text))
            st.text_area("Extracted Text", textbook_text, height=300)

    # Set retrieval query to current lesson topic
    retrieval_query = f"{subject} {session}"

    # If chapter-sized text → send directly
    if len(textbook_text) < 12000:
        textbook_text = textbook_text

    else:
        # Try section detection first
        sections = split_text_into_sections(textbook_text)

        if len(sections) > 1:
            textbook_text = retrieve_relevant_chunks(retrieval_query, sections)

        else:
            # If section detection fails → use paragraph chunking
            chunks = split_text_into_chunks(textbook_text)
            textbook_text = retrieve_relevant_chunks(retrieval_query, chunks)

    # Limit textbook size to prevent token overflow
            textbook_text = textbook_text[:5000]

            if textbook_text:
                with st.expander("Preview extracted textbook content"):
                    st.write(textbook_text[:1000])

    # Store button state in session
    if "generate" not in st.session_state:
        st.session_state["generate"] = False

    if st.session_state.get("generate") or "lesson_data" in st.session_state:

        if st.session_state.get("generate"):
            with st.spinner("AI is generating teaching materials..."):

                # LESSON PLAN PROMPT
                lesson_prompt = f"""
                You are an experienced vocational teacher preparing a lesson plan.

                The lesson should be primarily based on the provided textbook content.
                Use the textbook as the main reference.

                If helpful, you may add simple explanations for clarity,
                but do not introduce concepts not mentioned in the textbook.

                TEXTBOOK CONTENT:
                -----------------
                {textbook_text}

                TOPIC CONTROL RULES
                -------------------

    Focus on the specific lesson topic or session: {session}.

    The lesson plan must stay within the scope of the provided
    textbook content and the lesson topic.

    If the textbook section contains multiple subtopics,
    explain only those that belong to the current lesson topic.

    Do not expand into unrelated chapters or additional syllabus
    topics that are not present in the provided textbook section.

    Always prioritize the textbook content over general knowledge.
    Use the textbook as the primary source of truth.

    IMPORTANT INSTRUCTION ABOUT SYLLABUS
    ------------------------------------

    If the textbook lists specific items (for example types of welding),
    limit the explanation primarily to those items.

    If the textbook only mentions a small number of examples,
    explain those clearly.

    You may briefly mention that other examples exist in the real world
    (for example other types used in industry), but clearly tell students
    that for their syllabus or exam preparation they only need to focus
    on the examples mentioned in the textbook.

    LESSON DETAILS
    --------------
    Class: {class_level}
    Subject: {subject}
    Session: {session}
    Previous Class Topic: {previous_session}
    Duration: {duration} minutes
    Skill Objective: {objective}

    Include employability skills where appropriate if this setting is TRUE: {include_employability}

    If Employability Skills Integration is TRUE:
    - integrate communication skills
    - self-management
    - entrepreneurship
    - green skills
    - ICT skills

    Integrate them naturally during:
    • group discussion
    • activity
    • reflection
    • classroom interaction

    However the main focus must remain on the vocational subject session.

    Return the response strictly in JSON format like this:

    {{
    "knowledge": "Short 1-2 line summary of the key concept.",
    "what": "Explain what the students need to know about the topic.",
    "why": "Explain why learning this topic is practically important.",
    "objective": "",
    "tools": "",
    "raw_material": "",
    "safety": "",
    "grouping": "",
    "task_distribution": "",
    "sitting": "",
    "revision": "",
    "opening": "",
    "activity": "",
    "closing": "",
    "assessment": "",
    "homework": ""
    }}

    The lesson must be appropriate for Class 9 or Class 10 vocational students.

    Knowledge Key Points:
    - What - Explain the key concept students must understand.
    - Why - Explain why the topic is practically important.

    What:

    First write a short paragraph explaining the concept clearly in about three sentences.

    After the paragraph, write 3-4 discussion questions.
    Each question must begin with the word "What".

    These questions should help students think about the concept
    and encourage classroom discussion.

    Why:

    First write a short paragraph explaining why this concept is
    important in real-life work or industry.

    After the paragraph, write 3-4 discussion questions.
    Each question must begin with the word "Why".

    These questions should encourage students to think about
    the importance and practical value of the topic.

    Objective:

    Write the learning objectives using measurable action verbs.

    Start with:
    "By the end of the session, students will be able to:"

    Then provide 3-5 bullet points.

    Each bullet point must begin with an action verb such as:
    identify, explain, demonstrate, describe, classify, compare,
    operate, distinguish, or list.

    The objectives must be observable and measurable.

    Tools Required:

    First write a short paragraph explaining why these tools are needed
    for the activity or topic.

    After the paragraph, provide a clear list of tools required.
    Write the tools as bullet points.
    Each tool should be practical and relevant to the lesson.

    Raw Material Required:

    First write a short paragraph explaining what raw materials
    are needed for the activity or demonstration.

    After the paragraph, provide a clear list of materials required.
    Write the materials as bullet points.

    Ensure the materials are realistic and suitable for a school workshop
    or classroom demonstration.

    Safety of Students:

    Write important safety precautions students must follow
    during the activity or practical session.

    List the safety precautions clearly using bullet points.

    Focus on practical workshop safety such as protective
    equipment, safe handling of tools, and awareness of
    hazards related to the activity.

    Grouping:
    Task Distribution:
    Sitting Arrangement:

    Revision of Previous Class:

    First write a short paragraph briefly summarizing the topic
    from the previous class session: {previous_session}.

    After the paragraph, write 3-4 simple revision questions
    based on that previous topic.

    These questions should help students recall what they
    learned in the last lesson.

    Opening or Hook:

    Provide 3-4 possible ways the teacher can begin the lesson
    in an engaging way.

    Each option should be different and clearly labelled.

    Examples of hooks may include:
    • Asking an interesting question
    • Showing a real-life example
    • Demonstrating an object or tool
    • Telling a short real-world story related to the topic

    Each hook should include a short explanation of how the teacher
    can present it in the classroom.

    Learning by Doing:

    First write a short paragraph explaining the practical
    activity that students will perform in this lesson.

    Explain the purpose of the activity and what students
    will learn from doing it.

    After the paragraph, write a clear step-by-step list
    of instructions for the activity.

    Each step should be simple, practical, and suitable
    for students in a school workshop or classroom.

    Closing:

    First write a short paragraph summarizing the main ideas
    covered in the lesson.

    After the paragraph, provide a brief recap of the key
    points discussed in the lesson.

    Write the recap as bullet points so students can quickly
    review the important concepts they learned.

    Assessment:

    Create 4-5 short assessment questions based on the lesson.

    After each question, provide the correct answer.

    The questions should help check whether students understood
    the key concepts from the lesson.

    Format the section clearly like this:

    Question 1:
    Answer:

    Question 2:
    Answer:

    Homework:

    Provide 6-7 possible homework assignments related to the lesson topic.

    The homework tasks should be varied so the teacher can choose
    the most suitable one.

    Include different types of homework such as:
    • Short written questions
    • Drawing or diagram activity
    • Observation task
    • Small research task
    • Practical thinking question

    Write each homework task as a separate bullet point.

    Make it interesting and engaging.

    Do NOT include Reflection by Trainer.
    """

            lesson = client.chat.completions.create(
                model="gpt-4.1-mini", messages=[{"role": "user", "content": lesson_prompt}]
            )

            lesson_response = lesson.choices[0].message.content

            # Remove markdown code block if present
            lesson_response = (
                lesson_response.replace("```json", "").replace("```", "").strip()
            )

            start = lesson_response.find("{")
            end = lesson_response.rfind("}") + 1
            lesson_response = lesson_response[start:end]

            try:
                lesson_data = json.loads(lesson_response)

                # ✅ ADD THIS BLOCK (STORE FULL DATA)
                lesson_data["date"] = lesson_date.strftime("%d/%m/%Y")
                lesson_data["class_level"] = class_level
                lesson_data["subject"] = subject
                lesson_data["unit"] = selected_unit
                lesson_data["session"] = session
                lesson_data["objective"] = objective
                lesson_data["duration"] = duration
                lesson_data["boys"] = boys
                lesson_data["girls"] = girls

                # Save to session
                st.session_state["lesson_data"] = lesson_data

                # Save to database
                save_lesson(
                    lesson_date,
                    class_level,
                    subject,
                    selected_unit,
                    session,
                    objective,
                    lesson_data
                )

            except json.JSONDecodeError:
                st.error("AI response was not valid JSON.")
                st.code(lesson_response)
                st.stop()

            if "lesson_data" in st.session_state:
                lesson_data = st.session_state["lesson_data"]

                # WORKSHEET PROMPT
                worksheet_prompt = f"""
                You are preparing a STRICT student worksheet.

                Use ONLY the textbook content below.

                TEXTBOOK CONTENT
                ----------------
                {textbook_text}

                Lesson topic: {session}
                Subject: {subject}
                Class: {class_level}

                IMPORTANT RULES (DO NOT IGNORE):

                1. TOTAL LENGTH MUST FIT IN ONE PAGE.
                2. KEEP ANSWERS SHORT.
                3. DO NOT ADD EXTRA SECTIONS.
                4. DO NOT EXPLAIN TOO MUCH.

                STRUCTURE (FOLLOW EXACTLY):

                SECTION 1: SHORT QUESTIONS (4 QUESTIONS ONLY)
                - Ask 4 important short answer questions.
                - Keep them clear and exam-focused.

                SECTION 2: MCQs (6 QUESTIONS ONLY)
                - Each MCQ must have 4 options (A, B, C, D)
                - Do NOT give answers.

                SECTION 3: ACTIVITY (2 ONLY)
                - Give 2 simple practical or thinking activities.
                - Keep instructions short.

                SECTION 4: REFLECTION (2 QUESTIONS ONLY)
                - Ask 2 simple reflective questions.

                FORMAT CLEANLY. NO EXTRA TEXT.
                """

                worksheet = client.chat.completions.create(
                    model="gpt-4.1-mini",
                    messages=[{"role": "user", "content": worksheet_prompt}],
                )

                st.session_state["worksheet"] = worksheet

                # QUIZ PROMPT
                quiz_prompt = f"""
                You are preparing a quiz for students.

                Use the textbook content as the main reference.

                        TEXTBOOK CONTENT
                        ----------------
                        {textbook_text}

                        Subject: {subject}
                        Session: {session}
                        Class: {class_level}

                        Create:
                        5 multiple choice questions
                        5 short answer questions

                        Focus primarily on the concepts mentioned in the textbook.
                        Avoid introducing topics not present in the textbook.
                        """

                quiz = client.chat.completions.create(
                    model="gpt-4.1-mini",
                    messages=[{"role": "user", "content": quiz_prompt}],
                )

                st.session_state["quiz"] = quiz

        # ---- DISPLAY RESULTS ----

        tab1, tab2, tab3 = st.tabs(["📚 Lesson Plan", "📝 Worksheet", "❓ Quiz"])

        with tab1:
            st.subheader("Lesson Plan (Editable Sections)")

            if "lesson_data" not in st.session_state:
                st.info("Generate a lesson first.")
                st.stop()

            lesson_data = st.session_state["lesson_data"]

            # Reset Button
            if st.button("🔄 Reset Lesson"):
                st.session_state.clear()
                st.rerun()

            st.write("You can edit each section before downloading the final lesson plan.")

            knowledge = st.text_area(
                "Knowledge Key Points (Short Summary)", lesson_data["knowledge"], height=60
            )

            what = st.text_area(
                "What – What must students understand?", lesson_data["what"], height=80
            )

            why = st.text_area(
                "Why – Why is this topic important?", lesson_data["why"], height=80
            )

            objective = st.text_area("Objective", lesson_data["objective"], height=80)

            tools = st.text_area("Tools Required", lesson_data["tools"], height=80)

            raw_material = st.text_area(
                "Raw Material Required", lesson_data["raw_material"], height=80
            )

            safety = st.text_area("Safety of Students", lesson_data["safety"], height=80)

            grouping = st.text_area("Grouping", lesson_data["grouping"], height=80)

            task_distribution = st.text_area(
                "Task Distribution", lesson_data["task_distribution"], height=80
            )

            sitting = st.text_area("Sitting Arrangement", lesson_data["sitting"], height=80)

            revision = st.text_area(
                "Revision of Previous Class", lesson_data["revision"], height=80
            )

            opening = st.text_area("Opening / Hook", lesson_data["opening"], height=80)

            activity = st.text_area(
                "Learning by Doing (Step-by-step activity)",
                lesson_data["activity"],
                height=120,
            )

            closing = st.text_area("Closing", lesson_data["closing"], height=80)

            assessment = st.text_area("Assessment", lesson_data["assessment"], height=80)

            homework = st.text_area("Homework", lesson_data["homework"], height=80)

            reflection = st.text_area(
                "Reflection by Trainer (To be filled after class)",
                "",
                height=80,
                placeholder="Write reflection after conducting the class...",
            )

            homework = "• " + homework.replace(", ", "\n• ")

            # Prepare lesson data for PDF
            lesson_output = {
                "Date": lesson_date.strftime("%d/%m/%Y"),
                "Grade": class_level,
                "Topic": session,
                "Duration": duration,
                "Boys": boys,
                "Girls": girls,
                "Objective": objective,
                "Knowledge_Key_Points": knowledge,
                "What": what,
                "Why": why,
                "Tools_Required": tools,
                "Raw_Material_Required": raw_material,
                "Safety_of_Students": safety,
                "Grouping": grouping,
                "Task_Distribution": task_distribution,
                "Sitting_Arrangement": sitting,
                "Revision": revision,
                "Opening": opening,
                "Learning_by_Doing": activity,
                "Closing": closing,
                "Assessment": assessment,
                "Homework": homework,
                "Reflection_by_Trainer": reflection,
            }

            # Generate PDF
            pdf_file = generate_pdf(lesson_output)

            # Download button
            st.download_button(
                label="📄 Download Lesson Plan PDF",
                data=pdf_file,
                file_name="AI_SkillLab_Lesson_Plan.pdf",
                mime="application/pdf",
            )
        with tab2:
            st.subheader("Student Worksheet")

            if "worksheet" in st.session_state:

                if "edited_worksheet" not in st.session_state:
                    st.session_state["edited_worksheet"] = st.session_state["worksheet"].choices[0].message.content

                st.session_state["edited_worksheet"] = st.text_area(
                    "Edit Worksheet",
                    value=st.session_state["edited_worksheet"],
                    height=400
                )

                worksheet_pdf = generate_worksheet_pdf(st.session_state["edited_worksheet"])

                st.download_button(
                    label="📄 Download Worksheet PDF",
                    data=worksheet_pdf,
                    file_name="AI_SkillLab_Worksheet.pdf",
                    mime="application/pdf",
                )

        with tab3:
            st.subheader("Quiz Questions")

            if "quiz" in st.session_state:
                quiz = st.session_state["quiz"]
                st.write(quiz.choices[0].message.content)


# ---- TEACHER DASHBOARD ----

if page == "Teacher Dashboard":

    st.title("📚 Teacher Dashboard")

    conn = sqlite3.connect("lesson_history.db")
    c = conn.cursor()

    c.execute("""
    SELECT id, date, class_level, subject, unit, session
    FROM lessons
    ORDER BY id DESC
    """)

    lessons = c.fetchall()

    conn.close()

    if lessons:

        for lesson in lessons:

            lesson_id, date, class_level, subject, unit, session = lesson

            with st.expander(f"{date} | Class {class_level} | {session}"):

                st.write("Subject:", subject)
                st.write("Unit:", unit)

                if st.button(f"View Lesson {lesson_id}"):

                    conn = sqlite3.connect("lesson_history.db")
                    c = conn.cursor()

                    c.execute(
                        "SELECT lesson_json FROM lessons WHERE id=?",
                        (lesson_id,)
                    )

                    data = c.fetchone()

                    conn.close()

                    if data:

                        lesson_data = json.loads(data[0])

                        st.subheader("Lesson Plan")

                        st.write("### Knowledge")
                        st.write(lesson_data.get("knowledge", ""))

                        st.write("### What")
                        st.write(lesson_data.get("what", ""))

                        st.write("### Why")
                        st.write(lesson_data.get("why", ""))

                        st.write("### Activity")
                        st.write(lesson_data.get("activity", ""))

                        st.write("### Assessment")
                        st.write(lesson_data.get("assessment", ""))

                        st.write("### Homework")

                        homework_data = lesson_data.get("homework", "")

                        if isinstance(homework_data, list):
                            for item in homework_data:
                                st.write(f"• {item}")
                        else:
                            # fallback for string
                            homework_lines = str(homework_data).split(", ")
                            for line in homework_lines:
                                st.write(f"• {line}")

                        # -------- GENERATE PDF FROM STORED DATA --------

                        lesson_output = {
                            "Date": lesson_data.get("date", ""),
                            "Grade": lesson_data.get("class_level", ""),
                            "Topic": lesson_data.get("session", ""),
                            "Duration": lesson_data.get("duration", ""),
                            "Boys": lesson_data.get("boys", ""),
                            "Girls": lesson_data.get("girls", ""),
                            "Objective": lesson_data.get("objective", ""),
                            "Knowledge_Key_Points": lesson_data.get("knowledge", ""),
                            "What": lesson_data.get("what", ""),
                            "Why": lesson_data.get("why", ""),
                            "Tools_Required": lesson_data.get("tools", ""),
                            "Raw_Material_Required": lesson_data.get("raw_material", ""),
                            "Safety_of_Students": lesson_data.get("safety", ""),
                            "Grouping": lesson_data.get("grouping", ""),
                            "Task_Distribution": lesson_data.get("task_distribution", ""),
                            "Sitting_Arrangement": lesson_data.get("sitting", ""),
                            "Revision": lesson_data.get("revision", ""),
                            "Opening": lesson_data.get("opening", ""),
                            "Learning_by_Doing": lesson_data.get("activity", ""),
                            "Closing": lesson_data.get("closing", ""),
                            "Assessment": lesson_data.get("assessment", ""),
                            "Homework": lesson_data.get("homework", ""),
                            "Reflection_by_Trainer": ""
                        }

                        pdf_file = generate_pdf(lesson_output)

                        st.download_button(
                            label="📄 Download Lesson Plan PDF",
                            data=pdf_file,
                            file_name=f"Lesson_{lesson_id}.pdf",
                            mime="application/pdf"
                        )
    else:
        st.info("No lessons saved yet.")
