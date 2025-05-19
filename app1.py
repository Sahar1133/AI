# ====================== IMPORTS ======================
import matplotlib.pyplot as plt # For data visualization
import pandas as pd # For data manipulation and analysis
import numpy as np # For numerical operations
import streamlit as st # For building the web app interface
from sklearn.tree import DecisionTreeClassifier, export_text # Machine learning model
from sklearn.model_selection import train_test_split # For splitting data into train/test sets
from sklearn.preprocessing import LabelEncoder # For encoding categorical variables
from sklearn.metrics import accuracy_score # For evaluating model performance
import random # For randomizing questions

# ====================== STYLING & SETUP ======================
# Configure the Streamlit page settings
st.set_page_config(
    page_title="AI Powered Career Prediction Based on Personality Traits", # Browser tab title
    page_icon="🧭", # Browser tab icon
    layout="wide", # Use wider page layout
    initial_sidebar_state="expanded" # Start with sidebar expanded
)

def apply_custom_css():
    """Applies custom CSS styling to the Streamlit app"""
    st.markdown("""
    <style>
    /* Main background with image and overlay */
    .stApp {
        background-image: url('https://images.unsplash.com/photo-1451187580459-43490279c0fa?ixlib=rb-1.2.1&auto=format&fit=crop&w=1600&q=80');
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }
    
    /* Overlay for content readability */
    .main .block-container {
        background-color: rgba(255, 255, 255, 0.93);
        border-radius: 12px;
        padding: 2rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.15);
        margin-top: 2rem;
        margin-bottom: 2rem;
    }
    
    /* Card styling */
    .stCard {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        border: 1px solid #e1e4e8;
    }
    
    /* Button styling with animation */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 12px 28px;
        font-size: 16px;
        font-weight: 500;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
        background: linear-gradient(135deg, #5a67d8 0%, #6b46c1 100%);
    }
    
    /* Input fields with modern look */
    .stTextInput>div>div>input,
    .stNumberInput>div>div>input {
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 12px;
        background: rgba(255, 255, 255, 0.9);
        transition: all 0.3s;
    }
    .stTextInput>div>div>input:focus,
    .stNumberInput>div>div>input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.2);
    }
    
    /* Radio buttons with card-like appearance */
    .stRadio > div {
        flex-direction: column;
        gap: 12px;
    }
    .stRadio > div > label {
        background: rgba(255, 255, 255, 0.9);
        padding: 16px;
        border-radius: 10px;
        transition: all 0.2s;
        border: 1px solid #e2e8f0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.03);
    }
    .stRadio > div > label:hover {
        border-color: #667eea;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.08);
    }
    .stRadio > div > label[data-baseweb="radio"]:first-child {
        margin-top: 0;
    }
    
    /* Headers with modern typography */
    h1 {
        color: #2d3748;
        font-weight: 700;
        margin-bottom: 1rem;
        position: relative;
    }
    h1:after {
        content: "";
        position: absolute;
        bottom: -8px;
        left: 0;
        width: 60px;
        height: 4px;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        border-radius: 2px;
    }
    h2 {
        color: #4a5568;
        font-weight: 600;
        margin-top: 1.5rem;
    }
    h3 {
        color: #4a5568;
        font-weight: 500;
    }
    
    /* Expanders with card styling */
    .stExpander {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        border: 1px solid #e2e8f0;
    }
    .stExpander > summary {
        font-weight: 600;
        padding: 1rem 1.5rem;
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 12px 24px;
        border-radius: 8px 8px 0 0;
        transition: all 0.3s;
        background-color: rgba(255, 255, 255, 0.7);
    }
    .stTabs [aria-selected="true"] {
        background-color: white;
        color: #667eea;
        font-weight: 600;
    }
    .stTabs [aria-selected="false"] {
        background-color: rgba(255, 255, 255, 0.5);
        color: #4a5568;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(135deg, #2d3748 0%, #1a202c 100%);
        color: white;
    }
    [data-testid="stSidebar"] .stRadio > div > label {
        background: rgba(255,255,255,0.05);
        color: white;
        border-color: rgba(255,255,255,0.1);
    }
    [data-testid="stSidebar"] .stRadio > div > label:hover {
        background: rgba(255,255,255,0.1);
    }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] p {
        color: white !important;
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    ::-webkit-scrollbar-track {
        background: #f1f5f9;
    }
    ::-webkit-scrollbar-thumb {
        background: #cbd5e0;
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #a0aec0;
    }
    
    /* Success message styling */
    .stAlert .st-b7 {
        background-color: rgba(236, 253, 245, 0.95) !important;
        border-radius: 12px !important;
    }
    
    /* Dataframe styling */
    .stDataFrame {
        background-color: rgba(255, 255, 255, 0.95) !important;
        border-radius: 12px !important;
    }
    </style>
    """, unsafe_allow_html=True)

# ====================== DATA LOADING & PREPROCESSING ======================
@st.cache_data # Cache the data to avoid reloading on every interaction
def load_data():
    career_options = [
        'Zoologist', 'Pharmacist', 'Forensic Scientist', 'Pediatric Nurse',
        'Market Research Analyst', 'Game Developer', 'Geologist', 'Speech Therapist',
        'Data Analyst', 'Civil Engineer', 'Teacher', 'Genetic Counselor',
        'Environmental Engineer', 'Financial Planner', 'Air Traffic Controller',
        'Fashion Stylist', 'Human Resources Manager', 'Architect', 'Video Game Tester',
        'Administrative Officer', 'Industrial Engineer', 'Social Worker', 'Chef',
        'Film Director', 'Human Rights Lawyer', 'Astronomer', 'Fashion Designer',
        'Biologist', 'Public Health Analyst', 'Forestry Technician', 'Salesperson',
        'Investment Banker', 'Marketing Coordinator', 'Wildlife Biologist',
        'Software Quality Assurance Tester', 'Interior Designer', 'Public Relations Specialist',
        'Nurse', 'Aerospace Engineer', 'Marketing Manager', 'Database Administrator',
        'Web Developer', 'Mechanical Designer', 'IT Support Specialist', 'Dental Hygienist',
        'Psychologist', 'Occupational Therapist', 'Tax Accountant', 'Software Developer',
        'Musician', 'Journalist', 'Real Estate Agent', 'Speech Pathologist',
        'Biotechnologist', 'Environmental Scientist', 'Police Officer', 'Foreign Service Officer',
        'Accountant', 'Rehabilitation Counselor', 'Robotics Engineer', 'Artist',
        'Marketing Analyst', 'Event Photographer', 'Research Scientist', 'HR Recruiter',
        'Forensic Psychologist', 'Insurance Underwriter', 'Marine Biologist', 'Technical Writer'
    ]
    
    try:
        # Try to load real dataset
        data = pd.read_excel("new.xlsx")
        # If career field data is sparse, generate random career assignments
        if len(data['Predicted_Career_Field'].unique()) < 20:
            data['Predicted_Career_Field'] = np.random.choice(career_options, size=len(data))
    except FileNotFoundError:
        # Fallback to demo data if real dataset not found
        st.warning("⚠️ Dataset not found. Using demo data.")
        data = pd.DataFrame({
            'Interest': np.random.choice(['Technology', 'Business', 'Arts', 'Engineering', 'Medical', 'Science', 'Education', 'Law'], 200),
            'Work_Style': np.random.choice(['Independent', 'Collaborative', 'Flexible'], 200),
            'Strengths': np.random.choice(['Analytical', 'Creative', 'Strategic', 'Practical'], 200),
            'Communication_Skills': np.random.choice(['Low', 'Medium', 'High'], 200),
            'Leadership_Skills': np.random.choice(['Low', 'Medium', 'High'], 200),
            'Teamwork_Skills': np.random.choice(['Low', 'Medium', 'High'], 200),
            'Decision_Making': np.random.choice(['Low', 'Medium', 'High'], 200),
            'Aptitude_Level': np.random.choice(['Low', 'Medium', 'High'], 200),
            'Field_of_Study': np.random.choice(['Biology', 'Chemistry', 'Physics', 'Mathematics', 'Computer Science',
                                              'Engineering', 'Environmental Science', 'Psychology', 'Economics', 'Business',
                                              'Law', 'Art and Design', 'Political Science', 'Medicine', 'Pharmacy',
                                              'Nursing', 'Education', 'Communication', 'Journalism', 'Sociology'], 200),
            'Adaptability': np.random.choice(['Low', 'Medium', 'High'], 200),
            'Time_Management': np.random.choice(['Low', 'Medium', 'High'], 200),
            'Problem_Solving': np.random.choice(['Low', 'Medium', 'High'], 200),
            'Emotional_Intelligence': np.random.choice(['Low', 'Medium', 'High'], 200),
            'Stress_Tolerance': np.random.choice(['Low', 'Medium', 'High'], 200),
            'Learning_Style': np.random.choice(['Reading/Writing', 'Auditory', 'Visual', 'Kinesthetic'], 200),
            'Technical_Skill_Level': np.random.choice(['Low', 'Medium', 'High'], 200),
            'Risk_Tolerance': np.random.choice(['Low', 'Medium', 'High'], 200),
            'Introvert_Extrovert_Score': np.random.choice(['Low', 'Medium', 'High'], 200),
            'Workplace_Preference': np.random.choice(['Remote', 'On-site', 'Hybrid'], 200),
            'Work_Life_Balance_Preference': np.random.choice(['Low', 'Medium', 'High'], 200),
            'Salary_Expectation': np.random.choice(['Low', 'Medium', 'High'], 200),
            'Openness': np.random.choice(['Low', 'Medium', 'High'], 200),
            'Conscientiousness': np.random.choice(['Low', 'Medium', 'High'], 200),
            'Extraversion': np.random.choice(['Low', 'Medium', 'High'], 200),
            'Agreeableness': np.random.choice(['Low', 'Medium', 'High'], 200),
            'Neuroticism': np.random.choice(['Low', 'Medium', 'High'], 200),
            'Creativity_Score': np.random.choice(['Low', 'Medium', 'High'], 200),
            'Location_Preference': np.random.choice(['Urban', 'Rural', 'Flexible'], 200),
            'Willing_to_Relocate': np.random.choice(['Yes', 'No'], 200),
            'Industry_of_Experience': np.random.choice(['Arts', 'Construction', 'Education', 'Finance', 'Healthcare', 'Law', 'Retail', 'Tech'], 200),
            'Internship_Experience': np.random.choice(['Yes', 'No'], 200),
            'Remote_Work_Experience': np.random.choice(['Yes', 'No'], 200),
            'LinkedIn_Portfolio': np.random.choice(['Yes', 'No'], 200),
            'Public_Speaking_Experience': np.random.choice(['Yes', 'No'], 200),
            'GPA': np.round(np.random.uniform(2.0, 4.0, 200), 1),
            'Years_of_Experience': np.random.randint(0, 20, 200),
            'Certifications_Count': np.random.randint(0, 5, 200),
            'Predicted_Career_Field': np.random.choice(career_options, 200)
        })
    # Clean GPA data if it exists
    if 'GPA' in data.columns:
        data['GPA'] = pd.to_numeric(data['GPA'], errors='coerce')
        data['GPA'].fillna(data['GPA'].median(), inplace=True)
    
    return data

data = load_data()

# ====================== MODEL TRAINING ======================
def preprocess_data(data):
    le = LabelEncoder()
    # Identify categorical columns (excluding the target)
    object_cols = [col for col in data.select_dtypes(include=['object']).columns 
                  if col in data.columns]
    # Encode each categorical column
    for col in object_cols:
        if col != 'Predicted_Career_Field':
            data[col] = le.fit_transform(data[col].astype(str))
    # Encode the target variable (career field)
    if 'Predicted_Career_Field' in data.columns:
        data['Predicted_Career_Field'] = le.fit_transform(data['Predicted_Career_Field'])
    return data, le # Return processed data and the target encoder

# Process the data
processed_data, target_le = preprocess_data(data.copy())

# ====================== MODEL TRAINING ======================
def train_model(data):
    if 'Predicted_Career_Field' not in data.columns:
        st.error("Target column not found in data")
        return None, 0
    # Prepare features (X) and target (y)
    X = data.drop('Predicted_Career_Field', axis=1)
    y = data['Predicted_Career_Field']
    
    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize and train the decision tree model
    model = DecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate model accuracy
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return model, accuracy
    
# Train the model
model, accuracy = train_model(processed_data)

# ====================== QUESTIONNAIRE ======================
def get_all_questions():
    """Returns a pool of questions organized by category"""
    questions_dict = {
        "Interest": [
            {
                "question": "What is your area of interest?",
                "options": [
                    {"text": "Zoologist", "value": "Zoologist"},
                    {"text": "Pharmacist", "value": "Pharmacist"},
                    {"text": "Forensic Scientist", "value": "Forensic Scientist"},
                    {"text": "Pediatric Nurse", "value": "Pediatric Nurse"},
                    {"text": "Market Research Analyst", "value": "Market Research Analyst"},
                    {"text": "Game Developer", "value": "Game Developer"},
                    {"text": "Geologist", "value": "Geologist"},
                    {"text": "Speech Therapist", "value": "Speech Therapist"},
                    {"text": "Data Analyst", "value": "Data Analyst"},
                    {"text": "Civil Engineer", "value": "Civil Engineer"},
                    {"text": "Teacher", "value": "Teacher"},
                    {"text": "Genetic Counselor", "value": "Genetic Counselor"},
                    {"text": "Environmental Engineer", "value": "Environmental Engineer"},
                    {"text": "Financial Planner", "value": "Financial Planner"},
                    {"text": "Air Traffic Controller", "value": "Air Traffic Controller"},
                    {"text": "Fashion Stylist", "value": "Fashion Stylist"},
                    {"text": "Human Resources Manager", "value": "Human Resources Manager"},
                    {"text": "Architect", "value": "Architect"},
                    {"text": "Video Game Tester", "value": "Video Game Tester"},
                    {"text": "Administrative Officer", "value": "Administrative Officer"}
                ],
                "feature": "Interest"
            }
        ],
        "Work_Style": [
            {
                "question": "How do you prefer to structure your day?",
                "options": [
                    {"text": "I create and follow my own plan", "value": "Independent"},
                    {"text": "I adjust based on how the day unfolds", "value": "Flexible"},
                    {"text": "I coordinate closely with others", "value": "Collaborative"}
                ],
                "feature": "Work_Style"
            },
            {
                "question": "When working on a group project, what role do you naturally take?",
                "options": [
                    {"text": "I prefer to work on my own tasks solo", "value": "Independent"},
                    {"text": "I switch roles based on what's needed", "value": "Flexible"},
                    {"text": "I bring people together and coordinate efforts", "value": "Collaborative"}
                ],
                "feature": "Work_Style"
            }
        ],
        "Strengths": [
            {
                "question": "When you're faced with a complex issue, what's your first instinct?",
                "options": [
                    {"text": "Create something new and imaginative", "value": "Creative"},
                    {"text": "Strategize and plan the steps carefully", "value": "Strategic"},
                    {"text": "Break it down into logical, solvable parts", "value": "Analytical"}
                ],
                "feature": "Strengths"
            },
            {
                "question": "Which compliment resonates the most with you?",
                "options": [
                    {"text": "You're incredibly imaginative!", "value": "Creative"},
                    {"text": "You always see the big picture.", "value": "Strategic"},
                    {"text": "You have excellent problem-solving skills.", "value": "Analytical"}
                ],
                "feature": "Strengths"
            }
        ],
        "Communication_Skills": [
            {
                "question": "During group discussions, how do you usually participate?",
                "options": [
                    {"text": "I lead the conversation and clarify ideas", "value": "High"},
                    {"text": "I share my thoughts and support others", "value": "Medium"},
                    {"text": "I prefer to stay quiet and observe", "value": "Low"}
                ],
                "feature": "Communication_Skills"
            }
        ],
        "Leadership_Skills": [
            {
                "question": "When a team faces a challenge, what is your typical role?",
                "options": [
                    {"text": "I take charge and guide the team toward a solution", "value": "High"},
                    {"text": "I support the leader and help implement ideas", "value": "Medium"},
                    {"text": "I wait for others to take charge and follow along", "value": "Low"}
                ],
                "feature": "Leadership_Skills"
            }
        ],
        "Teamwork_Skills": [
            {
                "question": "How do you contribute to a team's success?",
                "options": [
                    {"text": "I actively collaborate, offering ideas and support to everyone", "value": "High"},
                    {"text": "I participate when needed and contribute in specific areas", "value": "Medium"},
                    {"text": "I mostly work alone and only contribute when asked", "value": "Low"}
                ],
                "feature": "Teamwork_Skills"
            }
        ],
        "Decision_Making": [
            {
                "question": "When faced with a difficult decision, how do you approach it?",
                "options": [
                    {"text": "I gather information, weigh the pros and cons, and make an informed decision", "value": "High"},
                    {"text": "I seek input from others and try to make a balanced decision", "value": "Medium"},
                    {"text": "I avoid making decisions and leave them to others", "value": "Low"}
                ],
                "feature": "Decision_Making"
            }
        ],
        "Aptitude_Level": [
            {
                "question": "If 5 machines take 5 minutes to make 5 products, how long will 100 machines take to make 100 products?",
                "options": [
                    {"text": "5 minutes", "value": "High"},
                    {"text": "50 minutes", "value": "Medium"},
                    {"text": "100 minutes", "value": "Low"}
                ],
                "feature": "Aptitude_Level"
            }
        ],
        "Field_of_Study": [
            {
                "question": "Which academic field do you feel most passionate about?",
                "options": [
                    {"text": "Biology", "value": "Biology"},
                    {"text": "Chemistry", "value": "Chemistry"},
                    {"text": "Physics", "value": "Physics"},
                    {"text": "Mathematics", "value": "Mathematics"},
                    {"text": "Computer Science", "value": "Computer Science"},
                    {"text": "Engineering", "value": "Engineering"},
                    {"text": "Environmental Science", "value": "Environmental Science"},
                    {"text": "Psychology", "value": "Psychology"},
                    {"text": "Economics", "value": "Economics"},
                    {"text": "Business", "value": "Business"}
                ],
                "feature": "Field_of_Study"
            }
        ],
        "Adaptability": [
            {
                "question": "How do you react when plans change at the last minute?",
                "options": [
                    {"text": "I quickly adjust and move forward", "value": "High"},
                    {"text": "I feel a bit uneasy but manage", "value": "Medium"},
                    {"text": "I get frustrated and find it hard to cope", "value": "Low"}
                ],
                "feature": "Adaptability"
            }
        ],
        "Time_Management": [
            {
                "question": "How often do you create a to-do list or plan your day in advance?",
                "options": [
                    {"text": "Daily - I rely on it to stay organized", "value": "High"},
                    {"text": "Occasionally, when I have a lot to do", "value": "Medium"},
                    {"text": "Rarely - I go with the flow", "value": "Low"}
                ],
                "feature": "Time_Management"
            }
        ],
        "Problem_Solving": [
            {
                "question": "When faced with a complex problem, what is your first reaction?",
                "options": [
                    {"text": "Break it down and analyze step by step", "value": "High"},
                    {"text": "Try a few things and see what works", "value": "Medium"},
                    {"text": "Feel overwhelmed and unsure how to begin", "value": "Low"}
                ],
                "feature": "Problem_Solving"
            }
        ],
        "Emotional_Intelligence": [
            {
                "question": "How do you react when you feel angry or frustrated?",
                "options": [
                    {"text": "I recognize my emotions and take a step back to calm down", "value": "High"},
                    {"text": "I try to ignore it or bottle it up", "value": "Medium"},
                    {"text": "I express it right away, often without thinking", "value": "Low"}
                ],
                "feature": "Emotional_Intelligence"
            }
        ],
        "Stress_Tolerance": [
            {
                "question": "How do you typically feel when you are faced with multiple deadlines or tasks at once?",
                "options": [
                    {"text": "I stay calm, prioritize, and work through the tasks methodically", "value": "High"},
                    {"text": "I feel stressed but manage to get through it with some effort", "value": "Medium"},
                    {"text": "I feel overwhelmed and struggle to complete tasks", "value": "Low"}
                ],
                "feature": "Stress_Tolerance"
            }
        ],
        "Learning_Style": [
            {
                "question": "When you are studying or learning new information, which method works best for you?",
                "options": [
                    {"text": "Writing notes, summarizing what I've learned, or reading books/articles", "value": "Reading/Writing"},
                    {"text": "Listening to podcasts, lectures, or discussions", "value": "Auditory"},
                    {"text": "Watching videos, diagrams, or other visual content", "value": "Visual"},
                    {"text": "Actively doing hands-on activities, practicing tasks, or using physical examples", "value": "Kinesthetic"}
                ],
                "feature": "Learning_Style"
            }
        ],
        "Technical_Skill_Level": [
            {
                "question": "When using a new software or tool, how do you usually proceed?",
                "options": [
                    {"text": "I explore it confidently and figure it out on my own", "value": "High"},
                    {"text": "I need some guidance or tutorials to get started", "value": "Medium"},
                    {"text": "I feel unsure and prefer someone else sets it up", "value": "Low"}
                ],
                "feature": "Technical_Skill_Level"
            }
        ],
        "Risk_Tolerance": [
            {
                "question": "When presented with a new opportunity that has both high rewards and high risk, what is your initial reaction?",
                "options": [
                    {"text": "I feel excited and consider how to take it strategically", "value": "High"},
                    {"text": "I weigh the pros and cons carefully before deciding", "value": "Medium"},
                    {"text": "I avoid it unless the risk is minimal", "value": "Low"}
                ],
                "feature": "Risk_Tolerance"
            }
        ],
        "Introvert_Extrovert_Score": [
            {
                "question": "How do you usually feel after spending a few hours at a lively social gathering?",
                "options": [
                    {"text": "Energized and excited to keep socializing", "value": "High"},
                    {"text": "It was fun, but I need a bit of alone time now", "value": "Medium"},
                    {"text": "Drained and ready for quiet solitude", "value": "Low"}
                ],
                "feature": "Introvert_Extrovert_Score"
            }
        ],
        "Workplace_Preference": [
            {
                "question": "What type of setting makes you feel most productive during the day?",
                "options": [
                    {"text": "A quiet home office", "value": "Remote"},
                    {"text": "A collaborative open office", "value": "On-site"},
                    {"text": "A mix depending on the task", "value": "Hybrid"}
                ],
                "feature": "Workplace_Preference"
            }
        ],
        "Work_Life_Balance_Preference": [
            {
                "question": "What do you typically do after work?",
                "options": [
                    {"text": "Unplug and relax", "value": "High"},
                    {"text": "Catch up on more tasks", "value": "Low"},
                    {"text": "Check emails but relax later", "value": "Moderate"}
                ],
                "feature": "Work_Life_Balance_Preference"
            }
        ],
        "Salary_Expectation": [
            {
                "question": "What makes a job offer attractive?",
                "options": [
                    {"text": "Salary", "value": "High"},
                    {"text": "Culture and learning", "value": "Low"},
                    {"text": "A mix", "value": "Moderate"}
                ],
                "feature": "Salary_Expectation"
            }
        ],
        "Openness": [
            {
                "question": "What's your reaction to new, abstract ideas?",
                "options": [
                    {"text": "Excited and curious", "value": "High"},
                    {"text": "Interested but cautious", "value": "Medium"},
                    {"text": "Skeptical or uninterested", "value": "Low"}
                ],
                "feature": "Openness"
            }
        ],
        "Conscientiousness": [
            {
                "question": "How do you usually manage your tasks?",
                "options": [
                    {"text": "I plan everything ahead", "value": "High"},
                    {"text": "I manage as things come", "value": "Medium"},
                    {"text": "I often forget or delay", "value": "Low"}
                ],
                "feature": "Conscientiousness"
            }
        ],
        "Extraversion": [
            {
                "question": "What gives you energy?",
                "options": [
                    {"text": "Social interaction", "value": "High"},
                    {"text": "A mix of social and alone time", "value": "Medium"},
                    {"text": "Solitude", "value": "Low"}
                ],
                "feature": "Extraversion"
            }
        ],
        "Agreeableness": [
            {
                "question": "When a team member makes a mistake, how do you respond?",
                "options": [
                    {"text": "Help them correct it gently", "value": "High"},
                    {"text": "Point it out constructively", "value": "Medium"},
                    {"text": "Get frustrated or blame", "value": "Low"}
                ],
                "feature": "Agreeableness"
            }
        ],
        "Neuroticism": [
            {
                "question": "How do you handle criticism?",
                "options": [
                    {"text": "Stay calm and reflect", "value": "Low"},
                    {"text": "Feel a bit affected but move on", "value": "Medium"},
                    {"text": "Take it personally and dwell on it", "value": "High"}
                ],
                "feature": "Neuroticism"
            }
        ],
        "Creativity_Score": [
            {
                "question": "When faced with a problem, how do you usually approach finding a solution?",
                "options": [
                    {"text": "I look for new, unconventional ways to solve it", "value": "High"},
                    {"text": "I consider some different options but mostly rely on proven methods", "value": "Medium"},
                    {"text": "I prefer sticking to the usual, well-established solutions", "value": "Low"}
                ],
                "feature": "Creativity_Score"
            }
        ],
        "Location_Preference": [
            {
                "question": "Where would you prefer to work?",
                "options": [
                    {"text": "Urban area with many opportunities", "value": "Urban"},
                    {"text": "Rural area with peaceful environment", "value": "Rural"},
                    {"text": "I'm flexible about location", "value": "Flexible"}
                ],
                "feature": "Location_Preference"
            }
        ],
        "Willing_to_Relocate": [
            {
                "question": "Are you willing to relocate for a job opportunity?",
                "options": [
                    {"text": "Yes, I'm open to relocating", "value": "Yes"},
                    {"text": "No, I prefer to stay in my current location", "value": "No"}
                ],
                "feature": "Willing_to_Relocate"
            }
        ],
        "Industry_of_Experience": [
            {
                "question": "Which industry do you have the most experience in?",
                "options": [
                    {"text": "Arts", "value": "Arts"},
                    {"text": "Construction", "value": "Construction"},
                    {"text": "Education", "value": "Education"},
                    {"text": "Finance", "value": "Finance"},
                    {"text": "Healthcare", "value": "Healthcare"},
                    {"text": "Law", "value": "Law"},
                    {"text": "Retail", "value": "Retail"},
                    {"text": "Tech", "value": "Tech"}
                ],
                "feature": "Industry_of_Experience"
            }
        ],
        "Internship_Experience": [
            {
                "question": "Do you have any internship experience?",
                "options": [
                    {"text": "Yes", "value": "Yes"},
                    {"text": "No", "value": "No"}
                ],
                "feature": "Internship_Experience"
            }
        ],
        "Remote_Work_Experience": [
            {
                "question": "Do you have experience working remotely?",
                "options": [
                    {"text": "Yes", "value": "Yes"},
                    {"text": "No", "value": "No"}
                ],
                "feature": "Remote_Work_Experience"
            }
        ],
        "LinkedIn_Portfolio": [
            {
                "question": "Do you have a LinkedIn profile or professional portfolio?",
                "options": [
                    {"text": "Yes", "value": "Yes"},
                    {"text": "No", "value": "No"}
                ],
                "feature": "LinkedIn_Portfolio"
            }
        ],
        "Public_Speaking_Experience": [
            {
                "question": "Do you have any public speaking experience?",
                "options": [
                    {"text": "Yes", "value": "Yes"},
                    {"text": "No", "value": "No"}
                ],
                "feature": "Public_Speaking_Experience"
            }
        ]
    }
    
    # Flatten the dictionary into a single list of questions
    all_questions = []
    for category in questions_dict.values():
        all_questions.extend(category)
    
    return all_questions

def get_randomized_questions():
    """Selects 15 random questions from the pool of questions."""
    all_questions = get_all_questions()
    features = list(set(q['feature'] for q in all_questions))
    selected = []

    # First pick one from each feature category
    for feature in features:
        feature_questions = [q for q in all_questions if q['feature'] == feature]
        if feature_questions:
            selected.append(random.choice(feature_questions))

    # Remove selected questions from the pool
    remaining = [q for q in all_questions if q not in selected]

    # Calculate how many more we need to reach 15
    needed = 15 - len(selected)

    # Only sample if we have remaining questions and need more
    if needed > 0 and remaining:
        selected.extend(random.sample(remaining, min(needed, len(remaining))))

    random.shuffle(selected)
    return selected

direct_input_features = {
    "GPA": {
        "question": "What is your approximate GPA (0.0-4.0)?",
        "type": "number", 
        "min": 0.0, 
        "max": 4.0, 
        "step": 0.1, 
        "default": 3.0
    },
    "Years_of_Experience": {
        "question": "Years of professional experience (if any):",
        "type": "number", 
        "min": 0, 
        "max": 50, 
        "step": 1, 
        "default": 0
    },
    "Certifications_Count": {
        "question": "How many professional or technical certifications have you earned?",
        "type": "number", 
        "min": 0, 
        "max": 20, 
        "step": 1, 
        "default": 0
    }
}

# ====================== STREAMLIT APP ======================
def main():
    apply_custom_css()
    
    # Initialize session state
    if 'user_responses' not in st.session_state:
        st.session_state.user_responses = {}
    if 'questions' not in st.session_state:
        st.session_state.questions = get_randomized_questions()

    # Set up page title and description
    st.title("🧭 AI Powered Career Prediction Based on Personality Traits")
    st.markdown("Discover careers that match your unique strengths and preferences.")

    # Sidebar information
    st.sidebar.title("About This Tool")
    st.sidebar.info("This assessment helps match your profile with suitable career options.")
    st.sidebar.write(f"*Based on analysis of {len(data)} career paths*")

    # Create two tabs for different functionalities
    tab1, tab2 = st.tabs(["Take Assessment", "Career Insights"])

    # Assessment Tab
    with tab1:
        st.header("Career Compatibility Assessment")
        st.write("Answer these questions to discover careers that fit your profile.")

        # Background information section
        with st.expander("Your Background"):
            for feature, config in direct_input_features.items():
                st.session_state.user_responses[feature] = st.number_input(
                    config["question"],
                    min_value=config["min"],
                    max_value=config["max"],
                    value=config["default"],
                    step=config["step"],
                    key=f"num_{feature}"
                )

        # Display randomized questions
        st.subheader("Personality and Preferences")
        for i, q in enumerate(st.session_state.questions):
            selected_option = st.radio(
                q["question"],
                [opt["text"] for opt in q["options"]],
                key=f"q_{i}"
            )
            selected_value = q["options"][[opt["text"] for opt in q["options"]].index(selected_option)]["value"]
            st.session_state.user_responses[q["feature"]] = selected_value

        # Prediction button
        if st.button("🔮 Find My Career Match"):
            required_fields = list(direct_input_features.keys()) + ['Interest', 'Work_Style', 'Strengths']
            filled_fields = [field for field in required_fields if field in st.session_state.user_responses]
            
            if len(filled_fields) < 3:
                st.warning("Please answer at least 3 questions (including GPA and Experience) for better results.")
            else:
                with st.spinner("Analyzing your unique profile..."):
                    input_data = processed_data.drop('Predicted_Career_Field', axis=1).iloc[0:1].copy()
                    
                    le_dict = {}
                    for col in data.select_dtypes(include=['object']).columns:
                        if col in data.columns and col != 'Predicted_Career_Field':
                            le = LabelEncoder()
                            le.fit(data[col].astype(str))
                            le_dict[col] = le

                    # Map user responses to model input format
                    for col in input_data.columns:
                        if col in st.session_state.user_responses:
                            if col in ['Communication_Skills', 'Leadership_Skills', 'Teamwork_Skills', 
                                      'Decision_Making', 'Aptitude_Level', 'Adaptability', 
                                      'Time_Management', 'Problem_Solving', 'Emotional_Intelligence',
                                      'Stress_Tolerance', 'Technical_Skill_Level', 'Risk_Tolerance',
                                      'Introvert_Extrovert_Score', 'Work_Life_Balance_Preference',
                                      'Salary_Expectation', 'Openness', 'Conscientiousness',
                                      'Extraversion', 'Agreeableness', 'Neuroticism', 'Creativity_Score']:
                                level_map = {"Low": 0, "Medium": 1, "High": 2, "Moderate": 1}
                                input_data[col] = level_map.get(st.session_state.user_responses[col], 1)
                            elif col in le_dict:
                                try:
                                    input_data[col] = le_dict[col].transform([st.session_state.user_responses[col]])[0]
                                except ValueError:
                                    input_data[col] = processed_data[col].mode()[0]
                            else:
                                # Direct numerical values
                                input_data[col] = st.session_state.user_responses[col]
                        else:
                            input_data[col] = processed_data[col].median()
                    
                    try:
                        # Make prediction
                        prediction = model.predict(input_data)
                        predicted_career = target_le.inverse_transform(prediction)[0]

                        # Display results
                        st.success(f"### Your Best Career Match: **{predicted_career}**")
                        
                        with st.expander("💡 Why this career matches you"):
                            st.write("This career aligns well with your profile because of:")
                            
                            feat_importances = pd.Series(model.feature_importances_, index=input_data.columns)
                            top_features = feat_importances.sort_values(ascending=False).head(3)
                            
                            for feat in top_features.index:
                                importance_desc = ""
                                if feat == "Interest":
                                    interest_val = st.session_state.user_responses.get("Interest", "Various")
                                    importance_desc = f"Your strong interest in {interest_val} fields"
                                elif feat == "Work_Style":
                                    style_val = st.session_state.user_responses.get("Work_Style", "Various")
                                    importance_desc = f"Your preference for {style_val.lower()} work environments"
                                elif feat == "Strengths":
                                    strength_val = st.session_state.user_responses.get("Strengths", "Various")
                                    importance_desc = f"Your {strength_val.lower()} strengths"
                                elif feat == "GPA":
                                    gpa_val = st.session_state.user_responses.get("GPA", 3.0)
                                    importance_desc = f"Your academic performance (GPA: {gpa_val})"
                                elif feat == "Years_of_Experience":
                                    exp_val = st.session_state.user_responses.get("Years_of_Experience", 0)
                                    importance_desc = f"Your professional experience ({exp_val} years)"
                                elif feat == "Certifications_Count":
                                    cert_val = st.session_state.user_responses.get("Certifications_Count", 0)
                                    importance_desc = f"Your professional certifications ({cert_val} certifications)"
                                else:
                                    importance_desc = f"Your responses about {feat.replace('_', ' ').lower()}"
                                
                                st.write(f"- **{importance_desc}** (weight: {top_features[feat]:.2f})")
                            
                            st.write("\nThis career path typically requires these characteristics, which match well with your profile.")
                        
                        with st.expander("📚 Learn more about this career"):
                            career_data = data[data['Predicted_Career_Field'] == predicted_career]
                            if not career_data.empty:
                                st.write(f"**Typical profile for {predicted_career}:**")
                                cols = st.columns(3)
                                with cols[0]:
                                    if 'GPA' in career_data.columns and not career_data['GPA'].isnull().all():
                                        st.metric("Average GPA", f"{career_data['GPA'].mean():.1f}")
                                    else:
                                        st.metric("Average GPA", "N/A")
                                with cols[1]:
                                    if 'Years_of_Experience' in career_data.columns:
                                        st.metric("Avg. Experience", f"{career_data['Years_of_Experience'].mean():.1f} years")
                                    else:
                                        st.metric("Avg. Experience", "N/A")
                                with cols[2]:
                                    if 'Certifications_Count' in career_data.columns:
                                        st.metric("Avg. Certifications", f"{career_data['Certifications_Count'].mean():.1f}")
                                    else:
                                        st.metric("Avg. Certifications", "N/A")
                                
                                st.write("\n**Common characteristics:**")
                                if 'Work_Style' in career_data.columns:
                                    st.write(f"- Work Style: {career_data['Work_Style'].mode()[0]}")
                                if 'Strengths' in career_data.columns:
                                    st.write(f"- Strengths: {career_data['Strengths'].mode()[0]}")
                                if 'Communication_Skills' in career_data.columns:
                                    st.write(f"- Communication: {career_data['Communication_Skills'].mode()[0]}")
                                if 'Technical_Skill_Level' in career_data.columns:
                                    st.write(f"- Technical Skills: {career_data['Technical_Skill_Level'].mode()[0]}")
                            else:
                                st.write("No additional information available for this career in our dataset.")
                        
                    except Exception as e:
                        st.error(f"We encountered an issue analyzing your profile. Please try again.")
    
    with tab2:
        st.header("📊 Career Insights")
        st.write("Explore different career paths and their characteristics.")
        
        if st.checkbox("Show Career Options"):
            if 'Predicted_Career_Field' in data.columns:
                st.dataframe(data['Predicted_Career_Field'].value_counts().reset_index().rename(
                    columns={'index': 'Career', 'Predicted_Career_Field': 'Frequency'}))
            else:
                st.warning("Career field data not available.")
        
        st.subheader("Popular Career Paths")
        if 'Predicted_Career_Field' in data.columns:
            fig, ax = plt.subplots(figsize=(10, 6))
            data['Predicted_Career_Field'].value_counts().head(15).plot(kind='barh', ax=ax, color='skyblue')
            ax.set_title("Most Common Career Paths")
            ax.set_xlabel("Frequency")
            st.pyplot(fig)
        else:
            st.warning("No career path data available.")
        
        st.subheader("Career Characteristics")
        if 'Predicted_Career_Field' in data.columns:
            selected_career = st.selectbox(
                "Select a career to learn more:",
                sorted(data['Predicted_Career_Field'].unique()),
                key="career_select"
            )
            
            career_data = data[data['Predicted_Career_Field'] == selected_career]
            
            if not career_data.empty:
                st.write(f"**Typical profile for {selected_career}:**")
                cols = st.columns(3)
                with cols[0]:
                    if 'GPA' in career_data.columns and not career_data['GPA'].isnull().all():
                        st.metric("Average GPA", f"{career_data['GPA'].mean():.1f}")
                    else:
                        st.metric("Average GPA", "N/A")
                with cols[1]:
                    if 'Years_of_Experience' in career_data.columns:
                        st.metric("Avg. Experience", f"{career_data['Years_of_Experience'].mean():.1f} years")
                    else:
                        st.metric("Avg. Experience", "N/A")
                with cols[2]:
                    if 'Certifications_Count' in career_data.columns:
                        st.metric("Avg. Certifications", f"{career_data['Certifications_Count'].mean():.1f}")
                    else:
                        st.metric("Avg. Certifications", "N/A")

                st.write("\n**Common characteristics:**")
                if 'Work_Style' in career_data.columns:
                    st.write(f"- Work Style: {career_data['Work_Style'].mode()[0]}")
                if 'Strengths' in career_data.columns:
                    st.write(f"- Strengths: {career_data['Strengths'].mode()[0]}")
                if 'Communication_Skills' in career_data.columns:
                    st.write(f"- Communication: {career_data['Communication_Skills'].mode()[0]}")
                if 'Technical_Skill_Level' in career_data.columns:
                    st.write(f"- Technical Skills: {career_data['Technical_Skill_Level'].mode()[0]}")
                if 'Learning_Style' in career_data.columns:
                    st.write(f"- Learning Style: {career_data['Learning_Style'].mode()[0]}")
                if 'Risk_Tolerance' in career_data.columns:
                    st.write(f"- Risk Tolerance: {career_data['Risk_Tolerance'].mode()[0]}")
            else:
                st.warning("No data available for this career.")
        else:
            st.warning("No career data available for analysis.")

if __name__ == "__main__":
    main()
    
