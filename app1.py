import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectFromModel

# -----------------------------
# Set page config
# -----------------------------
st.set_page_config(
    page_title="Career Path Predictor",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# Custom CSS Styling
# -----------------------------
st.markdown("""
<style>
    /* Main container styling */
    .stApp {
        background-color: #f5f7fa;
        background-image: linear-gradient(135deg, #f5f7fa 0%, #e4e8eb 100%);
    }
    
    /* Header styling */
    .stMarkdown h1 {
        color: #2c3e50;
        text-align: center;
        padding-bottom: 15px;
        border-bottom: 2px solid #3498db;
        margin-bottom: 30px;
    }
    
    /* Button styling */
    .stButton>button {
        background-color: #3498db;
        color: white;
        border-radius: 8px;
        padding: 10px 24px;
        border: none;
        font-weight: 500;
        transition: all 0.3s;
        width: 100%;
    }
    
    .stButton>button:hover {
        background-color: #2980b9;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    /* Form styling */
    .stForm {
        background-color: white;
        border-radius: 12px;
        padding: 25px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    
    /* Prediction card */
    .prediction-card {
        background: linear-gradient(135deg, #3498db, #2c3e50);
        color: white;
        padding: 30px;
        border-radius: 12px;
        text-align: center;
        margin: 30px auto;
        box-shadow: 0 8px 24px rgba(0,0,0,0.15);
        max-width: 800px;
    }
    
    /* Trait bubbles */
    .trait-bubble {
        display: inline-block;
        background-color: #3498db;
        color: white;
        padding: 8px 16px;
        border-radius: 20px;
        margin: 5px;
        font-size: 14px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.1);
    }
    
    /* Suggestion cards */
    .suggestion-card {
        background-color: white;
        border-radius: 8px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        border-left: 4px solid #3498db;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Load data
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_excel("Book1.xlsx", sheet_name="Sheet1")
    return df

# -----------------------------
# Level Mapping
# -----------------------------
level_mapping = {"Low": 0, "Medium": 1, "High": 2}

# -----------------------------
# Enhanced Preprocessing
# -----------------------------
def preprocess_data(df):
    df = df.copy()
    le_dict = {}
    
    # First pass: Identify all possible categories for each column
    category_mapping = {}
    for col in df.columns:
        if df[col].dtype == 'object' and col != "Predicted_Career_Field":
            # Convert to string in case there are mixed types
            unique_values = df[col].astype(str).unique()
            category_mapping[col] = list(unique_values)
    
    # Second pass: Create label encoders with all known categories
    for col in df.columns:
        if df[col].dtype == 'object' and col != "Predicted_Career_Field":
            le = LabelEncoder()
            # Ensure we're working with strings and handle NaN/None values
            categories = [str(x) for x in category_mapping[col]]
            le.fit(categories)
            df[col] = le.transform(df[col].astype(str))
            le_dict[col] = le
    
    target_le = LabelEncoder()
    df["Predicted_Career_Field"] = target_le.fit_transform(df["Predicted_Career_Field"].astype(str))
    
    return df, le_dict, target_le, category_mapping

# -----------------------------
# Train Model with Feature Selection
# -----------------------------
def train_model(X, y, n_features=10):
    # First train to get feature importances
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X, y)
    
    # Select top N features
    selector = SelectFromModel(clf, max_features=n_features, threshold=-np.inf)
    selector.fit(X, y)
    selected_features = X.columns[selector.get_support()]
    
    # Retrain with selected features
    X_reduced = X[selected_features]
    X_train, X_test, y_train, y_test = train_test_split(
        X_reduced, y, test_size=0.2, random_state=42
    )
    clf.fit(X_train, y_train)
    
    return clf, selected_features

# -----------------------------
# Career Descriptions
# -----------------------------
career_descriptions = {
    "Software Engineer": {
        "description": "Software engineers design, develop, and maintain software systems. They solve complex problems with code and create applications that power our digital world.",
        "traits": ["Analytical", "Problem-solving", "Detail-oriented", "Logical"],
        "suggestions": [
            "Learn a new programming language every year",
            "Contribute to open-source projects",
            "Attend tech meetups and hackathons",
            "Develop strong algorithms knowledge"
        ]
    },
    "Data Scientist": {
        "description": "Data scientists extract insights from complex data sets. They use statistical analysis and machine learning to help organizations make data-driven decisions.",
        "traits": ["Curious", "Mathematical", "Visual thinker", "Storytelling"],
        "suggestions": [
            "Master Python and R for data analysis",
            "Learn advanced statistics and machine learning",
            "Build a portfolio of data projects",
            "Develop business acumen to complement technical skills"
        ]
    },
    "Marketing Manager": {
        "description": "Marketing managers develop strategies to promote products and services. They analyze market trends and oversee campaigns to reach target audiences effectively.",
        "traits": ["Creative", "Strategic", "Communication", "Adaptable"],
        "suggestions": [
            "Stay updated with digital marketing trends",
            "Develop strong analytical skills",
            "Build expertise in consumer psychology",
            "Gain experience with marketing automation tools"
        ]
    },
    "Financial Analyst": {
        "description": "Financial analysts assess financial data to help businesses make investment decisions. They evaluate economic trends and company performance to provide recommendations.",
        "traits": ["Detail-oriented", "Numerical", "Risk-aware", "Organized"],
        "suggestions": [
            "Obtain financial certifications (CFA, CPA)",
            "Develop advanced Excel modeling skills",
            "Stay informed about global markets",
            "Improve presentation and reporting skills"
        ]
    },
    "Graphic Designer": {
        "description": "Graphic designers create visual concepts to communicate ideas. They combine art and technology to produce designs that inspire, inform, and captivate consumers.",
        "traits": ["Creative", "Visual", "Patient", "Trend-aware"],
        "suggestions": [
            "Master industry-standard design software",
            "Build a diverse portfolio",
            "Stay updated with design trends",
            "Develop basic web development skills"
        ]
    },
    "Teacher": {
        "description": "Teachers educate and inspire students of all ages. They develop lesson plans, assess student progress, and create engaging learning environments.",
        "traits": ["Patient", "Communicative", "Empathetic", "Adaptable"],
        "suggestions": [
            "Develop classroom management techniques",
            "Learn innovative teaching methodologies",
            "Stay current with educational technology",
            "Specialize in a subject area"
        ]
    },
    "Doctor": {
        "description": "Doctors diagnose and treat illnesses and injuries. They examine patients, take medical histories, and prescribe medications or treatments.",
        "traits": ["Compassionate", "Detail-oriented", "Resilient", "Problem-solver"],
        "suggestions": [
            "Develop strong bedside manner",
            "Stay current with medical research",
            "Build teamwork and leadership skills",
            "Practice continuous learning throughout your career"
        ]
    },
    "Lawyer": {
        "description": "Lawyers advise and represent individuals, businesses, and government agencies on legal issues and disputes. They interpret laws and apply them to specific situations.",
        "traits": ["Analytical", "Persuasive", "Detail-oriented", "Ethical"],
        "suggestions": [
            "Develop strong research and writing skills",
            "Build expertise in a specific legal area",
            "Practice public speaking and debate",
            "Stay updated with changes in legislation"
        ]
    }
}

# -----------------------------
# Questions dictionary (truncated for brevity - same as before)
# -----------------------------
questions_dict = {
    # ... (same questions dictionary as in your original code)
}

# -----------------------------
# Ask Questions
# -----------------------------
def ask_questions(features):
    st.subheader("Career Assessment")
    user_input = {}
    
    # Initialize session state for selected questions if not exists
    if 'selected_questions' not in st.session_state:
        st.session_state.selected_questions = {}
    
    # Process all features in order
    for feature in features:
        # Check if feature has questions in dictionary
        if feature in questions_dict and len(questions_dict[feature]) > 0:
            # If we haven't selected a question for this feature yet, pick one randomly
            if feature not in st.session_state.selected_questions:
                st.session_state.selected_questions[feature] = np.random.choice(questions_dict[feature])
            
            # Get the randomly selected question
            qa = st.session_state.selected_questions[feature]
            question = qa["question"]
            options = list(qa["options"].keys())
            
            # Display the question and get response
            response = st.radio(question, options, key=f"q_{feature}")
            level = qa["options"][response]
            user_input[feature] = level_mapping.get(level, level)
        else:
            # Special handling for specific fields
            if feature == "GPA":
                user_input[feature] = st.number_input(
                    f"What is your {feature.replace('_', ' ')}?",
                    min_value=0.0, max_value=4.0, value=3.0, step=0.1,
                    key=f"num_{feature}"
                )
            elif feature == "Years_of_Experience":
                user_input[feature] = st.number_input(
                    f"How many years of {feature.replace('_', ' ').lower()} do you have?",
                    min_value=0, max_value=50, value=2, step=1,
                    key=f"num_{feature}"
                )
            elif feature == "Certifications_Count":
                user_input[feature] = st.number_input(
                    f"How many {feature.replace('_', ' ').lower()} do you have?",
                    min_value=0, max_value=100, value=2, step=1,
                    key=f"num_{feature}"
                )
            elif feature == "Field_of_Study":
                user_input[feature] = st.selectbox(
                    "What is your field of study?",
                    options=["Accounting", "Computer Science", "Medicine", "Law", "Fine Arts", 
                            "Education", "Engineering", "Business Administration", "Psychology"],
                    key=f"sel_{feature}"
                )
            elif feature == "Highest_Degree":
                user_input[feature] = st.selectbox(
                    "What is your highest degree?",
                    options=["Diploma", "Bachelors", "Masters", "PhD"],
                    key=f"sel_{feature}"
                )
            else:
                # For other features that don't have questions in the dict, show a warning
                st.warning(f"No question available for feature: {feature}")
                # Default to medium level if we must proceed
                user_input[feature] = 1
    
    return user_input

# -----------------------------
# Main App
# -----------------------------
def main():     
    # Header with logo
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        st.title("🧭 Career Path Predictor")
        st.markdown("""
        <div style="text-align: center; color: #7f8c8d; margin-bottom: 30px;">
            Discover your ideal career based on your skills, preferences, and personality
        </div>
        """, unsafe_allow_html=True)
    
    # Load data
    df = load_data()

    # Enhanced preprocessing
    df_processed, le_dict, target_le, category_mapping = preprocess_data(df)

    # Prepare features and target
    X = df_processed.drop("Predicted_Career_Field", axis=1)
    y = df_processed["Predicted_Career_Field"]

    # Train model with feature selection (keeping top 30 features)
    model, selected_features = train_model(X, y, n_features=30)
    
    # Get user input
    with st.form("career_form"):
        user_input = ask_questions(selected_features)
        
        # Form submit button
        submit_button = st.form_submit_button("🔮 Predict My Career", type="primary")

    # Handle form submission
    if submit_button:
        if len(user_input) == len(selected_features):
            # Convert user input to DataFrame with selected features
            input_df = pd.DataFrame([user_input], columns=selected_features)
            
            # Enhanced encoding handling
            for col in input_df.columns:
                if col in le_dict:
                    if isinstance(user_input[col], str):
                        if user_input[col] in category_mapping[col]:
                            input_df[col] = le_dict[col].transform([user_input[col]])[0]
                        else:
                            input_df[col] = 0  # Default value
                    else:
                        input_df[col] = user_input[col]
                elif isinstance(user_input[col], str):
                    input_df[col] = 0  # Default for unencoded strings
            
            # Make prediction
            try:
                prediction = model.predict(input_df)[0]
                predicted_career = target_le.inverse_transform([prediction])[0]
                
                # Get career details
                career_info = career_descriptions.get(predicted_career, {
                    "description": "This career path offers exciting opportunities to apply your skills and interests.",
                    "traits": [],
                    "suggestions": []
                })
                
                # Display prediction in a styled card
                st.markdown(f"""
                <div class="prediction-card">
                    <h2 style="color: white; margin-bottom: 15px;">Your Career Prediction</h2>
                    <p style="font-size: 32px; font-weight: bold; margin-bottom: 20px;">{predicted_career}</p>
                    <p style="font-size: 18px;">{career_info['description']}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Display traits
                if career_info['traits']:
                    st.subheader("Your Key Traits")
                    st.markdown("<div style='text-align: center; margin-bottom: 20px;'>" + 
                               "".join([f"<span class='trait-bubble'>{trait}</span>" for trait in career_info['traits']]) + 
                               "</div>", unsafe_allow_html=True)
                
                # Display suggestions
                if career_info['suggestions']:
                    st.subheader("Career Development Suggestions")
                    for suggestion in career_info['suggestions']:
                        st.markdown(f"""
                        <div class="suggestion-card">
                            <p style="margin: 0; font-size: 16px;">✓ {suggestion}</p>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Confidence indicator
                st.markdown("---")
                st.markdown("""
                <div style="text-align: center; color: #7f8c8d; font-size: 14px; margin-top: 20px;">
                    Note: This prediction is based on your responses and our career matching algorithm. 
                    Consider it as one potential path among many possibilities that might suit you.
                </div>
                """, unsafe_allow_html=True)
                
            except Exception as e:
                st.error("We encountered an issue generating your prediction. Please try again.")
        else:
            st.error("Please answer all questions before predicting.")

if __name__ == "__main__":
    main()
