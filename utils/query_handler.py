import google.generativeai as genai
from dotenv import load_dotenv
import os
import streamlit as st

# Load the API key from the environment
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Initialize the generative model
model = genai.GenerativeModel("gemini-1.5-flash")

def interpret_query(df, query):
    """
    Interpret a natural language query and generate Python code to visualize the data.
    """
    # Convert dataset preview to a string format for context
    data_structure = {col: str(dtype) for col, dtype in df.dtypes.items()}

    # Craft the prompt for Google Generative AI
    prompt = prompt = f"""
    The DataFrame has the following columns and data types:\n{str(data_structure)}.\n
    Generate a Python code snippet to visualize the answer to this query: "{query}". 
    The code should use pandas and matplotlib or seaborn for visualization. It should be a single code which is executable directly. I will be using st.pyplot(local_scope['plt']). Also in the local_scope, I am passing my dataframe df, it must use that.
    local_scope = {{'df': df}}.exec(code, globals(), local_scope). Dont explain. Use df directly assuming it is present.
    """
    
    try:
        # Use Gemini to generate the response
        response = model.generate_content(prompt)

        # Extract and return the generated code
        generated_code = response.text
        return generated_code

    except Exception as e:
        return f"Error in query processing: {e}"

def execute_code(code, df):
    """
    Execute a Python code snippet and return the generated plot.
    """
    local_scope = {'df': df}  # Pass the df DataFrame into the local scope
    try:
        
        code = code.strip().splitlines()
        if len(code) > 2:
            code = "\n".join(code[1:-1])  # Remove first and last lines (assuming they are not part of the code)

        exec(code, globals(), local_scope)

        # Check if the plot object (plt) is in local_scope and return it
        if 'plt' in local_scope:
            # Display the plot using Streamlit's st.pyplot()
            st.pyplot(local_scope['plt'])
            return "Plot displayed successfully."
        else:
            return "No plot generated in the code."
    
    except Exception as e:
        st.error(f"Error in executing code in execute_code: {e}")
        return None

