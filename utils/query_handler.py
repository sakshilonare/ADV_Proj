import openai
from dotenv import load_dotenv # type: ignore
import os

load_dotenv()
# Set your OpenAI API key here
api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = api_key

def interpret_query(df, query):
    """
    Interpret a natural language query and generate Python code to visualize the data.
    """
    # Convert dataset preview to a string format for context
    dataset_preview = df.head().to_dict()
    
    # Send prompt to OpenAI
    prompt = f"""
    Given the dataset: {dataset_preview}
    
    Generate a Python code snippet to visualize the answer to this query: "{query}"
    The code should use pandas and matplotlib or seaborn for visualization. 
    """
    try:
        response = openai.Completion.create(
            model="text-davinci-003",
            prompt=prompt,
            max_tokens=300,
            temperature=0
        )
        # Extract the generated code
        generated_code = response['choices'][0]['text'].strip()
        return generated_code
    except Exception as e:
        return f"Error in query processing: {e}"

def execute_code(code):
    """
    Execute a Python code snippet and return the generated plot.
    """
    local_scope = {}
    try:
        # Execute the code
        exec(code, globals(), local_scope)
        return local_scope.get("plt", None)  # Return the matplotlib plot object
    except Exception as e:
        raise RuntimeError(f"Error in executing code: {e}")
