import streamlit as st
import pandas as pd
from utils.data_preprocessing import preprocess_data, categorize_columns, identify_key_columns
from utils.query_handler import interpret_query, execute_code
from utils.visualizations import bar_chart, scatter_plot, histogram, boxplot, heatmap, pairplot
import matplotlib.pyplot as plt

# Streamlit app configuration
st.set_page_config(page_title="Automated Data Visualization System", layout="wide")

def main():
    st.title("Automated Data Visualization System")
    st.write("Upload a dataset, query it in natural language, and generate key visualizations.")

    # Step 1: File Upload
    uploaded_file = st.file_uploader("Upload your dataset (CSV/Excel)", type=["csv", "xlsx"])
    if uploaded_file is not None:
        # Load the dataset
        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith(".xlsx"):
                df = pd.read_excel(uploaded_file)

            st.success("Dataset uploaded successfully!")
            st.dataframe(df.head())  # Display a preview of the dataset

            # Step 2: Preprocess the dataset
            df = preprocess_data(df)
            st.write("### Preprocessed Dataset")
            st.dataframe(df.head())

            # Categorize columns
            categorized_cols = categorize_columns(df)
            key_columns = identify_key_columns(df)

            st.write("### Key Columns")
            st.write(key_columns)

            # Step 3: Automated Visualizations
            st.subheader("Automated Visualizations")
            st.write("Below are some automatically generated visualizations for the key columns:")

            # Generate some key visualizations
            for col in key_columns:
                if col in categorized_cols["numeric"]:
                    st.write(f"### Histogram for {col}")
                    plt = histogram(df, col)
                    st.pyplot(plt)
                elif col in categorized_cols["categorical"]:
                    st.write(f"### Bar Chart for {col}")
                    plt = bar_chart(df, col)
                    st.pyplot(plt)

            if len(categorized_cols["numeric"]) > 1:
                st.write("### Correlation Heatmap")
                plt = heatmap(df)
                st.pyplot(plt)

                st.write("### Pairplot for Numeric Columns")
                plt = pairplot(df, categorized_cols["numeric"])
                st.pyplot(plt)

            # Step 4: Query the dataset
            st.subheader("Natural Language Query")
            query = st.text_input("Ask a question about the dataset (e.g., 'How many males and females are there?')")
            if query:
                # Send query to OpenAI API
                with st.spinner("Processing your query..."):
                    code = interpret_query(df, query)
                
                st.code(code, language="python")  # Show the generated code

                # Execute the code and display the result
                try:
                    plt = execute_code(code)
                    if plt:
                        st.pyplot(plt)
                    else:
                        st.error("No visualization was generated.")
                except Exception as e:
                    st.error(f"Error executing code: {e}")
        except Exception as e:
            st.error(f"Error processing the file: {e}")
    else:
        st.info("Please upload a dataset to get started.")

if __name__ == "__main__":
    main()
