import streamlit as st
import pandas as pd
from utils.data_preprocessing import preprocess_data, categorize_columns, identify_key_columns, categorize_time_series_columns
from utils.query_handler import interpret_query, execute_code
from utils.visualizations import bar_chart, scatter_plot, histogram, boxplot, heatmap, pairplot, sankey_diagram, wordcloud_plot, time_series_plot, regression_plot
import matplotlib.pyplot as plt
from PIL import Image
import csv

# Prevent decompression bomb errors
Image.MAX_IMAGE_PIXELS = None

# Resize image function
def resize_image(image_path, max_width=1024, max_height=1024):
    try:
        with Image.open(image_path) as img:
            img.thumbnail((max_width, max_height))
            return img
    except Exception as e:
        st.error(f"Error resizing image {image_path}: {e}")
        return None

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
                content = uploaded_file.getvalue().decode("utf-8")

                sniffer = csv.Sniffer()
                detected_delimiter =  sniffer.sniff(content).delimiter
                df = pd.read_csv(uploaded_file,delimiter=detected_delimiter)
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
            time_Series_col = categorize_time_series_columns(df)

            st.write("### Key Columns")
            st.write(key_columns)

            st.write(categorized_cols["numeric"])

            # Step 3: User Selection for Visualizations
            st.subheader("Select Visualizations")
            selected_visualizations = []

            # Checkboxes for available visualizations
            if len(categorized_cols["numeric"]) > 0:
                if st.checkbox("Generate Histograms for Numeric Columns"):
                    selected_visualizations.append("histograms")
                
            if len(categorized_cols["categorical"]) > 0:
                if st.checkbox("Generate Bar Charts for Categorical Columns"):
                    selected_visualizations.append("bar_charts")
                
                if st.checkbox("Generate Word Cloud for Categorical Columns"):
                    selected_visualizations.append("wordclouds")
                
                if st.checkbox("Generate Boxplots for Categorical Columns"):
                    selected_visualizations.append("boxplots")

            if len(categorized_cols["numeric"]) > 1:
                if st.checkbox("Generate Correlation Heatmap"):
                    selected_visualizations.append("heatmap")
                
                if st.checkbox("Generate Pairplot for Numeric Columns"):
                    selected_visualizations.append("pairplot")

            if len(time_Series_col) > 0:
                if st.checkbox("Generate Time Series Plot"):
                    selected_visualizations.append("time_series")

            if 'numeric_col1' in df.columns and 'numeric_col2' in df.columns:
                if st.checkbox("Generate Regression Plot"):
                    selected_visualizations.append("regression")

            if 'source_column' in df.columns and 'target_column' in df.columns:
                if st.checkbox("Generate Sankey Diagram"):
                    selected_visualizations.append("sankey")

            # Generate the selected visualizations
            histograms_count = 0
            bar_charts_count = 0
            wordcloud_count = 0
            box_plot_count = 0

            for col in key_columns:
                if "histograms" in selected_visualizations and col in categorized_cols["numeric"] and histograms_count < 3:
                    st.write(f"### Histogram for {col}")
                    plt = histogram(df, col)
                    st.pyplot(plt, use_container_width=True)
                    histograms_count += 1

                elif "bar_charts" in selected_visualizations and col in categorized_cols["categorical"] and bar_charts_count < 3:
                    st.write(f"### Bar Chart for {col}")
                    plt = bar_chart(df, col)
                    st.pyplot(plt, use_container_width=True)
                    bar_charts_count += 1

                elif "wordclouds" in selected_visualizations and col in categorized_cols["categorical"] and wordcloud_count < 3:
                    st.write("### Word Cloud")
                    plt = wordcloud_plot(df[col])
                    st.pyplot(plt, use_container_width=True)
                    wordcloud_count += 1

                elif "boxplots" in selected_visualizations:
                    for cat_col in categorized_cols["categorical"]:
                        if box_plot_count < 3:
                            st.write(f"### Boxplot of {col} by {cat_col}")
                            plt = boxplot(df, cat_col, col)
                            st.pyplot(plt, use_container_width=True)  
                            box_plot_count += 1

            if "heatmap" in selected_visualizations and len(categorized_cols["numeric"]) > 1:
                st.write("### Correlation Heatmap")
                plt = heatmap(df)
                st.pyplot(plt, use_container_width=True)

            if "pairplot" in selected_visualizations and len(categorized_cols["numeric"]) > 1:
                st.write("### Pairplot for Numeric Columns")
                plt = pairplot(df, categorized_cols["numeric"])
                st.pyplot(plt, use_container_width=True)
            
            if "scatter_plot" in selected_visualizations and len(categorized_cols["numeric"]) > 1:
                st.write("### Scatter Plots for Numeric Columns")
                numeric_columns = categorized_cols["numeric"]
                for i in range(len(numeric_columns)):
                    for j in range(i + 1, len(numeric_columns)):
                        x_col = numeric_columns[i]
                        y_col = numeric_columns[j]
                        st.write(f"Scatter Plot: {x_col} vs {y_col}")
                        plt = scatter_plot(df, x_col, y_col)
                        st.pyplot(plt, use_container_width=True)

            if "time_series" in selected_visualizations and "year" in df.columns:
                st.write("### Time Series Plot")

                # Use "year" as the time column
                time_col = "year"
                numeric_cols = categorized_cols["numeric"]
                numeric_col = st.selectbox("Select a numeric column for visualization:", numeric_cols)

                if numeric_col:
                    try:
                        # Ensure the year column is treated as numeric or datetime
                        if not pd.api.types.is_numeric_dtype(df[time_col]):
                            df[time_col] = pd.to_numeric(df[time_col], errors='coerce')

                        # Drop rows with invalid year values
                        df = df.dropna(subset=[time_col])

                        # Sort the dataframe by the year column
                        df = df.sort_values(by=time_col)

                        # Generate the time series plot
                        plt = time_series_plot(df, time_col, numeric_col)
                        st.pyplot(plt, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error generating time series plot: {e}")

            if "regression" in selected_visualizations and 'numeric_col1' in df.columns and 'numeric_col2' in df.columns:
                st.write("### Regression Plot")
                plt = regression_plot(df, 'numeric_col1', 'numeric_col2')
                st.pyplot(plt)

            if "sankey" in selected_visualizations and 'source_column' in df.columns and 'target_column' in df.columns:
                st.write("### Sankey Diagram")
                sankey_fig = sankey_diagram(
                    df['source_column'].tolist(),
                    df['target_column'].tolist(),
                    df['value_column'].tolist()
                )
                st.plotly_chart(sankey_fig)

            # Handle images (resize if necessary)
            if 'image_column' in df.columns:
                st.write("### Image Resizing and Display")
                for image_path in df['image_column']:
                    resized_img = resize_image(image_path)
                    if resized_img:
                        st.image(resized_img, caption=image_path)

            # Step 4: Query the dataset
            st.subheader("Natural Language Query")
            query = st.text_input("Ask a question about the dataset (e.g., 'How many males and females are there?')")
            if query:
                # Send query to OpenAI API
                with st.spinner("Processing your query..."):
                    code = interpret_query(df, query)
                
                st.code(code, language="python") 
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
