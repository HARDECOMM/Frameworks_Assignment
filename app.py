import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os # Import os module to check for file existence

# Set Streamlit page configuration (optional)
st.set_page_config(layout="wide")

# --- App Title and Description ---
st.title("COVID-19 Vaccination Data Explorer")
st.write("Exploring vaccination data across countries based on the loaded dataset.")

# --- Data Loading ---
# Define the path to the dataset file
# In a real deployment, you might use st.file_uploader or have the data in a known location
DATA_FILE_PATH = 'country_vaccinations.csv'

# Function to load data (with caching for performance)
@st.cache_data # Cache the data so it's loaded only once
def load_data(file_path):
    """
    Loads a CSV dataset into a pandas DataFrame.

    Args:
      file_path: The path to the CSV file.

    Returns:
      A pandas DataFrame containing the data, or None if an error occurs.
    """
    # Check if the file exists
    if not os.path.exists(file_path):
        st.error(f"Error: Data file not found at {file_path}. Please make sure the file is in the correct directory.")
        return None
    try:
        df = pd.read_csv(file_path)
        # Ensure 'date' column is in datetime format
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        # Drop rows where date conversion failed (if any)
        df.dropna(subset=['date'], inplace=True)
        return df
    except Exception as e:
        st.error(f"An error occurred while loading the data: {e}")
        return None

# Load the data using the defined function
df = load_data(DATA_FILE_PATH)

# --- Data Cleaning and Preparation (applied if data loaded) ---
if df is not None:
    # Add a sidebar for data cleaning options (optional)
    st.sidebar.header("Data Cleaning Options")
    # Basic cleaning: Fill missing numerical values with 0 and drop rows with missing daily_vaccinations
    df_cleaned = df.copy()

    # Define numerical columns to fill missing values
    numerical_cols = ['total_vaccinations', 'people_vaccinated', 'people_fully_vaccinated',
                      'daily_vaccinations_raw', 'daily_vaccinations',
                      'total_vaccinations_per_hundred', 'people_vaccinated_per_hundred',
                      'people_fully_vaccinated_per_hundred', 'daily_vaccinations_per_million']

    # Fill missing numerical values with 0
    for col in numerical_cols:
        if col in df_cleaned.columns:
            df_cleaned[col] = df_cleaned[col].fillna(0) # Use non-inplace operation

    # Drop rows with missing daily_vaccinations as it's key for analysis
    df_cleaned.dropna(subset=['daily_vaccinations'], inplace=True)

    # Convert 'date' column to datetime objects for time-based analysis
    if 'date' in df_cleaned.columns:
        df_cleaned['year'] = df_cleaned['date'].dt.year
        df_cleaned['month'] = df_cleaned['date'].dt.month

    st.sidebar.write("Data cleaning applied: Missing numerical values filled with 0, rows with missing daily vaccinations dropped.")


    # --- Data Exploration and Visualization ---
    st.header("Data Sample and Overview")
    # Display the first few rows of the cleaned data
    st.write(df_cleaned.head())
    # Display the shape of the cleaned DataFrame
    st.write(f"Dataset shape: {df_cleaned.shape}")
    # Display data information (column types, non-null counts)
    st.write("Data Info:")
    st.text(df_cleaned.info())
    # Display missing values after cleaning
    st.write("Missing values after cleaning:")
    st.write(df_cleaned.isnull().sum())

    # Plotting Daily Vaccinations Over Time Across Selected Countries
    st.header("Daily Vaccinations Over Time Across Selected Countries")
    # Get list of unique countries for multiselect widget
    available_countries = df_cleaned['country'].unique().tolist()
    default_countries = ['United States', 'India', 'Brazil', 'United Kingdom']
    # Filter default countries to only include those available in the dataset
    countries_to_compare = st.multiselect(
        "Select countries to compare (Daily Vaccinations):",
        available_countries,
        default=[country for country in default_countries if country in available_countries]
    )

    # Create and display the line plot if countries are selected
    if countries_to_compare:
        comparison_df = df_cleaned[df_cleaned['country'].isin(countries_to_compare)].copy()
        if not comparison_df.empty:
            comparison_df = comparison_df.sort_values(by='date')
            fig1, ax1 = plt.subplots(figsize=(14, 7))
            sns.lineplot(data=comparison_df, x='date', y='daily_vaccinations', hue='country', ax=ax1)
            ax1.set_title('Daily Vaccinations Over Time Across Selected Countries')
            ax1.set_xlabel('Date')
            ax1.set_ylabel('Daily Vaccinations')
            plt.xticks(rotation=45)
            ax1.legend(title='Country')
            plt.tight_layout()
            st.pyplot(fig1)
            plt.close(fig1) # Close the figure to prevent it from displaying twice

        else:
            st.write("Data for the selected countries not found after filtering.")
    else:
        st.write("Please select at least one country to compare daily vaccinations.")


    # Plotting Top Countries by Total Vaccinations
    st.header("Top Countries by Total Vaccinations")
    # Group by country and sum total_vaccinations
    total_vaccinations_by_country = df_cleaned.groupby('country')['total_vaccinations'].sum().reset_index()
    # Sort by total_vaccinations in descending order
    top_countries_total_vaccinations = total_vaccinations_by_country.sort_values(by='total_vaccinations', ascending=False)

    # Slider to select number of top countries to display
    top_n_countries = st.slider("Select number of top countries to display (Total Vaccinations):", 5, min(20, top_countries_total_vaccinations.shape[0]), 10)

    # Get the top N countries data
    top_countries_plot_data = top_countries_total_vaccinations.head(top_n_countries)

    # Create and display the bar chart
    if not top_countries_plot_data.empty:
        fig2, ax2 = plt.subplots(figsize=(12, 7))
        sns.barplot(x='total_vaccinations', y='country', data=top_countries_plot_data, palette='viridis', ax=ax2)
        ax2.set_title(f'Top {top_n_countries} Countries by Total Vaccinations')
        ax2.set_xlabel('Total Vaccinations')
        ax2.set_ylabel('Country')
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2) # Close the figure
    else:
        st.write("No data available to display top countries by total vaccinations.")

    # Add more visualizations here based on your analysis (e.g., people vaccinated, daily vaccinations per million)
    st.header("Daily Vaccinations Per Million Over Time Across Selected Countries")
    # Reuse the countries_to_compare from the previous multiselect for consistency
    if countries_to_compare:
        comparison_df_per_million = df_cleaned[df_cleaned['country'].isin(countries_to_compare)].copy()
        if not comparison_df_per_million.empty:
            comparison_df_per_million = comparison_df_per_million.sort_values(by='date')
            fig3, ax3 = plt.subplots(figsize=(14, 7))
            sns.lineplot(data=comparison_df_per_million, x='date', y='daily_vaccinations_per_million', hue='country', ax=ax3)
            ax3.set_title('Daily Vaccinations Per Million Over Time Across Selected Countries')
            ax3.set_xlabel('Date')
            ax3.set_ylabel('Daily Vaccinations Per Million')
            plt.xticks(rotation=45)
            ax3.legend(title='Country')
            plt.tight_layout()
            st.pyplot(fig3)
            plt.close(fig3)
        else:
             st.write("Data for the selected countries not found after filtering (Daily Vaccinations Per Million).")
    else:
        st.write("Please select at least one country to compare daily vaccinations per million.")


else:
    # Message to display if data was not loaded
    st.warning("Data not loaded. Please check the file path and try again.")