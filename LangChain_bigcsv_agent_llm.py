'''
This script reads data on the website, performs statistical analysis, and plot figures. 
Below are input examples (via Q&A) for running the script: 
# url: https://www2.census.gov/programs-surveys/popest/datasets/2020-2023/state/asrh/sc-est2023-alldata6.csv
# var: POPESTIMATE2023 (or any numeric column)
# cat: RACE (or STATE, SEX, etc.)
'''

from langchain_core.tools import tool
import pandas as pd
import matplotlib.pyplot as plt

from langchain.tools import Tool
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@tool
def bigdata(url: str, cat: str, var: str):
    """
    Fetch data from a website URL and analyze it.
    Args:
        url (str): The URL of the website to fetch data from
        cat (str): Category column name
        var (str): Variable column name to analyze      
    Returns:
        str: Analysis results
    """
    try:
        print(f"📥 Loading data from: {url}")
        data = pd.read_csv(url)
        num_variables = len(data.columns)
        name_variables = data.columns.tolist()
        
        print(f"✅ Data loaded successfully!")
        print(f"📊 Dataset info:")
        print(f"   - Shape: {data.shape}")
        print(f"   - Columns: {num_variables}")
        print(f"   - Column names: {name_variables}")
        
        # Check if columns exist
        if cat not in data.columns:
            return f"❌ Error: Column '{cat}' not found in the dataset.\n📋 Available columns: {name_variables}"
        if var not in data.columns:
            return f"❌ Error: Column '{var}' not found in the dataset.\n📋 Available columns: {name_variables}"
        
        # Check if var column is numeric
        if not pd.api.types.is_numeric_dtype(data[var]):
            return f"❌ Error: Column '{var}' is not numeric. Please select a numeric column for analysis.\n📋 Numeric columns: {data.select_dtypes(include=[int, float]).columns.tolist()}"
        
        print(f"🔢 Analyzing '{var}' by '{cat}'...")
        cat_sums = data.groupby(cat)[var].sum()
        total_value = data[var].sum()
        cat_percentages = (cat_sums / total_value) * 100
        
        print(f"📈 Creating visualizations...")
        # Create visualizations
        plt.figure(figsize=(12, 6))
        cat_percentages.plot(kind='bar')
        plt.ylabel('Percentage (%)')
        plt.title(f'Percentage of {var} by {cat}')
        plt.xlabel(cat)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

        # Plot the pie chart
        plt.figure(figsize=(8, 8))
        plt.pie(cat_percentages, labels=cat_percentages.index, autopct='%1.1f%%', startangle=140)
        plt.title(f'Distribution of {var} by {cat}')
        plt.axis('equal')
        plt.show()
        
        result = (
            f"📊 Analysis Results:\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"🌐 URL analyzed: {url}\n"
            f"📏 Dataset dimensions: {data.shape[0]} rows × {data.shape[1]} columns\n"
            f"🔢 Variable analyzed: {var}\n"
            f"🏷️ Category: {cat}\n"
            f"💯 Total sum of '{var}': {total_value:,.0f}\n\n"
            f"📈 Breakdown by {cat}:\n"
            f"{cat_percentages.to_string()}\n\n"
            f"🎨 Visualizations: Bar chart and pie chart have been displayed."
        )
        
        return result
        
    except Exception as e:
        return f"Error processing data: {str(e)}"

# Initialize the LLM
llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-Coder-32B-Instruct",
    huggingfacehub_api_token='YOUR_HUGGINGFACE_TOKEN',
)

chat = ChatHuggingFace(llm=llm, verbose=True)

tools = [bigdata]
chat_with_tools = chat.bind_tools(tools)

# use the tool
print("🤖 Welcome to the Big Data Analysis Agent!")
print("📊 I can help you fetch data from a website, perform statistical analysis, and create visualizations.\n")

try:
    url = input("🌐 Please enter the website URL to read the CSV file: ")
    
    if not url.strip():
        print("❌ No URL provided. Using the sample dataset...")
        url = "https://www2.census.gov/programs-surveys/popest/datasets/2020-2023/state/asrh/sc-est2023-alldata6.csv"
    
    # First, let's check the data structure
    print("\n� Loading data preview...")
    data = pd.read_csv(url)
    num_variables = len(data.columns)
    name_variables = data.columns.tolist()
    
    print(f"✅ Data loaded successfully!")
    print(f"📊 Dataset Information:")
    print(f"   • Shape: {data.shape[0]} rows × {data.shape[1]} columns")
    print(f"   • Available columns: {name_variables}")
    
    # Show sample data
    print(f"\n📝 First 3 rows of data:")
    print(data.head(3).to_string())
    
    # Show numeric columns
    numeric_cols = data.select_dtypes(include=[int, float]).columns.tolist()
    print(f"\n🔢 Numeric columns (for analysis): {numeric_cols}")
    
    # Show categorical columns  
    categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
    print(f"🏷️  Categorical columns (for grouping): {categorical_cols}")
    
    var = input(f"\n📈 Please select a NUMERIC variable for analysis: ")
    cat = input("🏷️  Please select a CATEGORICAL variable for grouping: ")
    
    print("\n🔍 Starting analysis...")
    # Call the tool function directly
    result = bigdata.invoke({"url": url, "cat": cat, "var": var})
    #result = chat_with_tools.invoke(bigdata, {"url": url, "cat": cat, "var": var})
    print("\n" + "="*50)
    print("📈 FINAL RESULTS")
    print("="*50)
    print(result)
    
except KeyboardInterrupt:
    print("\n\n👋 Analysis cancelled by user. Goodbye!")

except Exception as e:
    print(f"\n❌ An error occurred: {e}")
    print("\n💡 Tips:")
    print("   • Make sure the URL is valid and points to a CSV file")
    print("   • Check that the column names you entered exist in the dataset")  
    print("   • Ensure the variable column contains numeric data")
    print("   • Try using the sample URL: https://www2.census.gov/programs-surveys/popest/datasets/2020-2023/state/asrh/sc-est2023-alldata6.csv")

