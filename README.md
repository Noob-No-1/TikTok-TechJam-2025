# TikTok TechJam 2025  
**Review Reviewer**  
Product of Team Null for TikTok TechJam 2025  

## Project Overview  
### Problem  
Low-quality comments such as advertisements and irrelevant feedback often clutter platforms like Google Maps, reducing user experience and making it harder for businesses to receive genuine reviews.  

### Solution  
Our product is an AI-based comment classifier that analyzes comments based on text, images, ranking, and information about the place. It automatically filters out low-quality comments to improve the overall quality of reviews.  

### Impact  
- **Users:** Enhance user experience by removing low-quality and irrelevant comments.  
- **Business Owners:** Help businesses receive more legitimate reviews, enabling them to improve their services effectively.  
- **Platforms:** Reduce manual work required for comment review by enabling automation, saving time and resources.  

## Setup Instructions  
1. **Clone the repository:**  
   ```bash  
   git clone https://github.com/your-repo/TikTok-TechJam-2025.git  
   cd TikTok-TechJam-2025  
   ```  
2. **Install dependencies:**  
   Ensure you have Python installed, then run:  
   ```bash  
   pip install groq python-dotenv  
   ```  
3. **Create a `.env` file:**  
   Add your API keys and environment variables securely in a `.env` file in the project root.  
   To get your Groq API key, visit the [GROQ website](https://groq.com/).  
   For example:  
   ```env  
   GROQ_API_KEY=your_api_key_here  
   ```  
4. **Run the comment classifier:**  
   Use the provided scripts or notebooks to start classifying comments.  

### Tools and APIs  
- **Python:** Primary programming language for backend logic and AI integration.  
- **VSCode:** Main IDE used for development.  
- **Jupyter Notebook:** Used for prototyping and interactive testing.  
- **Groq API:** Provides access to large language models (Llama 3.1 8B) for AI-based text classification.  
- **JSON:** Used for structured data exchange between the AI model and application.  

### Libraries  
- **dotenv:** Securely loads environment variables from `.env` files.  
- **groq:** Official Python client for interacting with the Groq API.  
- **json:** Standard Python library for parsing and generating JSON data.  

## How to Reproduce Results  
1. **Run a sample classification:**  
   Execute the sample script or notebook to classify a single comment and observe the output.  
2. **Run batch mode on dataset:**  
   Use the batch processing script to classify multiple comments from a dataset and save the results.  
3. **Expected outputs:**  
   The classifier will output labels indicating comment quality, filtering out advertisements and irrelevant comments, improving review authenticity.
