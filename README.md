# TikTok TechJam 2025  
**Review Reviewer**  
Product of Team Null for TikTok TechJam 2025  

## Project Overview  
### Problem  
Low-quality comments such as advertisements and irrelevant feedback often clutter platforms like Google Maps, reducing user experience and making it harder for businesses to receive genuine reviews.  

### Solution  
Our product is an AI-based comment classifier that analyzes comments based on text, images, ranking, and information about the place. It automatically flags low-quality and irrelevant comments (text and images) for moderation to improve the overall quality of reviews.  

### Impact  
- **Users:** Enhance trust in location-based reviews by flagging low-quality and irrelevant comments.  
- **Business Owners:** Ensure fair representation by highlighting irrelevant or malicious reviews.  
- **Platforms:** Automate moderation by flagging suspicious comments, reducing manual workload.  

## Setup Instructions  
1. **Clone the repository:**  
   ```bash  
   git clone https://github.com/Noob-No-1/TikTok-TechJam-2025.git  
   cd TikTok-TechJam-2025  
   ```  
2. **Install dependencies:**  
   Ensure you have Python installed, then run:  
   ```bash  
   pip install -r requirements.txt  
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
- **Groq API:** Provides access to free hosted large language models (Llama 3.1 8B) for AI-based text classification.  
- **JSON:** Used for structured data exchange between the AI model and application.  

### Libraries  
- **dotenv:** Securely loads environment variables from `.env` files.  
- **groq:** Official Python client for interacting with the Groq API.  
- **json:** Standard Python library for parsing and generating JSON data.  
- **transformers:** Hugging Face library for leveraging multimodal models (e.g., BLIP for image captioning).  
- **torch:** Backend for running deep learning models.  
- **pandas:** For dataset manipulation and batch processing.  

## Multimodal Support  
Reviews that include images are handled by generating image captions using the BLIP model from the transformers library. These captions are appended to the review text to provide richer context for classification, enabling the AI to assess both textual and visual content effectively.  

## How to Reproduce Results  
1. **Run a sample classification:**  
   Execute the sample script or notebook to classify a single comment and observe the output.  
   *Note: You can also open and run the `demo.ipynb` notebook under `src` to see interactive demonstrations of text-only and image-augmented review classification.*  
2. **Run batch mode on dataset:**  
   Use the batch processing script to classify multiple comments from a dataset and save the results.  
3. **Expected outputs:**  
   The classifier will output structured JSON labels indicating review relevance and any policy violations (e.g., advertisement, irrelevant, rant_no_visit, image_irrelevant, image_advertisement).
