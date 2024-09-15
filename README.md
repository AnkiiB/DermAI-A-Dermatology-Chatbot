# DermAI : A Dermatology Expert

DermAI is an advanced dermatology expert system powered by OpenAI's GPT-4 model. Leveraging Retrieval-Augmented Generation (RAG) techniques, DermAI answers questions about dermatology, including disease symptoms and potential diagnoses based on user inputs. The system utilizes a comprehensive PDF textbook as its knowledge base to provide accurate and informative responses.

Key Features:

Dermatology Q&A: Responds to queries about dermatological conditions and symptoms.
Image Analysis: Detects and classifies skin cancer from user-submitted images, identifying specific types of skin cancer.
Knowledge Base: Derived from a detailed PDF textbook on dermatology.
DermAI is designed to offer reliable dermatological insights and image-based skin cancer detection.

# Installation
1. Clone this repository to your local machine: <br>
    ``` git clone https://github.com/AnkiiB/DermAI ```

2. Navigate to the project directory: <br>
   ```cd 

3. Install the required dependencies using pip: <br>
   ```pip install -r requirements.txt```

4. Enter OpenAI API key in langchain.py: <br>
   ```openai_api_key = os.getenv('OPENAI_API_KEY')```

5. Run main.py: <br>
   ```python3 main.py```

6. Go to terminal and run: <br>
   ```streamlit run langchain.py```

   
