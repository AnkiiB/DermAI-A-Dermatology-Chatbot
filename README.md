# DermAI : A Dermatology Expert

DermAI is an advanced dermatology expert system powered by OpenAI's GPT-4 model. Leveraging Retrieval-Augmented Generation (RAG) techniques, DermAI answers questions about dermatology, including disease symptoms and potential diagnoses based on user inputs. The system utilizes a comprehensive PDF textbook as its knowledge base to provide accurate and informative responses.

## Key Features:

**Dermatology Q&A:** Responds to queries about dermatological conditions and symptoms.

![WhatsApp Image 2024-09-15 at 8 30 02 PM](https://github.com/user-attachments/assets/927b3c66-919d-4ba1-9d83-b61e781ec1c2)

**Image Analysis:** Detects and classifies skin cancer from user-submitted images, identifying specific types of skin cancer.

![WhatsApp Image 2024-09-15 at 8 30 12 PM](https://github.com/user-attachments/assets/ea0e08c9-6b5c-4bfa-95c9-ecc55fdd3fd1)

**Knowledge Base:** Derived from a detailed PDF textbook on dermatology.

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

   
