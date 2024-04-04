Notable requirements for running all code present in the IGNGameRecommendation repo.

1. Ensure all libraries in the requirements.txt file are installed using:
   pip install -r requirements.txt

2. Two api keys are needed to run the frontend.py file; a Cohere api key, groq api key. and optionally a pinecone api key. API keys can be acquired by making an account at with the services below:
   
   https://console.groq.com/keys
   
   https://dashboard.cohere.com/api-keys
   
   https://www.pinecone.io/

4. API keys will need to be set as registry variables using the following commands (on a Windows OS)
   
   setx GROQ_API_KEY your_groq_api_key
   
   setx COHERE_API_KEY your_cohere_api_key
   
   setx PINECONE_API_KEY your_pinecone_api_key
   

   To set the registry variables on a Linux/Unix OS
   
   export GROQ_API_KEY=your-api-key-here
   
   export COHERE_API_KEY=your-api-key-here
   
   export PINECONE_API_KEY=your-api-key-here

6. To run any of the scripts included in the repo, clone the repo to the IDE of choice, and run scripts within the repo. Executing scripts at the command line is not supported.
