# Lucy PDF Chat

This is a Retrieval-Augmented Generation (RAG) model that allows users to upload documents and ask questions. It uses the BM25 algorithm to find the best match for the user's query and utilizes a large language model (LLM), specifically GPT, to generate responses based on the retrieved documents.
# Features

Document Upload: Users can upload their documents to the model.

Question Answering: Users can ask questions related to the uploaded documents.

BM25 Best Match: The model uses BM25 to retrieve the most relevant document sections.

LLM Response Generation: The model generates coherent and contextually relevant responses using GPT.

## Demo

![image](https://firebasestorage.googleapis.com/v0/b/tomatoguard-2110e.appspot.com/o/LC.png?alt=media&token=823847e4-3cfe-4ced-bbba-f72fb4759fd6)
## Authors

- [Markovian99](https://github.com/Markovian99)
- [PrxncE-LixH](https://github.com/PrxncE-LixH)


## Installation

```
cd root directory
```
 - Install required packages
   ``` 
   pip install -r requirements.txt
   ```

- add API KEY for LLM model, and select the model from the dropdown on the homepage
  ```
  modify the file name of example.env to .env and add your API KEY 
  ```

## How to run

```
cd root directory

streamlit run .\app.py
```

- Upload a document, click to build vector database, then start asking questions. 
## Additional Information

- Model was built on Ubuntu 24.04.1 LTS. Some packages like Unstructured may require its Windows build to work.

- Modify the temperature slider on the homepage to adjust the creativity and variability of the generated responses
