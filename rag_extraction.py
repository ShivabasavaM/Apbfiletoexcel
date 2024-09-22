import os
import openai
import pandas as pd
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor
import PyPDF2


load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')

def query_llm(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=500
    )
    return response['choices'][0]['message']['content'].strip()

def extract_features(document):
    prompt = f"Extract all key features from the following technical document:\n\n{document}"
    return query_llm(prompt)


def get_validation_steps(feature):
    prompt = f"For the following feature: '{feature}', list the steps required to validate and test its correctness."
    return query_llm(prompt)

def process_feature(feature):
    validation_steps = get_validation_steps(feature)
    return (feature, validation_steps)


def read_pdf(file_path):
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ''
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            text += page.extract_text()
    return text


def process_document(document):
    print("Processing the document...")
    

    print("Extracting features from the document...")
    features = extract_features(document).split("\n")

    results = []
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_feature, feature) for feature in features]
        for future in futures:
            results.append(future.result())
    
    df = pd.DataFrame(results, columns=["Feature", "Validation Steps"])
    df.to_csv("output1.csv", index=False)
    print("Results exported to output.csv")

pdf_path = "apb.pdf"
apb_document = read_pdf(pdf_path)

process_document(apb_document)
