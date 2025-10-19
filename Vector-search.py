from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

import pandas as pd

books = pd.read_csv(r'book-recommender\final_books.csv')

#tagged_description col to text file
books['tagged_description'].to_csv("tagged_descripton.txt",
                                   sep="\n",
                                   index=False,
                                   header=False)

#load txt document -> document object
raw_document = TextLoader('tagged_descripton.txt',encoding='utf-8').load()
#Splitting
text_splitter = CharacterTextSplitter(
    chunk_size=0,chunk_overlap=0,separator="\n"
)

document = text_splitter(raw_document)