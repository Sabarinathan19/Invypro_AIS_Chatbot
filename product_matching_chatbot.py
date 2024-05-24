import streamlit as st
import pymongo
from langchain.llms import GPT4All
from langchain.chains import LLMChain
from langchain.prompts.prompt import PromptTemplate
import re
from sentence_transformers import SentenceTransformer

# MongoDB connection details
client = pymongo.MongoClient("mongodb+srv://dbadmin:G2CsbXDJaaaOMtP0K@invypro-universaldata.yqeb5ql.mongodb.net/")
db = client.invypro_ais
collection = db.universal_product_master

# Check if the 'productName' index already exists
index_name = "product_index"
indexes = collection.index_information()
if "productName_text" not in indexes:
    collection.create_index([("productName", pymongo.TEXT)], name=index_name)

# Initialize the GPT4All model
model_path = r"C:\Users\Sabarinathan T\AppData\Local\nomic.ai\GPT4All\orca_mini_v3_7b.Q4_0.gguf"
gpt4all_model = GPT4All(model=model_path, n_threads=4)  # Adjust threads as needed

# Define the prompt template
prompt_template = PromptTemplate(
    input_variables=["query"],
    template="""You are an AI language model assistant. Your task is to retrieve relevant product details
    from a product database based on the user's query.
    User query: {query}\n
    Please extract the key product names from the query. Provide the product names only, separated by commas.
    """
)

# Create the LLM chain with the new prompt template
chain = LLMChain(llm=gpt4all_model, prompt=prompt_template)

# Initialize the embedding model
embedding_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Function to generate embeddings for the given product names
def generate_embedding(text: str) -> list:
    embeddings = embedding_model.encode(text, convert_to_tensor=False, convert_to_numpy=True)
    return embeddings.tolist()

# Function to query MongoDB using vector search
def query_database(product_names):
    results = []
    for product_name in product_names:
        query_embedding = generate_embedding(product_name)
        results.extend(find_matching_products(query_embedding))
    return results

def find_matching_products(query_embedding):
    results = collection.aggregate([
        {
            "$vectorSearch": {
                "queryVector": query_embedding,
                "path": "name_embedding",
                "numCandidates": 100,
                "limit": 3,
                "index": "vector_index",
            }
        },
        {
            "$project": {
                "_id": 0,
                "productName": 1,
                "hsnCode": 1,
                "score": {"$meta": "vectorSearchScore"},
            }
        }
    ])

    assumed_products_lists = []

    for result in results:
        assumed_products_list = {
            "product_name": result["productName"],
            "hsn_code": result["hsnCode"],
            "confidence_score": result["score"]
        }
        assumed_products_lists.append(assumed_products_list)

    return assumed_products_lists

# Function to handle the process from user query to database results
def get_response(query):
    # Generate response from the LLM
    input_dict = {"query": query}
    response = chain.invoke(input=input_dict)
    
    # Access the relevant part of the response dictionary
    response_text = response['text'].strip()
    
    # Extract the part after the colon
    if ':' in response_text:
        response_text = response_text.split(':', 1)[1].strip()
    
    # Extract product names from the response text using a more robust method
    product_names = re.split(r',\s*', response_text)  # Splitting by comma and stripping whitespace
    
    # Query MongoDB to find the matching products
    results = query_database(product_names)
    
    return results

# Main function to run Streamlit app
def main():
    st.title("Product Search with LLM")
    
    # User input for the product query
    user_query = st.text_area("Enter your product query:")
    
    # Search button
    if st.button("Search"):
        if user_query:
            # Process user's query using the LLM
            results = get_response(user_query)
            
            # Display the results
            if results:
                st.write("### Matching Products:")
                for result in results:
                    st.write(f"**Product Name**: {result['product_name']}")
                    st.write(f"**HSN Code**: {result['hsn_code']}")
                    st.write(f"**Confidence Score**: {result['confidence_score']}")
                    st.write("---")
            else:
                st.write("No matching products found.")

# Run the Streamlit app
if __name__ == "__main__":
    main()
































# import streamlit as st
# import pymongo
# from langchain.llms import GPT4All
# from langchain.chains import LLMChain
# from langchain.prompts.prompt import PromptTemplate
# import re
# # MongoDB connection details
# client = pymongo.MongoClient("mongodb+srv://dbadmin:G2CsbXDJaaaOMtP0K@invypro-universaldata.yqeb5ql.mongodb.net/")
# db = client.invypro_ais
# collection = db.universal_product_master

# # Initialize the GPT4All model
# model_path = r"C:\Users\Sabarinathan T\AppData\Local\nomic.ai\GPT4All\orca_mini_v3_7b.Q4_0.gguf"
# gpt4all_model = GPT4All(model=model_path, n_threads=8)

# # Define the prompt template
# prompt_template = PromptTemplate(
#     input_variables=["query"],
#     template="""You are an AI language model assistant. Your task is to retrieve relevant product details
#     from a product database based on the user's query.
#     User query: {query}\n
#     Please extract the key product names from the query. Provide the product names only, separated by commas.
#     """
# )

# # Create the LLM chain with the new prompt template
# chain = LLMChain(llm=gpt4all_model, prompt=prompt_template)

# from sentence_transformers import SentenceTransformer

# # Initialize the embedding model
# model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# # Function to generate embeddings for the given product names
# def generate_embedding(text: str) -> list[float]:
#     embeddings = model.encode(text, convert_to_tensor=False, convert_to_numpy=True)
#     return embeddings.tolist()

# # Function to query MongoDB using vector search
# def query_database(product_names):
#     results = []
#     for product_name in product_names:
#         query_embedding = generate_embedding(product_name)
#         results.extend(find_matching_products(query_embedding))
#     return results

# def find_matching_products(query_embedding):
#     results = collection.aggregate([
#         {
#             "$vectorSearch": {
#                 "queryVector": query_embedding,
#                 "path": "name_embedding",
#                 "numCandidates": 100,
#                 "limit": 3,
#                 "index": "vector_index",
#             }
#         },
#         {
#             "$project": {
#                 "_id": 0,
#                 "productName": 1,
#                 "hsnCode": 1,
#                 "score": {"$meta": "vectorSearchScore"},
#             }
#         }
#     ])

#     assumed_products_lists = []

#     for result in results:
#         assumed_products_list = {
#             "product_name": result["productName"],
#             "hsn_code": result["hsnCode"],
#             "confidence_score": result["score"]
#         }
#         assumed_products_lists.append(assumed_products_list)

#     return assumed_products_lists

# # Main function to run Streamlit app
# def main():
#     st.title("Product Search with LLM")
    
#     # User input for the product query
#     user_query = st.text_area("Enter your product query:")
    
#     # Search button
#     if st.button("Search"):
#         if user_query:
#             # Process user's query using the LLM
#             results = get_response(user_query)
            
#             # Display the results
#             if results:
#                 st.write("### Matching Products:")
#                 for result in results:
#                     st.write(f"**Product Name**: {result['product_name']}")
#                     st.write(f"**HSN Code**: {result['hsn_code']}")
#                     st.write(f"**Confidence Score**: {result['confidence_score']}")
#                     st.write("---")
#             else:
#                 st.write("No matching products found.")
                
# def get_response(query):
#     # Generate response from the LLM
#     input_dict = {"query": query}
#     response = chain.invoke(input=input_dict)
    
#     # Access the relevant part of the response dictionary
#     response_text = response['text'].strip()
    
#     # Extract the part after the colon
#     if ':' in response_text:
#         response_text = response_text.split(':', 1)[1].strip()
    
#     # Extract product names from the response text using a more robust method
#     product_names = re.split(r',\s*', response_text)  # Splitting by comma and stripping whitespace
    
#     print(product_names)  # Added print statement for debugging
    
#     # Query MongoDB to find the matching products
#     results = query_database(product_names)
    
#     return results

# # Run the Streamlit app
# if __name__ == "__main__":
#     main()








































# import streamlit as st
# import pymongo
# from langchain.llms import GPT4All
# from langchain.chains import LLMChain
# from langchain.prompts.prompt import PromptTemplate
# import re
# # MongoDB connection details
# client = pymongo.MongoClient("mongodb+srv://dbadmin:G2CsbXDJaaaOMtP0K@invypro-universaldata.yqeb5ql.mongodb.net/")
# db = client.invypro_ais
# collection = db.universal_product_master

# # Initialize the GPT4All model
# model_path = r"C:\Users\Sabarinathan T\AppData\Local\nomic.ai\GPT4All\orca_mini_v3_7b.Q4_0.gguf"
# gpt4all_model = GPT4All(model=model_path, n_threads=8)

# # Define the prompt template
# prompt_template = PromptTemplate(
#     input_variables=["query"],
#     template="""You are an AI language model assistant. Your task is to retrieve relevant product details
#     from a product database based on the user's query.
#     User query: {query}\n
#     Please extract the key product names from the query. Provide the product names only, separated by commas.
#     """
# )

# # Create the LLM chain with the new prompt template
# chain = LLMChain(llm=gpt4all_model, prompt=prompt_template)

# # Function to query MongoDB based on the LLM's generated response
# def query_database(product_names):
#     results = []
#     for product_name in product_names:
#         query = {"productName": {"$regex": product_name, "$options": "i"}}
#         projection = {"_id": 0, "productName": 1, "hsnCode": 1}
#         product_results = list(collection.find(query, projection).limit(3))
#         results.extend(product_results)
#     return results

# # Main function to run Streamlit app
# def main():
#     st.title("Product Search with LLM")
    
#     # User input for the product query
#     user_query = st.text_area("Enter your product query:")
    
#     # Search button
#     if st.button("Search"):
#         if user_query:
#             # Process user's query using the LLM
#             results = get_response(user_query)
            
#             # Display the results
#             if results:
#                 st.write("### Matching Products:")
#                 for result in results:
#                     st.write(f"**Product Name**: {result['productName']}")
#                     st.write(f"**HSN Code**: {result['hsnCode']}")
#                     # st.write(f"**Confidence Score**: {result['confidence_score']}")
#                     st.write("---")
#             else:
#                 st.write("No matching products found.")
                
# def get_response(query):
#     # Generate response from the LLM
#     input_dict = {"query": query}
#     response = chain.invoke(input=input_dict)
    
#     # Access the relevant part of the response dictionary
#     response_text = response['text'].strip()
    
#     # Extract the part after the colon
#     if ':' in response_text:
#         response_text = response_text.split(':', 1)[1].strip()
    
#     # Extract product names from the response text using a more robust method
#     product_names = re.split(r',\s*', response_text)  # Splitting by comma and stripping whitespace
    
#     # Query MongoDB to find the matching products
#     results = query_database(product_names)
    
#     return results

# # Run the Streamlit app
# if __name__ == "__main__":
#     main()













# import pymongo
# from langchain.llms import GPT4All
# from langchain.chains import LLMChain
# from langchain.prompts.prompt import PromptTemplate
# import json

# # MongoDB connection details
# client = pymongo.MongoClient("mongodb+srv://dbadmin:G2CsbXDJaaaOMtP0K@invypro-universaldata.yqeb5ql.mongodb.net/")
# db = client.invypro_ais
# collection = db.universal_product_master

# # Initialize the GPT4All model
# model_path = r"C:\Users\Sabarinathan T\AppData\Local\nomic.ai\GPT4All\orca_mini_v3_7b.Q4_0.gguf"
# gpt4all_model = GPT4All(model=model_path, n_threads=8)


# # Define the new prompt template
# prompt_template = PromptTemplate(
#     input_variables=["question"],
#     template="""You are an AI language model assistant. Your task is to retrieve relevant product names
#     from a product database based on the user's query.
#     User query: {question}\n
#     Please provide the product names only, separated by commas.
#     """
# )

# # Create the LLM chain with the new prompt template
# chain = LLMChain(llm=gpt4all_model, prompt=prompt_template)

# # Function to query MongoDB based on the LLM's generated response
# def query_database(product_names):
#     product_names_list = [name.strip() for name in product_names.split(",")]
#     results = []
#     for product_name in product_names_list:
#         query = {"productName": {"$regex": product_name, "$options": "i"}}
#         projection = {"_id": 0, "productName": 1, "hsnCode": 1}
#         product_results = list(collection.find(query, projection).limit(3))
#         results.extend(product_results)
#     return results

# # Function to handle the process from user query to database results
# def get_response(question):
#     # Generate query from the LLM
#     input_dict = {"question": question}
#     response = chain.invoke(input=input_dict)
    
#     # Access the relevant part of the response dictionary
#     response_text = response['text'].strip().split(":")[1]
    
#     # Query MongoDB to find the matching products
#     results = query_database(response_text)
    
#     return results

# # Example input query
# input_query = input("Enter a product name: ")

# # Run the function with the input query and get the response
# response = get_response(question=input_query)

# # Print the response to ensure it's correct
# print("The matching products are as follows: ", json.dumps(response, indent=2))