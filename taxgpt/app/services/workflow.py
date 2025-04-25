from typing import List
from typing_extensions import TypedDict
from langchain.schema import Document
from pydantic import BaseModel, Field
from langgraph.graph import END, START, StateGraph
from app.core.config import settings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools.tavily_search import TavilySearchResults
import requests

# Initialize components
llm = ChatGroq(groq_api_key=settings.GROQ_API_KEY, model_name="mixtral-8x7b-32768")
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
web_search_tool = TavilySearchResults(k=3)


# Model for relevance grading
class classify_query(BaseModel):
    """To classify the user question."""
    binary_score: str = Field(description="Question is classified as 'related', 'illegal', or 'notrelated'.")

structured_llm_classifier = llm.with_structured_output(classify_query)

query_classifier_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a query clssifier . 
                  Classify the given query into the following catagories.
                  tax related legal / tax related illegal / non tax related. 
                  Give response as 'related' , 'illegal' , 'notrelated'."""),
    ("human", "User question: {question}")
])

query_classifier=query_classifier_prompt | structured_llm_classifier

class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""
    binary_score: str = Field(description="Documents are relevant to the question, 'Yes' or 'No'")

structured_llm_grader = llm.with_structured_output(GradeDocuments)

# Prompts
grade_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a grader assessing relevance of a retrieved document to a user question. 
                  If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant. 
                  Give a binary score 'Yes' or 'No'."""),
    ("human", "Retrieved document: \n\n {document} \n\n User question: {question}")
])
retrieval_grader = grade_prompt | structured_llm_grader

re_write_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a question re-writer optimizing questions for web search. 
                  Improve the input for better semantic intent and give back only the question.
                  The question will be related to Incometax so rewrite in a way that you mention indian incometax in the question"""),
    ("human", "Here is the initial question: \n\n {question} \n Formulate an improved question.")
])
question_rewriter = re_write_prompt | llm | StrOutputParser()


rag_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a highly knowledgeable tax expert specializing in Indian tax laws and regulations.
                  Your task is to provide clear, step-by-step explanations to address tax-related questions. 
                  Ensure to include relevant examples, references to applicable tax sections, and any tips that could be helpful.
                  If you are unsure about an answer, be honest and state that you are unsure rather than guessing.
                  """),
    ("human", """Here is the context and question for you:
                  \n\n Context: {context}
                  \n\n Question: {question} 
                  \n Please provide a detailed and well-structured answer based on the given context. 
                  Ensure the answer aligns with Indian tax laws and provides actionable insights where possible."""),
])

rag_chain = rag_prompt | llm | StrOutputParser()

#Response for non tax related or illegal queries
out_of_scope_response_prompt=re_write_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a polite tax assistant. Respond appropriately to the user's question based on the following rules:
                    If the question is non-tax-related, say: "I can only assist with tax-related queries. Please try rephrasing your question."
                    If the question is illegal, say: "I cannot provide assistance with queries involving illegal activities."""),
    ("human", """Here is the  question for you.
                    \n\n Question: {question}
                    """)
])
out_of_scope_generation = out_of_scope_response_prompt | llm | StrOutputParser()

# Workflow Functions
def classify_user_query(state):
    type=query_classifier.invoke({"question":state["question"]})
    state["query_type"]=type.binary_score
    return {"query_type":type.binary_score}

def non_related_generation(state):
    return "retrieve" if state["query_type"] == "related" else "generate_response"

def retrieve(state):
    try:
        response = requests.post(
            "http://http://52.66.236.117:5001/vector",
            params={"question": state["question"]}
        )
        response.raise_for_status()
        
        # Get the documents from the API response
        response = response.json()

        doc=response.get("document")

        # Return in the same format as the initial method
        return {"documents": [Document(page_content=doc)], "question": state["question"]}
        
    except requests.exceptions.RequestException as e:
        return {"generation": "Error: Unable to fetch the response. Please try again later."}




def grade_documents(state):
    filtered_docs = []
    for doc in state["documents"]:
        score = retrieval_grader.invoke({"question": state["question"], "document": doc.page_content})
        if score.binary_score == "Yes":
            filtered_docs.append(doc)
    return {"documents": filtered_docs, "question": state["question"], "web_search": "Yes" if not filtered_docs else "No"}

def transform_query(state):
    better_question = question_rewriter.invoke({"question": state["question"]})
    return {"documents": state["documents"], "question": better_question}

def web_search(state):
    docs = web_search_tool.invoke({"query": state["question"]})
    web_results = "\n".join([d["content"] for d in docs])
    return {"documents": [Document(page_content=web_results)], "question": state["question"]}

def generate_response(state):
    if state["query_type"]=="related":
        generation = rag_chain.invoke({"context": state["documents"], "question": state["question"]})
        return {"documents": state["documents"], "question": state["question"], "generation": generation}
    else:
        generation=out_of_scope_generation.invoke({"question": state["question"]})
        return {"documents": state["documents"], "question": state["question"], "generation": generation}



def decide_to_generate(state):
    return "transform_query" if state["web_search"] == "Yes" else "generate_response"



# Define StateGraph workflow logic
class State(TypedDict):
    question: str
    generation: str
    web_search: str
    query_type: str
    documents: List[Document]


workflow = StateGraph(State)
workflow.add_node("classify_user_query",classify_user_query)
workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("transform_query", transform_query)
workflow.add_node("web_search_node", web_search)
workflow.add_node("generate_response", generate_response)

workflow.add_edge(START, "classify_user_query")
workflow.add_conditional_edges("classify_user_query",non_related_generation,{
    "retrieve":"retrieve",
    "generate_response": "generate_response"
})
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges("grade_documents", decide_to_generate, {
    "transform_query": "transform_query",
    "generate_response": "generate_response"
})
workflow.add_edge("transform_query", "web_search_node")
workflow.add_edge("web_search_node", "generate_response")
workflow.add_edge("generate_response", END)

tax_app = workflow.compile()
