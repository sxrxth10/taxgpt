from typing import List
from typing_extensions import TypedDict
from langchain.schema import Document
from pydantic import BaseModel
from pydantic import Field
from langgraph.graph import END
from langgraph.graph import START
from langgraph.graph import StateGraph
from app.core.config import settings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools.tavily_search import TavilySearchResults
import requests
from langchain_openai import ChatOpenAI  # Changed from ChatGroq
import logging

logging.basicConfig(level=logging.DEBUG)

# Initialize components
llm = ChatOpenAI(
    openai_api_key=settings.OPENAI_API_KEY1,  # Changed to OpenAI API key
    model_name="gpt-3.5-turbo"
)
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
web_search_tool = TavilySearchResults(k=3)

logging.info("Initialized LLM, embeddings, and web search tool.")


# Model for relevance grading
class classify_query(BaseModel):
    """To classify the user question."""
    binary_score: str = Field(description="Question is classified as"
                              " 'related', 'illegal', or 'notrelated'.")


structured_llm_classifier = llm.with_structured_output(classify_query)

logging.info("Classify query model created.")

# Query classifier prompt
query_classifier_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a query clssifier .
                  Classify the given query into the following catagories.
                  tax related legal / tax related illegal / non tax related.
                  Give response as 'related' , 'illegal' , 'notrelated'."""),
    ("human", "User question: {question}")
])

logging.info("Query classifier prompt created.")

# Query classifier(Query types:Related, Non related, Illegal)
query_classifier = query_classifier_prompt | structured_llm_classifier


class GradeDocuments(BaseModel):
    """Schema for grading the relevance of documents against a query."""

    binary_score: str = Field(description="Documents are relevant to the"
                              " question, 'Yes' or 'No'")


structured_llm_grader = llm.with_structured_output(GradeDocuments)

logging.info("Grade documents model created.")

# Prompt for grading the retrieved document
grade_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a grader assessing relevance of a retrieved document
                to a user question.
                If the document contains keyword(s) or semantic meaning
                related to the question, grade it as relevant.
                Give a binary score 'Yes' or 'No'."""),
    ("human", "Retrieved document: \n\n {document} \n\n "
        "User question: {question}")
])
retrieval_grader = grade_prompt | structured_llm_grader

# Prompt for query rewrite for web search
re_write_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a question re-writer optimizing questions for
            web search
        Improve the input for better semantic intent and give back
            only the question.
        The question will be related to Incometax so rewrite in a
            way that you mention indian incometax in the question"""),
    ("human", "Here is the initial question: \n\n {question} \n Formulate an "
        "improved question.")
])
question_rewriter = re_write_prompt | llm | StrOutputParser()

logging.info("Query rewrite prompt created.")

# Prompt for the LLM
rag_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a highly knowledgeable tax expert specializing
      in Indian tax laws and regulations.
                  Your task is to provide clear, step-by-step explanations
      to address tax-related questions.
                  Ensure to include relevant examples, references to
      applicable tax sections, and any tips that could be helpful.
                  If you are unsure about an answer, be honest and state
      that you are unsure rather than guessing.
                  """),
    ("human", """Here is the context and question for you:
                  \n\n Context: {context}
                  \n\n Question: {question}
                  \n Please provide a detailed and well-structured answer
      based on the given context.
                    if the context is not relevent or avoid using it.
                  Ensure the answer aligns with Indian tax laws and provides
      actionable insights where possible."""),
])

# Define the chain
rag_chain = rag_prompt | llm | StrOutputParser()

logging.info("RAG chain created.")

# Response for non tax related or illegal queries
out_of_scope_response_prompt = re_write_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a polite tax assistant. Respond appropriately to the
     user's question based on the following rules:
                If the question is non-tax-related, say: "I can only assist
      with tax-related queries. Please try rephrasing your question."
                If the question is illegal, say:"I cannot provide assistance
      with queries involving illegal activities."""),
    ("human", """Here is the  question for you.
                    \n\n Question: {question}
                    """)
])
out_of_scope_generation = out_of_scope_response_prompt | llm | StrOutputParser()

logging.info("Out of scope response chain created.")

# Workflow Functions


def classify_user_query(state):
    """Classify the user's query type using an LLM classifier.

    Args:
        state (dict): Current state containing user's question.

    Returns:
        dict: Updated state with query_type field.
    """
    logging.debug(f"Classifying query: {state['question']}")
    try:
        type = query_classifier.invoke({"question": state["question"]})
        state["query_type"] = type.binary_score
        logging.debug(f"Query classified as: {state['query_type']}")
        return {"query_type": type.binary_score}
    except Exception as e:
        logging.error(f"Failed to classify query: {e}")
        state["query_type"] = "notrelated" 
        return {"query_type": "notrelated"}

def non_related_generation(state):
    """Generates response for non related and illegal queries based on the
     query type classification.
    Returns:
        str: Next node name based on query type.
    """
    try:
        logging.debug(f"Generating response for query type: {state['query_type']}")
        return "retrieve" if state["query_type"] == "related" else "generate_response"
    except Exception as e:
        logging.error(f"Failed in non_related_generation: {e}")
        return "generate_response"


def retrieve(state):
    """Fetch relevant documents using vector similarity search.

    Returns:
        dict: Retrieved documents or error message.
    """
    logging.info(f"Retrieving documents for query: {state['question']}")
    try:
        response = requests.post(
            "http://3.109.157.165:5001/vector",

            params={"question": state["question"]}
        )
        response.raise_for_status()
        # Get the documents from the API response
        response = response.json()

        doc = response.get("document")
        logging.debug(f"Documents retrieved: {doc}")
        # Return in the same format as the initial method
        return {"documents": [Document(page_content=doc)],
                "question": state["question"]}
    except requests.exceptions.RequestException as e:
        logging.error(f"Error retrieving documents: {e}")
        return {"generation": "Error: Unable to fetch the response."
                " Please try again later."}
    
    except Exception as e:
        logging.error(f"An unexpected error occured: {e}")
        return {"generation": "Error: Unable to fetch the response."
                " Please try again later."}


def grade_documents(state):
    """Evaluate retrieved documents for relevance to the query.

    Returns:
        dict: State with filtered relevant documents and web search decision.
    """
    logging.debug("Grading retrieved documents.")
    filtered_docs = []
    try:
        for doc in state["documents"]:
            score = retrieval_grader.invoke({"question": state["question"],
                                            "document": doc.page_content})
            if score.binary_score == "Yes":
                filtered_docs.append(doc)
        logging.info(f"Filtered documents: {len(filtered_docs)}")
        return {
            "documents": filtered_docs,
            "question": state["question"],
            "web_search": "Yes" if not filtered_docs else "No"
        }
    except Exception as e:
        logging.error(f"Error grading documents: {e}")
        return {
            "documents": [],
            "question": state["question"],
            "web_search": "Yes"
        }


def transform_query(state):
    """Rewrite the query to improve semantic matching for web search.

    Returns:
        dict: Updated question with transformed version.
    """
    logging.debug(f"Rewriting query: {state['question']}")
    try:
        better_question = question_rewriter.invoke({"question":
                                                    state["question"]})
        return {"documents": state["documents"], "question": better_question}
    except Exception as e:
        logging.error(f"Error rewriting query: {e}")
        return {"documents": state["documents"], "question": state["question"]}


def web_search(state):
    """Perform web search using the rewritten query.

    Returns:
        dict: Retrieved web content as documents.
    """
    try:
        docs = web_search_tool.invoke({"query": state["question"]})
        web_results = "\n".join([d.get("content", "") for d in docs if
                                "content" in d])
        return {"documents": [Document(page_content=web_results)],
                "question": state["question"]}
    except Exception as e:
        logging.error(f"Web search failed: {e}")
        return {"documents": [], "question": state["question"]}


def generate_response(state):
    """Generate a final response based on query type and documents.

    Returns:
        dict: Final response with generation output.
    """
    logging.debug("Generating response")
    try:
        if state["query_type"] == "related":
            response = rag_chain.invoke({
                "context": state["documents"],
                "question": state["question"]
            })
            logging.debug(f"Generated response: {response}")
        else:
            response = out_of_scope_generation.invoke({"question": state["question"]})
            logging.debug(f"Out of scope response generated: {response}")
        return {
            "documents": state["documents"],
            "question": state["question"],
            "generation": response
        }
    except Exception as e:
        logging.error(f"Failed to generate response: {e}")
        return {
            "documents": state.get("documents", []),
            "question": state.get("question", ""),
            "generation": "Error: Unable to generate response."
            " Please try again later."
        }


def decide_to_generate(state):
    """Decide whether to use web search or generate response directly
      based on graded docs.

    Returns:
        str: Next node name based on availability of relevant documents.
    """
    try:
        logging.debug("Deciding whether to generate response"
                      " or search the web")
        return "transform_query" if state["web_search"] == "Yes" else "generate_response"
    except Exception as e:
        logging.error(f"Decision logic failed: {e}")
        return "generate_response"


# Define StateGraph workflow logic
class State(TypedDict):
    """Defines the structure and datatypes of the workflow state."""
    question: str
    generation: str
    web_search: str
    query_type: str
    documents: List[Document]


workflow = StateGraph(State)
workflow.add_node("classify_user_query", classify_user_query)
workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("transform_query", transform_query)
workflow.add_node("web_search_node", web_search)
workflow.add_node("generate_response", generate_response)

workflow.add_edge(START, "classify_user_query")
workflow.add_conditional_edges("classify_user_query", non_related_generation, {
    "retrieve": "retrieve",
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

logging.info("Workflow compiled and ready.")
