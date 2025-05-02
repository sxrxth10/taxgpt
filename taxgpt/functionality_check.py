import unittest
from unittest.mock import patch, MagicMock
from langchain.schema import Document
import logging


# Import your workflow functions from the actual module
from app.services.workflow import (
    classify_user_query,
    non_related_generation,
    retrieve,
    grade_documents,
    transform_query,
    web_search,
    generate_response,
    decide_to_generate
)


class TestClassifyUserQuery(unittest.TestCase):

    @patch("app.services.workflow.query_classifier")
    def test_query_related_to_tax(self, mock_classifier):
        mock_response = MagicMock()
        mock_response.binary_score = "related"
        mock_classifier.invoke.return_value = mock_response

        state = {"question": "How can I file income tax in India?"}
        result = classify_user_query(state)

        self.assertEqual(result["query_type"], "related")
        self.assertEqual(state["query_type"], "related")

    @patch("app.services.workflow.query_classifier")
    def test_query_not_related_to_tax(self, mock_classifier):
        mock_response = MagicMock()
        mock_response.binary_score = "notrelated"
        mock_classifier.invoke.return_value = mock_response

        state = {"question": "Who won the IPL last year?"}
        result = classify_user_query(state)

        self.assertEqual(result["query_type"], "notrelated")
        self.assertEqual(state["query_type"], "notrelated")

    @patch("app.services.workflow.query_classifier")
    def test_query_classifier_exception(self, mock_classifier):
        mock_classifier.invoke.side_effect = Exception("LLM failed")

        state = {"question": "DROP TABLE users;"}
        result = classify_user_query(state)

        self.assertEqual(result["query_type"], "notrelated")
        self.assertEqual(state["query_type"], "notrelated")


class TestNonRelatedGeneration(unittest.TestCase):

    def test_related_query_type(self):
        state = {"query_type": "related"}
        self.assertEqual(non_related_generation(state), "retrieve")

    def test_notrelated_query_type(self):
        state = {"query_type": "notrelated"}
        self.assertEqual(non_related_generation(state), "generate_response")

    def test_illegal_query_type(self):
        state = {"query_type": "illegal"}
        self.assertEqual(non_related_generation(state), "generate_response")

    def test_missing_query_type(self):
        state = {}
        self.assertEqual(non_related_generation(state), "generate_response")


class TestRetrieve(unittest.TestCase):

    @patch("app.services.workflow.requests.post")
    def test_successful_retrieve(self, mock_post):
        mock_response = MagicMock()
        mock_response.json.return_value = {"document": "Test content"}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        state = {"question": "What is income tax?"}
        result = retrieve(state)

        self.assertIn("documents", result)
        self.assertEqual(result["documents"][0].page_content, "Test content")
        self.assertEqual(result["question"], state["question"])

    @patch("app.services.workflow.requests.post")
    def test_failed_request(self, mock_post):
        mock_post.side_effect = Exception("API down")

        state = {"question": "What is GST?"}
        result = retrieve(state)

        self.assertIn("generation", result)
        self.assertTrue(result["generation"].startswith("Error:"))


class TestGradeDocuments(unittest.TestCase):

    @patch("app.services.workflow.retrieval_grader")
    def test_grades_documents_relevant(self, mock_grader):
        mock_grader.invoke.return_value = MagicMock(binary_score="Yes")
        state = {
            "question": "What is income tax?",
            "documents": [Document(page_content="Income tax is a tax levied...")]
        }

        result = grade_documents(state)
        self.assertEqual(len(result["documents"]), 1)
        self.assertEqual(result["web_search"], "No")

    @patch("app.services.workflow.retrieval_grader")
    def test_grades_documents_not_relevant(self, mock_grader):
        mock_grader.invoke.return_value = MagicMock(binary_score="No")
        state = {
            "question": "What is income tax?",
            "documents": [Document(page_content="Unrelated content")]
        }

        result = grade_documents(state)
        self.assertEqual(len(result["documents"]), 0)
        self.assertEqual(result["web_search"], "Yes")

    @patch("app.services.workflow.retrieval_grader")
    def test_grades_documents_exception(self, mock_grader):
        mock_grader.invoke.side_effect = Exception("Grader error")
        state = {
            "question": "What is income tax?",
            "documents": [Document(page_content="Income tax explanation")]
        }

        result = grade_documents(state)
        self.assertEqual(len(result["documents"]), 0)
        self.assertEqual(result["web_search"], "Yes")


class TestTransformQuery(unittest.TestCase):

    @patch("app.services.workflow.question_rewriter")
    def test_successful_rewrite(self, mock_rewriter):
        mock_rewriter.invoke.return_value = "What is the tax rate for freelancers in India?"
        state = {
            "question": "freelancer tax India",
            "documents": [Document(page_content="Some tax info")]
        }

        result = transform_query(state)
        self.assertEqual(result["question"], "What is the tax rate for freelancers in India?")
        self.assertEqual(result["documents"], state["documents"])

    @patch("app.services.workflow.question_rewriter")
    def test_rewrite_exception(self, mock_rewriter):
        mock_rewriter.invoke.side_effect = Exception("LLM error")
        state = {
            "question": "freelancer tax India",
            "documents": [Document(page_content="Some tax info")]
        }

        result = transform_query(state)
        self.assertEqual(result["question"], "freelancer tax India")
        self.assertEqual(result["documents"], state["documents"])


class TestWebSearch(unittest.TestCase):

    @patch("app.services.workflow.web_search_tool")
    def test_successful_web_search(self, mock_search_tool):
        # Mocking the web search tool's response
        mock_search_tool.invoke.return_value = [
            {"content": "Document 1 content."},
            {"content": "Document 2 content."}
        ]
        state = {
            "question": "taxation on freelancers",
            "documents": [Document(page_content="Some tax info")]
        }

        result = web_search(state)
        
        self.assertEqual(len(result["documents"]), 1)
        self.assertIn("Document 1 content.", result["documents"][0].page_content)
        self.assertIn("Document 2 content.", result["documents"][0].page_content)
        self.assertEqual(result["question"], "taxation on freelancers")

    @patch("app.services.workflow.web_search_tool")
    def test_web_search_exception(self, mock_search_tool):
        # Simulating an exception during the web search
        mock_search_tool.invoke.side_effect = Exception("Web search failed")
        state = {
            "question": "taxation on freelancers",
            "documents": [Document(page_content="Some tax info")]
        }

        result = web_search(state)
        
        # Verifying the returned documents are empty
        self.assertEqual(len(result["documents"]), 0)
        self.assertEqual(result["question"], "taxation on freelancers")


class TestGenerateResponse(unittest.TestCase):

    @patch("app.services.workflow.rag_chain")
    def test_generate_response_related(self, mock_rag_chain):
        # Mocking the RAG chain response
        mock_rag_chain.invoke.return_value = "Generated response for related query."
        
        state = {
            "query_type": "related",
            "question": "What is the tax on income?",
            "documents": [Document(page_content="Document content related to income tax.")]
        }

        result = generate_response(state)
        
        self.assertEqual(result["generation"], "Generated response for related query.")
        self.assertEqual(result["question"], "What is the tax on income?")
        self.assertIn("Document content related to income tax.", result["documents"][0].page_content)

    @patch("app.services.workflow.out_of_scope_generation")
    def test_generate_response_out_of_scope(self, mock_out_of_scope_gen):
        # Mocking out-of-scope generation response
        mock_out_of_scope_gen.invoke.return_value = "Out of scope response generated."
        
        state = {
            "query_type": "notrelated",
            "question": "What is the weather like today?",
            "documents": [Document(page_content="Document content unrelated to tax.")]
        }

        result = generate_response(state)
        
        # Check if the mocked return value was used in the response
        self.assertEqual(result["generation"], "Out of scope response generated.")
        self.assertEqual(result["question"], "What is the weather like today?")
        self.assertIn("Document content unrelated to tax.", result["documents"][0].page_content)

class TestDecideToGenerate(unittest.TestCase):

    def test_decision_web_search_needed(self):
        state = {"web_search": "Yes"}
        result = decide_to_generate(state)
        self.assertEqual(result, "transform_query")

    def test_decision_generate_directly(self):
        state = {"web_search": "No"}
        result = decide_to_generate(state)
        self.assertEqual(result, "generate_response")

    def test_decision_missing_key(self):
        # Test fallback when 'web_search' is missing
        state = {}
        result = decide_to_generate(state)
        self.assertEqual(result, "generate_response")

    def test_decision_key_none(self):
        # Test when the key is present but value is None
        state = {"web_search": None}
        result = decide_to_generate(state)
        self.assertEqual(result, "generate_response")

from app.services.workflow import tax_app

class TestTaxAppWorkflow(unittest.TestCase):
    @patch("app.services.workflow.requests.post")
    @patch("app.services.workflow.web_search_tool")
    @patch("app.services.workflow.query_classifier")
    @patch("app.services.workflow.retrieval_grader")
    @patch("app.services.workflow.rag_chain")
    def test_full_workflow_related_query(
        self,
        mock_rag_chain,
        mock_retrieval_grader,
        mock_query_classifier,
        mock_web_search_tool,
        mock_post
    ):
        # Mock classifier
        mock_query_classifier.invoke.return_value = MagicMock(binary_score="related")
        # Mock vector search
        mock_post.return_value.json.return_value = {"document": "Tax info"}
        mock_post.return_value.raise_for_status.return_value = None
        # Mock document grader
        mock_retrieval_grader.invoke.return_value = MagicMock(binary_score="Yes")
        # Mock RAG chain
        mock_rag_chain.invoke.return_value = "Tax response"
        # Mock web search (not needed for relevant documents)
        mock_web_search_tool.invoke.return_value = []

        state = {"question": "What is income tax in India?"}
        result = tax_app.invoke(state)

        self.assertIn("generation", result)
        self.assertEqual(result["generation"], "Tax response")
        self.assertEqual(result["query_type"], "related")
        self.assertEqual(len(result["documents"]), 1)
        self.assertEqual(result["documents"][0].page_content, "Tax info")

if __name__ == "__main__":
    logging.disable(logging.CRITICAL)  # to suppress logging output in test results
    unittest.main()
