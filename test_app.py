import unittest
import spacy
import openai
from transformers import T5Tokenizer, T5ForConditionalGeneration, BertTokenizer, BertModel
from app import (
    process_text,
    generate_ngrams,
    extract_terms_with_t5,
    extract_terms_with_openai,
    embed_text,
    analyze_compliance,
    split_sections,
)

class TestAppFunctions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.nlp = spacy.load("en_core_web_sm")
        cls.tokenizer_t5 = T5Tokenizer.from_pretrained("t5-small")
        cls.model_t5 = T5ForConditionalGeneration.from_pretrained("t5-small")
        cls.tokenizer_bert = BertTokenizer.from_pretrained("bert-base-uncased")
        cls.model_bert = BertModel.from_pretrained("bert-base-uncased")
        openai.api_key = "YOUR_OPENAI_API_KEY"

    def test_process_text(self):
        text = "The quick brown fox jumps over the lazy dog."
        expected_output = ['quick', 'brown', 'fox', 'jump', 'lazy', 'dog']
        self.assertEqual(process_text(text), expected_output)

    def test_generate_bigrams(self):
        terms = ["quick", "brown", "fox"]
        expected_output = ["quick brown", "brown fox"]
        self.assertEqual(generate_ngrams(terms, 2), expected_output)

    def test_generate_trigrams(self):
        terms = ["quick", "brown", "fox"]
        expected_output = ["quick brown fox"]
        self.assertEqual(generate_ngrams(terms, 3), expected_output)

    def test_extract_terms_with_t5(self):
        contract_text = "The total fee for the services is USD 100,000."
        terms = extract_terms_with_t5(contract_text)
        self.assertIsInstance(terms, str)
        self.assertGreater(len(terms), 0)

    def test_extract_terms_with_openai(self):
        contract_text = "The total fee for the services is USD 100,000."
        terms = extract_terms_with_openai(contract_text)
        self.assertIsInstance(terms, str)
        self.assertGreater(len(terms), 0)

    def test_embed_text(self):
        text = "The quick brown fox jumps over the lazy dog."
        embedding = embed_text(text, self.model_bert, self.tokenizer_bert)
        self.assertEqual(embedding.shape, (768,))

    def test_analyze_compliance(self):
        tasks = ["The payment is due upon receipt."]
        anchor_points = [embed_text("payment term", self.model_bert, self.tokenizer_bert)]
        compliance_report = analyze_compliance(tasks, anchor_points)
        self.assertIsInstance(compliance_report, list)
        self.assertEqual(len(compliance_report), 1)
        self.assertIsInstance(compliance_report[0], tuple)

    def test_split_sections(self):
        contract_text = """
        3. Financial Terms
        3.1 The total fee for the services under this Agreement is USD 100,000.
        3.2 Payment is made by the Client in stages: 50% upfront before the commencement of work, and 50% upon completion of the work.
        """
        sections = split_sections(contract_text)
        self.assertIsInstance(sections, dict)
        self.assertGreater(len(sections), 0)
        self.assertIn('Section 1', sections)

if __name__ == '__main__':
    unittest.main()
