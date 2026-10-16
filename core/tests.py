import os
from unittest import mock

from django.test import TestCase
from django.urls import reverse


class HomePageTests(TestCase):
    def test_home_page_returns_200(self):
        response = self.client.get(reverse('home'))
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'home.html')


class GetLLMTests(TestCase):
    def test_returns_ollama_model(self):
        from langchain_ollama import ChatOllama

        from core.llm import get_llm

        with mock.patch.dict(os.environ, {'LLM_PROVIDER': 'ollama', 'LLM_MODEL': 'llama3.2'}):
            llm = get_llm()
        self.assertIsInstance(llm, ChatOllama)
        self.assertEqual(llm.model, 'llama3.2')

    def test_returns_openai_model(self):
        from langchain_openai import ChatOpenAI

        from core.llm import get_llm

        env = {'LLM_PROVIDER': 'openai', 'LLM_MODEL': 'gpt-4o-mini', 'OPENAI_API_KEY': 'test-key'}
        with mock.patch.dict(os.environ, env):
            llm = get_llm()
        self.assertIsInstance(llm, ChatOpenAI)
        self.assertEqual(llm.model_name, 'gpt-4o-mini')

    def test_rejects_unknown_provider(self):
        from core.llm import get_llm

        with mock.patch.dict(os.environ, {'LLM_PROVIDER': 'nope'}):
            with self.assertRaises(ValueError):
                get_llm()
