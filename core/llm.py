import os

DEFAULT_MODELS = {
    'openai': 'gpt-4o-mini',
    'ollama': 'llama3.2',
}


def get_llm(**kwargs):
    """Return a LangChain chat model chosen by LLM_PROVIDER and LLM_MODEL.

    LLM_PROVIDER is 'openai' or 'ollama' (default 'ollama'). LLM_MODEL
    falls back to a sensible default for the chosen provider. Extra
    keyword arguments (e.g. temperature) are passed to the model class.
    """
    provider = os.getenv('LLM_PROVIDER', 'ollama').strip().lower()
    if provider not in DEFAULT_MODELS:
        raise ValueError(
            f"Unsupported LLM_PROVIDER {provider!r}; expected 'openai' or 'ollama'."
        )
    model = os.getenv('LLM_MODEL') or DEFAULT_MODELS[provider]

    if provider == 'openai':
        from langchain_openai import ChatOpenAI

        # ChatOpenAI reads OPENAI_API_KEY from the environment.
        return ChatOpenAI(model=model, **kwargs)

    from langchain_ollama import ChatOllama

    base_url = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
    return ChatOllama(model=model, base_url=base_url, **kwargs)
