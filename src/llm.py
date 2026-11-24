import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

def get_llm():
    api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        raise ValueError("Gemini API key not found. Check your .env file.")

    llm = ChatGoogleGenerativeAI(
        model="gemini-3.5-flash-lite",
        temperature=0,
        google_api_key=api_key
    )

    return llm


if __name__ == "__main__":
    llm = get_llm()

    response = llm.invoke(
        "Explain Retrieval Augmented Generation in two sentences."
    )

    print("\nGemini Response:")
    print(response.content)