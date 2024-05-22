import os
import re
from pathlib import Path
from configparser import ConfigParser
from langchain_community.vectorstores import Neo4jVector
from langchain_community.graphs import Neo4jGraph
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import Tuple, List, Optional
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_community.vectorstores.neo4j_vector import remove_lucene_chars
import ast
from langchain_core.runnables import (
    RunnableBranch,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser

# Load configurations
def load_config():
    config = ConfigParser()
    config_path = Path(__file__).parent / "config.ini"
    config.read(config_path)
    return config

config = load_config()
# Initialize graph and LLM
os.environ["NEO4J_URI"] = config["Neo4j"]["uri"]
os.environ["NEO4J_USERNAME"] = config["Neo4j"]["username"]
os.environ["NEO4J_PASSWORD"] = config["Neo4j"]["password"]
graph = Neo4jGraph()

if config["LLM"]["llm"] == "OpenAI":
    from langchain_openai import ChatOpenAI
    openai_params = {
        "temperature": config["LLM"]["temperature"],
        "max_tokens": config["LLM"]["max_tokens"],
        "openai_api_key": config["OpenAI"]["api_key"],
        "openai_api_base": config["OpenAI"]["api_base"] if "api_base" in config["OpenAI"] else None,
        "stop": config["LLM"]["stop"],
    }
    llm = ChatOpenAI(**openai_params)
elif config["LLM"]["llm"] == "Ollama":
    from langchain_community.chat_models import ChatOllama
    options = {
        "temperature":config["LLM"]["temperature"],
        "num_ctx": config["Ollama"]["num_ctx"],
        "stop": config["LLM"]["stop"],
    }
    llm = ChatOllama(model=config["Ollama"]["model"], options=options)
else:
    raise ValueError("Invalid LLM model configuration")

if config["Embeddings"]["model"] == '':
    embeddings = llm
elif config["Embeddings"]["embeddings"] == "Ollama":
    from langchain_community.embeddings import OllamaEmbeddings
    embeddings = OllamaEmbeddings(model=config["Embeddings"]["model"])
elif config["Embeddings"]["embeddings"] == "OpenAI":
    from langchain_openai import EmbeddingsOpenAI
    embeddings = EmbeddingsOpenAI(model=config["Embeddings"]["model"], openai_api_key=config["OpenAI"]["api_key"])

vector_index = Neo4jVector.from_existing_graph(
    embeddings,
    search_type="hybrid",
    node_label="Document",
    text_node_properties=["text"],
    embedding_node_property="embedding"
)

# Retriever

graph.query(
    "CREATE FULLTEXT INDEX entity IF NOT EXISTS FOR (e:__Entity__) ON EACH [e.id]")

# Extract entities from text
class Entities(BaseModel):
    """Identifying information about entities."""

    names: List[str] = Field(
        ...,
        description="All the noun entities that "
        "appear in the text",
    )

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are extracting noun entities from the text. Don't include any explanation or text.",
        ),
        (
            "human",
            "Please extract all the noun entities into a list from the following "
            "input: {question}",
        ),
    ]
)

entity_chain = prompt | llm

def generate_full_text_query(input: str) -> str:
    """
    Generate a full-text search query for a given input string.

    This function constructs a query string suitable for a full-text search.
    It processes the input string by splitting it into words and appending a
    similarity threshold (~2 changed characters) to each word, then combines
    them using the AND operator. Useful for mapping entities from user questions
    to database values, and allows for some misspelings.
    """
    # print(input)
    full_text_query = ""
    words = [el for el in remove_lucene_chars(input).split() if el]
    if len(words) > 1:
        for word in words[:-1]:
            full_text_query += f" {word}~2 AND"
        full_text_query += f" {words[-1]}~2"
    else:
        full_text_query = f"{words[0]}~2"
    return full_text_query.strip()

# Fulltext index query
def structured_retriever(question: str) -> str:
    """
    Collects the neighborhood of entities mentioned
    in the question
    """
    result = ""
    entities = entity_chain.invoke({"question": question})
    # Add missing quotes around list elements
    fixed_content = re.sub(r'\[([^\'"\]]+)\]', r"['\1']", entities.content)
    list_entities = ast.literal_eval(fixed_content)
    for entity in list_entities:
        response = graph.query(
            """CALL db.index.fulltext.queryNodes('entity', $query, {limit:2})
            YIELD node,score
            CALL {
              WITH node
              MATCH (node)-[r:!MENTIONS]->(neighbor)
              RETURN node.id + ' - ' + type(r) + ' -> ' + neighbor.id AS output
              UNION ALL
              WITH node
              MATCH (node)<-[r:!MENTIONS]-(neighbor)
              RETURN neighbor.id + ' - ' + type(r) + ' -> ' +  node.id AS output
            }
            RETURN output LIMIT 50
            """,
            {"query": generate_full_text_query(entity)},
        )
        result += "\n".join([el['output'] for el in response])
    return result

def retriever(question: str):
    # print(f"Search query: {question}")
    structured_data = structured_retriever(question)
    unstructured_data = [el.page_content for el in vector_index.similarity_search(question)]
    final_data = f"""Structured data:
{structured_data}
Unstructured data:
{"#Document ". join(unstructured_data)}
    """
    return final_data

# Condense a chat history and follow-up question into a standalone question
_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question,
in English.
Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""  # noqa: E501
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

def _format_chat_history(chat_history: List[Tuple[str, str]]) -> List:
    buffer = []
    for human, ai in chat_history:
        buffer.append(HumanMessage(content=human))
        buffer.append(AIMessage(content=ai))
    return buffer

_search_query = RunnableBranch(
    # If input includes chat_history, we condense it with the follow-up question
    (
        RunnableLambda(lambda x: bool(x.get("chat_history"))).with_config(
            run_name="HasChatHistoryCheck"
        ),  # Condense follow-up question and chat into a standalone_question
        RunnablePassthrough.assign(
            chat_history=lambda x: _format_chat_history(x["chat_history"])
        )
        | CONDENSE_QUESTION_PROMPT
        | llm
        | StrOutputParser(),
    ),
    # Else, we have no chat history, so just pass through the question
    RunnableLambda(lambda x : x["question"]),
)

if config["Chat"]["chatbot"] == "Ollama":
    from langchain_community.chat_models import ChatOllama
    chatbot = ChatOllama(model=config["Chat"]["model"], options={"temperature":config["Chat"]["temperature"]})
elif config["Chat"]["chatbot"] == "OpenAI":
    from langchain_openai import ChatOpenAI
    chatbot = ChatOpenAI(
        temperature=config["Chat"]["temperature"],
        max_tokens=config["Chat"]["max_tokens"],
        openai_api_key=config["OpenAI"]["api_key"],
        openai_api_base=config["OpenAI"]["api_base"] if "api_base" in config["OpenAI"] else None,
        stop=config["Chat"]["stop"],
    )
else:
    chatbot = llm

template = """Answer the question based only on the following context:
{context}

Question: {question}
Use natural language and be concise.
Answer:"""
prompt = ChatPromptTemplate.from_template(template)

chain = (
    RunnableParallel(
        {
            "context": _search_query | retriever,
            "question": RunnablePassthrough(),
        }
    )
    | prompt
    | chatbot
    | StrOutputParser()
)

if __name__ == "__main__":
    while True:
        human_input = input("\n>>>")
        if human_input == "exit":
            break
        for chunk in chain.stream({"question": human_input}):
            print(chunk, end="", flush=True)