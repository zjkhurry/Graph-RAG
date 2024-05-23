# Graph RAG

## Introduction

 RAG (Retrieval Augmented Generation) is a technology that uses graph databases to store relationships between data points, replacing traditional text embedding methods like Chuck and enhancing the performance of RAG. This project is base on [Langchain](https://github.com/langchain-ai/langchain/tree/master) and [Neo4j](https://github.com/neo4j/neo4j?tab=readme-ov-file).

## Features

- Create a RAG from PDFs
- Connect to Zotero to retrival PDFs
- Graph visulation through Neo4j Browser
- All data are stored locally, supporting llama.cpp and Ollama local LLM

## Installation

**Step1**, clone this project and install dependencies
```
git clone https://github.com/zjkhurry/Graph-RAG.git
cd Graph-RAG
pip install requitements
```
Copy and midofy [config.ini](./config.ini)
```
cp config.ini.bak config.ini
```

**Step2**, you need to install Neo4j. Download [Neo4j Desktop](https://neo4j.com/download/) or Mac can install with homebrew
```
brew install --cask neo4j
```
1. Launch Neo4j Desktop, create a New Project and add a new Graph DBMS.
![avatar](res/1.png) ![avatar](res/2.png) ![avatar](res/3.png)
2. Enter the password into the [config.ini](./config.ini) - Neo4j - password. Then click on the Graph DBMS created in the last step and install [APOC plugin](https://github.com/neo4j/apoc).
![avatar](res/4.png)
3. Start the Graph DBMS.
   
**Step3 (optional)** Set up [Ollama](https://ollama.com) or [llama.cpp](https://github.com/ggerganov/llama.cpp) to use local LLM.
- For Ollama, download the app [here](https://ollama.com), run Ollama and then
```
ollama pull llama3
ollama pull mxbai-embed-large
```
- For llama.cpp, follow the instruction [here](https://github.com/ggerganov/llama.cpp) to build and run llama.cpp server.

**Step4**, config the zotero. [Pyzotero](https://github.com/urschrei/pyzotero) is used to connect to Zotero library. You'll need the ID of the personal or group library you want to access:
- Your personal library ID is available [here](https://www.zotero.org/settings/keys), in the section Your userID for use in API calls (you may need to login). Enter you ID into [config.ini](./config.ini)
- For group libraries, the ID can be found by opening the group's page: https://www.zotero.org/groups/groupname, and hovering over the group settings link. The ID is the integer after /groups/
- You'll also need to get an API key [here](https://www.zotero.org/settings/keys/new), enter you API key into [config.ini](./config.ini)
- Are you accessing your own Zotero library? library_type is 'user'
- Are you accessing a shared group library? library_type is 'group'.

*P.S. I can't make Zotero.file() work properly, maybe because I use WebDAV instead of zotero to store the pdf files, so Zotero_dir is needed to find the PDFs in the file system.* 

**Step5**, modify the config.ini, choose to use ollama or openai (llama.cpp), LLM model, embedding model and so on. You can choose different models for convert PDF, embedding and chat.

## Usage
To convert PDF files into graph, just use the dpf2graph.py
```
python pdf2graph.py
```
Enter the path to the PDF file or just enter the title of the paper stored in Zotero library. Each PDF may take several minutes.

After convert, you can run graphQA.py to query this graph, enter exit to stop.
```
python graphQA.py
```

## References
[openai cookbook](https://github.com/openai/openai-cookbook/blob/main/examples/RAG_with_graph_db.ipynb)
[Langchain](https://github.com/langchain-ai/langchain/tree/master)
[Neo4j](https://github.com/neo4j/neo4j?tab=readme-ov-file)
[Pyzotero](https://github.com/urschrei/pyzotero)
[blogs](https://github.com/tomasonjo/blogs/blob/master/llm/enhancing_rag_with_graph.ipynb?ref=blog.langchain.dev)
