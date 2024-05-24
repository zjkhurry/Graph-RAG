import os
import pickle
import json_repair
import json
from typing import Any, Dict, List
from pathlib import Path
from pydantic import BaseModel
from configparser import ConfigParser
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, PromptTemplate
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_experimental.graph_transformers.llm import UnstructuredRelation
from langchain_core.messages import SystemMessage
from unstructured.partition.pdf import partition_pdf
from langchain_core.output_parsers import JsonOutputParser
from langchain_community.graphs import Neo4jGraph
from langchain_community.graphs.graph_document import GraphDocument, Node, Relationship

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

class Element(BaseModel):
    type: str
    text: Any

def exterat_elements_from_pdf(file_path: str, 
                              metadata: dict,
                              images: bool = False, 
                              max_char: int = 1000, 
                              new_after_n_chars: int = 800,
                              combine: int = 200,) -> List[Document]:
    # Define parameters for Unstructured's library
    strategy = "hi_res" # Strategy for analyzing PDFs and extracting table structure
    model_name = "yolox" # Best model for table extraction. Other options are detectron2_onnx and chipper depending on file layout
    # Extract images, tables, and chunk text
    raw_pdf_elements = partition_pdf(
        filename=file_path,
        extract_images_in_pdf=images,
        infer_table_structure=True,
        chunking_strategy="by_title",
        max_characters=max_char,
        new_after_n_chars=new_after_n_chars,
        combine_text_under_n_chars=combine,
        image_output_dir_path="./",
        strategy=strategy,
        model_name=model_name
    )

    # Create a dictionary to store counts of each type
    category_counts = {}

    for element in raw_pdf_elements:
        category = str(type(element))
        if category in category_counts:
            category_counts[category] += 1
        else:
            category_counts[category] = 1

    # Unique_categories will have unique elements
    # TableChunk if Table > max chars set above
    unique_categories = set(category_counts.keys())


    # Categorize by type
    categorized_elements = []
    for element in raw_pdf_elements:
        if "unstructured.documents.elements.Table" in str(type(element)):
            categorized_elements.append(Element(type="table", text=str(element.metadata.text_as_html)))
        elif "unstructured.documents.elements.CompositeElement" in str(type(element)):
            categorized_elements.append(Element(type="text", text=str(element)))

    # Tables
    table_elements = [e for e in categorized_elements if e.type == "table"]
    table_elements = [Document(page_content=i.text, metadata=metadata) for i in table_elements if i.text != ""]
    # Text
    text_elements = [e for e in categorized_elements if e.type == "text"]
    documents = [Document(page_content=i.text, metadata=metadata) for i in text_elements if i.text != ""]
    
    documents.extend(table_elements)
    metaDoct = Document(page_content=json.dumps(metadata),metadata=metadata)
    documents.append(metaDoct)
    return documents

examples = [
    {
        "text": (
            "Adam is a software engineer in Microsoft since 2009, "
            "and last year he got an award as the Best Talent"
        ),
        "head": "Adam",
        "head_type": "Person",
        "relation": "WORKS_FOR",
        "tail": "Microsoft",
        "tail_type": "Company",
    },
    {
        "text": (
            "Adam is a software engineer in Microsoft since 2009, "
            "and last year he got an award as the Best Talent"
        ),
        "head": "Adam",
        "head_type": "Person",
        "relation": "HAS_AWARD",
        "tail": "Best Talent",
        "tail_type": "Award",
    },
    {
        "text": (
            "Microsoft is a tech company that provide "
            "several products such as Microsoft Word"
        ),
        "head": "Microsoft Word",
        "head_type": "Product",
        "relation": "PRODUCED_BY",
        "tail": "Microsoft",
        "tail_type": "Company",
    },
    {
        "text": "Microsoft Word is a lightweight app that accessible offline",
        "head": "Microsoft Word",
        "head_type": "Product",
        "relation": "HAS_CHARACTERISTIC",
        "tail": "lightweight app",
        "tail_type": "Characteristic",
    },
    {
        "text": "Microsoft Word is a lightweight app that accessible offline",
        "head": "Microsoft Word",
        "head_type": "Product",
        "relation": "HAS_CHARACTERISTIC",
        "tail": "accessible offline",
        "tail_type": "Characteristic",
    },
]

# Create Prompt
def create_prompt():
    base_string_parts = [
        "You are a top-tier algorithm designed for extracting information in "
        "structured formats to build a knowledge graph. Your task is to identify "
        "the entities and relations requested with the user prompt from a given "
        "text. You must generate the output in a JSON format containing a list "
        'with JSON objects. Each object should have the keys: "head", '
        '"head_type", "relation", "tail", and "tail_type". The "head" '
        "key must contain the text of the extracted entity with one type.",
        "Attempt to extract as many entities and relations as you can. Maintain "
        "Entity Consistency: When extracting entities, it's vital to ensure "
        'consistency. If an entity, such as "John Doe", is mentioned multiple '
        "times in the text but is referred to by different names or pronouns "
        '(e.g., "Joe", "he"), always use the most complete identifier for '
        "that entity. The knowledge graph should be coherent and easily "
        "understandable, so maintaining consistency in entity references is "
        "crucial.",
        "IMPORTANT NOTES:\n- Don't add any explanation and text.",
    ]
    system_prompt = "\n".join(filter(None, base_string_parts))
    system_message = SystemMessage(content=system_prompt)
    parser = JsonOutputParser(pydantic_object=UnstructuredRelation)
    human_prompt = PromptTemplate(
        template="""Based on the following example, extract entities and 
relations from the provided text. Attempt to extract as many entities and relations as you can.\n\n

Below are a number of examples of text and their extracted entities and relationships.
{examples}

For the following text or table, extract entities and relations as in the provided example. Table is in HTML format.
{format_instructions}\nText: {input}
IMPORTANT NOTES:\n- Each key must have a valid value, 'null' is not allowed. \n- Don't add any explanation and text. \n- Extract information as much as possible""",
        input_variables=["input"],
        partial_variables={
            "format_instructions": parser.get_format_instructions(),
            "examples": examples,
        },
    )

    human_message_prompt = HumanMessagePromptTemplate(prompt=human_prompt)
    chat_prompt = ChatPromptTemplate.from_messages([system_message, human_message_prompt])
    return chat_prompt

prompt = create_prompt()
llm_transformer = LLMGraphTransformer(llm=llm, prompt=prompt)
json_repair = json_repair
chain = prompt | llm 

def process_response(document,i,j,metadata) -> GraphDocument:
    print(f"processing document {i} out of {j}")
    
    raw_schema=chain.invoke({"input": document.page_content})
    parsed_json = json_repair.loads(raw_schema.content)

    nodes_set = set()
    relationships = []
    # print(type(metadata))
    for rel in parsed_json:
        try:
            # Nodes need to be deduplicated using a set
            rel["head_type"] = rel["head_type"] if rel["head_type"] else "Unknown"
            rel["tail_type"] = rel["tail_type"] if rel["tail_type"] else "Unknown"
            nodes_set.add((rel["head"], rel["head_type"]))
            nodes_set.add((rel["tail"], rel["tail_type"]))
            source_node = Node(
                id=rel["head"],
                type=rel["head_type"]
            )
            target_node = Node(
                id=rel["tail"],
                type=rel["tail_type"]
            )
            relationships.append(
                Relationship(
                    source=source_node,
                    target=target_node,
                    type=rel["relation"]
                )
            )
        except:
            print(f"Error processing relation: {rel}")
    # Create nodes list
    nodes = [Node(id=el[0], type=el[1]) for el in list(nodes_set)]
    return GraphDocument(nodes=nodes, relationships=relationships, source=document)

def flatten_json(y):
    out = {}

    def flatten(x, name=''):
        if type(x) is dict:
            for a in x:
                flatten(x[a], name + a + '_')
        elif type(x) is list:
            i = 0
            for a in x:
                flatten(a, name + str(i) + '_')
                i += 1
        else:
            out[name[:-1]] = x

    flatten(y)
    return out

def process_document(file_path: str, 
                     meta: dict,
                    images: bool = False, 
                    max_char: int = 1000, 
                    new_after_n_chars: int = 800,
                    combine: int = 200) -> None:
    # metadata = json.dumps(meta)
    metadata = flatten_json(meta)
    if os.path.exists("output.pkl"):
        with open("output.pkl", "rb") as f:
            text_summaries = pickle.load(f)
    else:
        documents = exterat_elements_from_pdf(file_path,metadata,images,max_char,new_after_n_chars,combine)

        text_summaries = [process_response(document,i,len(documents),metadata) for i,document in enumerate(documents)]
        with open("output.pkl", "wb") as f:
            pickle.dump(text_summaries, f)
        
    graph.add_graph_documents(text_summaries,
    baseEntityLabel=True, 
    include_source=True)

if config["Zotero"]["enabled"]:
    from pyzotero import zotero
    library_id = config["Zotero"]["library_id"]
    library_type = config["Zotero"]["library_type"]
    api_key = config["Zotero"]["api_key"]
    zot = zotero.Zotero(library_id, library_type, api_key)
    dir = config["Zotero"]["Zotero_dir"]

def process_zotero_document(item: dict, papers: list) -> None:
    if 'title' in item['data']:
        print('Item: %s ' % (item['data']['title']))
    else:
        print('Item: %s ' % (item['key']))
    
    meta = item
    zot_file = ''
    if item['data'].get('contentType') == 'application/pdf':
        zot_file = dir + item['key'] + '/'
        for file in os.listdir(zot_file):
            if file.endswith(".pdf"):
                zot_file += file
                break
            else:
                zot_file = ''
            
    else:
        if item['meta'].get('numChildren',0) > 0:
            it=zot.children(item['key'])
            for i in it:
                if i['data'].get('contentType') == 'application/pdf':
                    zot_file = dir + i['key'] + '/'
                    for file in os.listdir(zot_file):
                        if file.endswith(".pdf"):
                            zot_file += file
                            break
                        else:
                            zot_file = ''
        
    if zot_file == '':
        print(f"Document not found: {item['key']}")
        return 0
    if not os.path.exists(zot_file):
        return 0
    if zot_file in papers:
        print(f"Document already processed: {zot_file}")
        return 0
    process_document(
        zot_file,
        meta,
        images=bool(config["PDF"]["extract_images"]),
        max_char=int(config["PDF"]["max_char"]),
        new_after_n_chars=int(config["PDF"]["new_after_n_chars"]),
        combine=int(config["PDF"]["combine_text_under_n_chars"]),
        )
    papers.append(zot_file)
    with open("papers.pkl", "wb") as f:
        pickle.dump(papers, f)
    if os.path.exists("output.pkl"):
        os.remove("output.pkl")

def find_in_zotero(title: str) -> None:
    if os.path.exists("papers.pkl"):
        with open("papers.pkl", "rb") as f:
            papers = pickle.load(f)
    else:
        papers = []
    items = zot.items(q=title,limit=10)
    if len(items)>1:
        for i,item in enumerate(items):
            if 'title' in item['data']:
                print('[%d]: %s ' % (i,item['data']['title']))
            else:
                print('[%d]: %s ' % (i,item['key']))
        print("Select the document to process:")
        index = int(input())
        if index < 0 or index >= len(items):
            print("Invalid index.")
            return
    else:
        index = 0
    item = items[index]
    process_zotero_document(item,papers)
        

def all_in_zotero() -> None:
    if os.path.exists("papers.pkl"):
        with open("papers.pkl", "rb") as f:
            papers = pickle.load(f)
    else:
        papers = []
    processed = len(papers)
    num = zot.count_items()
    while(processed<num):
        items = zot.items(start = processed, item_type='attachment')
        for item in items:
            process_zotero_document(item,papers)

# CLI interaction
def cli():
    while True:
        action = input("\nEnter a PDF path or Zotero paper title, or 'all' for all the items in Zotero, or 'quit' to exit: ")
        if os.path.exists(action):
            _, ext = os.path.splitext(action)
            if ext == '.pdf':
                print("Processing document %s" % action)
                process_document(
                    action, 
                    {"source": action}, 
                    images=bool(config["PDF"]["extract_images"]),
                    max_char=int(config["PDF"]["max_char"]),
                    new_after_n_chars=int(config["PDF"]["new_after_n_chars"]),
                    combine=int(config["PDF"]["combine_text_under_n_chars"]),
                )
            else:
                try:
                    for file in os.listdir(action):
                        if file.endswith(".pdf"):
                            print("Processing document %s" % file)
                            process_document(
                                action + '/' + file, 
                                {"source": action}, 
                                images=bool(config["PDF"]["extract_images"]),
                                max_char=int(config["PDF"]["max_char"]),
                                new_after_n_chars=int(config["PDF"]["new_after_n_chars"]),
                                combine=int(config["PDF"]["combine_text_under_n_chars"]),
                            )
                except:
                    print("Invalid input.")
        elif action == 'all':
            if config["Zotero"]["enabled"]:
                all_in_zotero()
            else:
                print("Zotero is not enabled in the configuration file.")
        elif action == 'quit':
            break
        else:
            try:
                find_in_zotero(action)
            except:
                print("Invalid input.")

if __name__ == "__main__":
    cli()
