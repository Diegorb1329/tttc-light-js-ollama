#!/usr/bin/env python

########################################
# T3C Pyserver: LLM Pipeline in Python #
# --------------------------------------#
"""A minimal FastAPI Python server calling the T3C LLM pipeline.

Each pipeline call assumes the client has already included
any user edits of the LLM configuration, including the model
name to use, the system prompt, and the specific pipeline step prompts.

Currently only supports OpenAI (Anthropic soon!!!)
For local testing, load these from a config.py file
"""

import json
from json import JSONDecodeError
import math
import os
import re
import sys
from pathlib import Path
from typing import List

import wandb
from dotenv import load_dotenv
from fastapi import FastAPI, Header
from openai import OpenAI
from pydantic import BaseModel

# Importar adaptador Ollama
from .ollama_openai_adapter import create_client
from . import ollama_config

# Importar parser JSON desde tests
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "ollama-tests" / "tests" / "phase3_integration"))
from json_response_parser import extract_json_from_response



# Add the current directory to path for imports
current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir))
from . import config
from .utils import cute_print, full_speaker_map, token_cost, topic_desc_map, comment_is_meaningful

load_dotenv()

app = FastAPI()



def get_model_name(model_name: str) -> str:
    """
    Obtener el nombre del modelo correcto según la configuración
    """
    if ollama_config.should_use_ollama():
        return ollama_config.get_ollama_model(model_name)
    else:
        return model_name


def create_llm_client(api_key: str = None):
    """
    Crear cliente LLM según la configuración
    """
    if ollama_config.should_use_ollama():
        return create_client(
            base_url=ollama_config.OLLAMA_BASE_URL,
            model=ollama_config.OLLAMA_DEFAULT_MODEL
        )
    else:
        return OpenAI(api_key=api_key)


def get_llm_client(api_key: str = None, model_name: str = None):
    """
    Obtener cliente LLM (OpenAI o Ollama) basado en configuración
    """
    if ollama_config.should_use_ollama():
        # Usar Ollama con modelo mapeado
        ollama_model = ollama_config.get_ollama_model(model_name) if model_name else ollama_config.OLLAMA_DEFAULT_MODEL
        client = create_client(
            base_url=ollama_config.OLLAMA_BASE_URL,
            model=ollama_model
        )
        print(f"🦙 Usando Ollama: {ollama_model}")
        return client, ollama_model
    else:
        # Usar OpenAI original
        client = OpenAI(api_key=api_key)
        print(f"🤖 Usando OpenAI: {model_name}")
        return client, model_name

class Comment(BaseModel):
    id: str
    text: str
    speaker: str


class CommentList(BaseModel):
    comments: List[Comment]


class LLMConfig(BaseModel):
    model_name: str
    system_prompt: str
    user_prompt: str


class CommentsLLMConfig(BaseModel):
    comments: List[Comment]
    llm: LLMConfig


class CommentTopicTree(BaseModel):
    comments: List[Comment]
    llm: LLMConfig
    tree: dict


class ClaimTreeLLMConfig(BaseModel):
    tree: dict
    llm: LLMConfig
    sort: str


class CruxesLLMConfig(BaseModel):
    crux_tree: dict
    llm: LLMConfig
    topics: list
    top_k: int

@app.get("/")
def read_root():
    # TODO: setup/relevant defaults?
    return {"Hello": "World"}

###################################
# Step 1: Comments to Topic Tree  #
# ---------------------------------#
@app.post("/topic_tree")
def comments_to_tree(
    req: CommentsLLMConfig,
    x_openai_api_key: str = Header(..., alias="X-OpenAI-API-Key"),
    log_to_wandb: str = config.WANDB_GROUP_LOG_NAME,
    dry_run=False,
) -> dict:
    """Given the full list of comments, return a corresponding taxonomy of relevant topics and their
    subtopics, with a short description for each.

    Input format:
    - CommentLLMConfig object: JSON/dictionary with the following fields:
      - comments: a list of Comment (each has a field, "text", for the raw text of the comment, and an id)
      - llm: a dictionary of the LLM configuration:
        - model_name: a string of the name of the LLM to call ("gpt-4o-mini", "gpt-4-turbo-preview")
        - system_prompt: a string of the system prompt
        - user_prompt: a string of the user prompt to convert the raw comments into the
                             taxonomy/topic tree
    Example:
    {
      "llm": {
          "model_name": "gpt-4o-mini",
          "system_prompt": "\n\tYou are a professional research assistant.",
          "topic_tree_prompt": "\nI will give you a list of comments."
      },
      "comments": [
          {
              "id": "c1",
              "text": "I love cats"
          },
          {
              "id": "c2",
              "text": "dogs are great"
          },
          {
              "id": "c3",
              "text": "I'm not sure about birds"
          }
      ]
    }

    Output format:
    - data : the tree as a dictionary
      - taxonomy : a key mapping to a list of topics, where each topic has
        - topicName: a string of the short topic title
        - topicShortDescription: a string of a short description of the topic
        - subtopics: a list of the subtopics of this main/parent topic, where each subtopic has
          - subtopicName: a string of the short subtopic title
          - subtopicShortDescription: a string of a short description of the subtopic
    - usage: a dictionary of token counts
      - completion_tokens
      - prompt_tokens
      - total_tokens

    Example output:
    {
      "data": {
          "taxonomy": [
              {
                  "topicName": "Pets",
                  "topicShortDescription": "General opinions about common household pets.",
                  "subtopics": [
                      {
                          "subtopicName": "Cats",
                          "subtopicShortDescription": "Positive sentiments towards cats as pets."
                      },
                      {
                          "subtopicName": "Dogs",
                          "subtopicShortDescription": "Positive sentiments towards dogs as pets."
                      },
                      {
                          "subtopicName": "Birds",
                          "subtopicShortDescription": "Uncertainty or mixed feelings about birds as pets."
                      }
                  ]
              }
          ]
      },
      "usage": {
          "completion_tokens": 131,
          "prompt_tokens": 224,
          "total_tokens": 355
      }
    }
    """
    # skip calling an LLM
    if dry_run or config.DRY_RUN:
        print("dry_run topic tree")
        return config.MOCK_RESPONSE["topic_tree"]
    
    # Obtener cliente LLM (OpenAI o Ollama)
    client, actual_model = get_llm_client(x_openai_api_key, req.llm.model_name)

    # append comments to prompt
    full_prompt = req.llm.user_prompt
    for comment in req.comments:
        # skip any empty comments/rows
        if comment_is_meaningful(comment.text):
            full_prompt += "\n" + comment.text
        else:
            print("warning:empty comment in topic_tree:" + comment.text)

    # Para Ollama, modificar prompts para asegurar salida JSON
    system_prompt = req.llm.system_prompt
    if ollama_config.should_use_ollama():
        # Prompts optimizados para Llama3.2 - más directo y específico
        system_prompt = "You are a JSON generator. You MUST respond with ONLY valid JSON. No text before or after the JSON. Each topic MUST have at least one subtopic."
        full_prompt += "\n\n<JSON_OUTPUT_REQUIRED>\nGenerate a JSON taxonomy with this EXACT structure. Every topic MUST include subtopics array:\n{\"taxonomy\": [{\"topicName\": \"Topic Name\", \"topicShortDescription\": \"Description of the topic\", \"subtopics\": [{\"subtopicName\": \"Subtopic Name\", \"subtopicShortDescription\": \"Description of subtopic\"}]}]}\n\nIMPORTANT: Each topic must have at least 1 subtopic in the subtopics array. Do not omit the subtopics field.\n</JSON_OUTPUT_REQUIRED>"

    # Preparar argumentos para la llamada
    call_args = {
        "model": actual_model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": full_prompt},
        ],
        "temperature": 0.0,
    }
    
    # Configuraciones específicas según el backend
    if ollama_config.should_use_ollama():
        # Para Ollama: deshabilitar thinking
        call_args["think"] = False
    else:
        # Para OpenAI: usar response_format JSON
        call_args["response_format"] = {"type": "json_object"}
    
    response = client.chat.completions.create(**call_args)
    try:
        content = response.choices[0].message.content
        print(f"Raw response content: {content[:500]}...")  # Log para debug
        
        # Usar la función de extracción mejorada
        tree = extract_json_from_response(content)
        
        # Validar y normalizar estructura completa
        if not isinstance(tree, dict):
            print("Warning: Invalid tree structure, creating default")
            tree = {"taxonomy": []}
        
        if "taxonomy" not in tree:
            print("Warning: No taxonomy field found, creating empty taxonomy")
            tree["taxonomy"] = []
        
        if not isinstance(tree["taxonomy"], list):
            print("Warning: taxonomy field is not a list, converting to empty list")
            tree["taxonomy"] = []
            
        # Normalizar estructura: asegurar que cada topic tenga subtopics
        for topic in tree["taxonomy"]:
            if not isinstance(topic, dict):
                continue
                
            if "subtopics" not in topic or not isinstance(topic["subtopics"], list):
                # Si no tiene subtopics, crear uno genérico basado en el topic
                topic_name = topic.get("topicName", "Unknown Topic")
                topic["subtopics"] = [{
                    "subtopicName": f"General {topic_name}",
                    "subtopicShortDescription": f"General aspects of {topic_name.lower()}"
                }]
                print(f"Warning: Added default subtopic for topic '{topic_name}'")
        
        print(f"Successfully parsed JSON with {len(tree.get('taxonomy', []))} topics")
            
    except Exception as e:
        print("Step 1: no topic tree: ", response)
        print("Parse error:", str(e))
        tree = {}
    usage = response.usage
    # compute LLM costs for this step's tokens
    s1_total_cost = token_cost(
        req.llm.model_name, usage.prompt_tokens, usage.completion_tokens,
    )

    if log_to_wandb:
        try:
            exp_group_name = str(log_to_wandb)
            wandb.init(
                project=config.WANDB_PROJECT_NAME, group=exp_group_name, resume="allow",
            )
            wandb.config.update(
                {
                    "s1_topics/model": req.llm.model_name,
                    "s1_topics/user_prompt": req.llm.user_prompt,
                    "s1_topics/system_prompt": req.llm.system_prompt,
                },
            )
            comment_lengths = [len(c.text) for c in req.comments]
            
            # Manejo seguro de datos de tree para W&B
            taxonomy = tree.get("taxonomy", [])
            num_topics = len(taxonomy) if isinstance(taxonomy, list) else 0
            subtopic_bins = []
            for t in taxonomy:
                if isinstance(t, dict) and "subtopics" in t and isinstance(t["subtopics"], list):
                    subtopic_bins.append(len(t["subtopics"]))
                else:
                    subtopic_bins.append(0)

            # in case comments are empty / for W&B Table logging
            comment_list = "none"
            if len(req.comments) > 1:
                comment_list = "\n".join([c.text for c in req.comments])
            
            try:
                taxonomy_json = json.dumps(taxonomy, indent=1)
            except Exception:
                taxonomy_json = "Error serializing taxonomy"
            comms_tree_list = [[comment_list, taxonomy_json]]
            wandb.log(
                {
                    "comm_N": len(req.comments),
                    "comm_text_len": sum(comment_lengths),
                    "comm_bins": comment_lengths,
                    "num_topics": num_topics,
                    "num_subtopics": sum(subtopic_bins),
                    "subtopic_bins": subtopic_bins,
                    "rows_to_tree": wandb.Table(
                        data=comms_tree_list, columns=["comments", "taxonomy"],
                    ),
                    # token counts
                    "U_tok_N/taxonomy": usage.total_tokens,
                    "U_tok_in/taxonomy": usage.prompt_tokens,
                    "U_tok_out/taxonomy": usage.completion_tokens,
                    "cost/s1_topics": s1_total_cost,
                },
            )
        except Exception:
            print("Failed to create wandb run")
    # NOTE: El Express server espera que data sea directamente el array de topics
    return {
        "data": tree.get("taxonomy", []),
        "usage": usage.model_dump(),
        "cost": s1_total_cost,
    }


def comment_to_claims(llm: LLMConfig, comment: str, tree: dict, api_key: str) -> dict:
    """Given a comment and the full taxonomy/topic tree for the report, extract one or more claims from the comment.
    
    Args:
        llm (dict): The LLM configuration, including model name, system prompt, and user prompt.
        comment (str): The comment text to analyze and extract claims from.
        tree (dict): The taxonomy/topic tree to provide context for the comment.
        api_key (str): The API key for authenticating with the OpenAI client.
    
    Returns:
        dict: A dictionary containing the extracted claims and usage information.
    """
    # Obtener cliente LLM (OpenAI o Ollama)
    client, actual_model = get_llm_client(api_key, llm.model_name)

    # add taxonomy and comment to prompt template
    taxonomy_string = json.dumps(tree)

    # TODO: prompt nit, shorten this to just "Comment:"
    full_prompt = llm.user_prompt
    full_prompt += (
        "\n" + taxonomy_string + "\nAnd then here is the comment:\n" + comment
    )

    # Para Ollama, modificar prompts para asegurar salida JSON
    system_prompt = llm.system_prompt
    if ollama_config.should_use_ollama():
        # Prompts optimizados para Llama3.2
        system_prompt = "You are a JSON generator. You MUST respond with ONLY valid JSON. No text before or after the JSON."
        full_prompt += f"\n\n<JSON_OUTPUT_REQUIRED>\nExtract claims and respond with valid JSON in this EXACT format:\n{{\n  \"claims\": [\n    {{\n      \"claim\": \"string\",\n      \"quote\": \"string\",\n      \"topicName\": \"string\",\n      \"subtopicName\": \"string\"\n    }}\n  ]\n}}\nEnsure ALL claims have topicName and subtopicName fields.\n</JSON_OUTPUT_REQUIRED>"

    # Preparar argumentos para la llamada
    call_args = {
        "model": actual_model,
        "messages": [
            {
                "role": "system",
                "content": system_prompt,
            },
            {"role": "user", "content": full_prompt},
        ],
        "temperature": 0.0,
    }
    
    # Configuraciones específicas según el backend
    if ollama_config.should_use_ollama():
        # Para Ollama: deshabilitar thinking para mayor velocidad
        call_args["think"] = False
    else:
        # Para OpenAI: usar response_format JSON
        call_args["response_format"] = {"type": "json_object"}
    
    response = client.chat.completions.create(**call_args)
    try:
        content = response.choices[0].message.content
        print(f"Raw claims response: {content[:200]}...")  # Log para debug
        
        # Usar la función de extracción mejorada para claims también
        claims_obj = extract_json_from_response(content)
        
        # Normalizar la estructura de claims - manejar tanto arrays directos como objetos con clave 'claims'
        if isinstance(claims_obj, list):
            # Si es un array directo, envolver en objeto
            claims_obj = {"claims": claims_obj}
            claims_count = len(claims_obj["claims"])
        elif isinstance(claims_obj, dict) and "claims" in claims_obj:
            # Si es un objeto con clave 'claims'
            claims_count = len(claims_obj["claims"]) if isinstance(claims_obj["claims"], list) else 0
        else:
            # Formato inesperado, crear estructura por defecto
            claims_obj = {"claims": []}
            claims_count = 0
            
        print(f"Successfully parsed claims JSON with {claims_count} claims")
        
    except Exception as e:
        print("Step 2: no response: ", response)
        print("Claims parse error:", str(e))
        claims_obj = {"claims": []}  # Estructura por defecto para claims
        
    return {"claims": claims_obj, "usage": response.usage}


####################################
# Step 2: Extract and place claims #
# ----------------------------------#
@app.post("/claims")
def all_comments_to_claims(
    req: CommentTopicTree, x_openai_api_key: str = Header(..., alias="X-OpenAI-API-Key"), log_to_wandb: str = config.WANDB_GROUP_LOG_NAME, dry_run = False
) -> dict:
    """Given a comment and the taxonomy/topic tree for the report, extract one or more claims from the comment.
    Place each claim under the correct subtopic in the tree.

    Input format:
    - CommentTopicTree object: JSON/dictionary with the following fields:
      - comments: a list of Comment (each has a field, "text", for the raw text of the comment, and an id)
      - llm: a dictionary of the LLM configuration:
        - model_name: a string of the name of the LLM to call ("gpt-4o-mini", "gpt-4-turbo-preview")
        - system_prompt: a string of the system prompt
        - user_prompt: a string of the user prompt to convert the raw comments into the
                             taxonomy/topic tree
     - tree: a dictionary of the topics and nested subtopics, and their titles/descriptions
    Example:
    {
      "llm": {
          "model_name": "gpt-4o-mini",
          "system_prompt": "\n\tYou are a professional research assistant.",
          "user_prompt": "\nI'm going to give you a comment made by a participant",
      },
      "comments": [
          {
              "id": "c1",
              "text": "I love cats"
          },
          {
              "id": "c2",
              "text": "dogs are great"
          },
          {
              "id": "c3",
              "text": "I'm not sure about birds"
          }
      ],
      "tree": [
                {
                  "topicName": "Pets",
                  "topicShortDescription": "General opinions about common household pets.",
                  "subtopics": [
                      {
                          "subtopicName": "Cats",
                          "subtopicShortDescription": "Positive sentiments towards cats."
                      },
                      {
                          "subtopicName": "Dogs",
                          "subtopicShortDescription": "Positive sentiments towards dogs."
                      },
                      {
                          "subtopicName": "Birds",
                          "subtopicShortDescription": "Uncertainty or mixed feelings about birds."
                      }
                  ]
                }
             ]
    }

    Output format:
    - data: the dictionary of topics and subtopics with extracted claims listed under the
                   correct subtopic, along with the source quote
    - usage: a dictionary of token counts for the LLM calls of this pipeline step
      - completion_tokens
      - prompt_tokens
      - total_tokens

    Example output:
    {
      "data": {
          "Pets": {
              "total": 3,
              "subtopics": {
                  "Cats": {
                      "total": 1,
                      "claims": [
                          {
                              "claim": "Cats are the best household pets.",
                              "commentId":"c1",
                              "quote": "I love cats",
                              "topicName": "Pets",
                              "subtopicName": "Cats"
                          }
                      ]
                  },
                  "Dogs": {
                      "total": 1,
                      "claims": [
                          {
                              "claim": "Dogs are superior pets.",
                              "commentId":"c2",
                              "quote": "dogs are great",
                              "topicName": "Pets",
                              "subtopicName": "Dogs"
                          }
                      ]
                  },
                  "Birds": {
                      "total": 1,
                      "claims": [
                          {
                              "claim": "Birds are not suitable pets for everyone.",
                              "commentId":"c3",
                              "quote": "I'm not sure about birds.",
                              "topicName": "Pets",
                              "subtopicName": "Birds"
                          }
                      ]
                  }
              }
          }
      }
    }
    """
    # skip calling an LLM
    if dry_run or config.DRY_RUN:
        print("dry_run claims")
        return config.MOCK_RESPONSE["claims"]
    comms_to_claims = []
    comms_to_claims_html = []
    TK_2_IN = 0
    TK_2_OUT = 0
    TK_2_TOT = 0

    node_counts = {}
    # TODO: batch this so we're not sending the tree each time
    for i_c, comment in enumerate(req.comments):
        # TODO: timing for comments
        # print("comment: ", i_c)
        # print("time: ", datetime.now())
        if comment_is_meaningful(comment.text):
            response = comment_to_claims(req.llm, comment.text, req.tree, x_openai_api_key)
        else:
            print("warning: empty comment in claims:" + comment.text)
            continue
        try:
            claims = response["claims"]
            # Verificar que claims tenga la estructura esperada
            if not isinstance(claims, dict) or "claims" not in claims:
                print(f"Unexpected claims structure: {claims}")
                claims = {"claims": []}
            
            for claim in claims["claims"]:
                claim.update({"commentId": comment.id, "speaker": comment.speaker})
        except Exception as e:
            print(f"Step 2: no claims for comment (error: {str(e)}): ", response)
            claims = None
            continue
        # reference format
        # {'claims': [{'claim': 'Dogs are superior pets.', commentId:'c1', 'quote': 'dogs are great', 'topicName': 'Pets', 'subtopicName': 'Dogs'}]}
        usage = response["usage"]
        if claims and len(claims["claims"]) > 0:
            comms_to_claims.extend([c for c in claims["claims"]])

        TK_2_IN += usage.prompt_tokens
        TK_2_OUT += usage.completion_tokens
        TK_2_TOT += usage.total_tokens

        # format for logging to W&B
        if log_to_wandb:
            viz_claims = cute_print(claims["claims"])
            comms_to_claims_html.append([comment.text, viz_claims])

    # reference format
    # [{'claim': 'Cats are the best household pets.', 'commentId':'c1', 'quote': 'I love cats', 'speaker' : 'Alice', 'topicName': 'Pets', 'subtopicName': 'Cats'},
    # {'commentId':'c2','claim': 'Dogs are superior pets.', 'quote': 'dogs are great', 'speaker' : 'Bob', 'topicName': 'Pets', 'subtopicName': 'Dogs'},
    # {'commentId':'c3', 'claim': 'Birds are not suitable pets for everyone.', 'quote': "I'm not sure about birds.", 'speaker' : 'Alice', 'topicName': 'Pets', 'subtopicName': 'Birds'}]

    # count the claims in each subtopic
    for claim in comms_to_claims:
        if "topicName" not in claim:
            print("claim unassigned to topic: ", claim)
            # Intentar asignar a un tópico por defecto basado en los tópicos disponibles
            if req.tree and "taxonomy" in req.tree and len(req.tree["taxonomy"]) > 0:
                default_topic = req.tree["taxonomy"][0]["topicName"]
                print(f"Assigning claim to default topic: {default_topic}")
                claim["topicName"] = default_topic
                if "subtopics" in req.tree["taxonomy"][0] and len(req.tree["taxonomy"][0]["subtopics"]) > 0:
                    claim["subtopicName"] = req.tree["taxonomy"][0]["subtopics"][0]["subtopicName"]
                else:
                    claim["subtopicName"] = "General"
            else:
                continue
        if claim["topicName"] in node_counts:
            node_counts[claim["topicName"]]["total"] += 1
            node_counts[claim["topicName"]]["speakers"].add(claim["speaker"])
            if "subtopicName" in claim:
                if (
                    claim["subtopicName"]
                    in node_counts[claim["topicName"]]["subtopics"]
                ):
                    node_counts[claim["topicName"]]["subtopics"][claim["subtopicName"]][
                        "total"
                    ] += 1
                    node_counts[claim["topicName"]]["subtopics"][claim["subtopicName"]][
                        "claims"
                    ].append(claim)
                    node_counts[claim["topicName"]]["subtopics"][claim["subtopicName"]][
                        "speakers"
                    ].add(claim["speaker"])
                else:
                    node_counts[claim["topicName"]]["subtopics"][
                        claim["subtopicName"]
                    ] = {
                        "total": 1,
                        "claims": [claim],
                        "speakers": set([claim["speaker"]]),
                    }
        else:
            node_counts[claim["topicName"]] = {
                "total": 1,
                "speakers": set([claim["speaker"]]),
                "subtopics": {
                    claim["subtopicName"]: {
                        "total": 1,
                        "claims": [claim],
                        "speakers": set([claim["speaker"]]),
                    },
                },
            }
    # after inserting claims: check if any of the topics/subtopics are empty
    for topic in req.tree["taxonomy"]:
        if "subtopics" in topic:
            for subtopic in topic["subtopics"]:
                # check if subtopic in node_counts
                if topic["topicName"] in node_counts:
                    if (
                        subtopic["subtopicName"]
                        not in node_counts[topic["topicName"]]["subtopics"]
                    ):
                        # this is an empty subtopic!
                        print("EMPTY SUBTOPIC: ", subtopic["subtopicName"])
                        node_counts[topic["topicName"]]["subtopics"][
                            subtopic["subtopicName"]
                        ] = {"total": 0, "claims": [], "speakers": set()}
                else:
                    # could we have an empty topic? certainly
                    print("EMPTY TOPIC: ", topic["topicName"])
                    node_counts[topic["topicName"]] = {
                        "total": 0,
                        "speakers": set(),
                        "subtopics": {
                            "None": {"total": 0, "claims": [], "speakers": set()},
                        },
                    }
    # compute LLM costs for this step's tokens
    s2_total_cost = token_cost(req.llm.model_name, TK_2_IN, TK_2_OUT)

    # Note: we will now be sending speaker names to W&B
    if log_to_wandb:
        try:
            exp_group_name = str(log_to_wandb)
            wandb.init(
                project=config.WANDB_PROJECT_NAME, group=exp_group_name, resume="allow",
            )
            wandb.config.update(
                {
                    "s2_claims/model": req.llm.model_name,
                    "s2_claims/user_prompt": req.llm.user_prompt,
                    "s2_claims/system_prompt": req.llm.system_prompt,
                },
            )
            wandb.log(
                {
                    "U_tok_N/claims": TK_2_TOT,
                    "U_tok_in/claims": TK_2_IN,
                    "U_tok_out/claims": TK_2_OUT,
                    "rows_to_claims": wandb.Table(
                        data=comms_to_claims_html, columns=["comments", "claims"],
                    ),
                    "cost/s2_claims": s2_total_cost,
                },
            )
        except Exception:
            print("Failed to log wandb run")

    net_usage = {
        "total_tokens": TK_2_TOT,
        "prompt_tokens": TK_2_IN,
        "completion_tokens": TK_2_OUT,
    }
    return {"data": node_counts, "usage": net_usage, "cost": s2_total_cost}


def dedup_claims(claims: list, llm: LLMConfig, api_key: str) -> dict:
    """Given a list of claims for a given subtopic, identify which ones are near-duplicates.

    Args:  
        claims (list): A list of claims to be deduplicated.  
        llm (LLMConfig): The LLM configuration containing prompts and model details.  
        api_key (str): The API key for authenticating with the OpenAI client.  
    
    Returns:  
        dict: A dictionary containing the deduplicated claims and usage information.  
    """
    # Obtener cliente LLM (OpenAI o Ollama)
    client, actual_model = get_llm_client(api_key, llm.model_name)

    # add claims with enumerated ids (relative to this subtopic only)
    full_prompt = llm.user_prompt
    for i, orig_claim in enumerate(claims):
        full_prompt += "\nclaimId" + str(i) + ": " + orig_claim["claim"]

    # Para Ollama, modificar prompts para asegurar salida JSON
    system_prompt = llm.system_prompt
    if ollama_config.should_use_ollama():
        # Prompts optimizados para Llama3.1
        system_prompt = "You are a JSON generator. You MUST respond with ONLY valid JSON. No text before or after the JSON."
        full_prompt += "\n\n<JSON_OUTPUT_REQUIRED>\nAnalyze duplicates and respond with valid JSON containing the deduplicated claims.\n</JSON_OUTPUT_REQUIRED>"

    # Preparar argumentos para la llamada
    call_args = {
        "model": actual_model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": full_prompt},
        ],
        "temperature": 0.0,
    }
    
    # Configuraciones específicas según el backend
    if ollama_config.should_use_ollama():
        # Para Ollama: deshabilitar thinking para mayor velocidad
        call_args["think"] = False
    else:
        # Para OpenAI: usar response_format JSON
        call_args["response_format"] = {"type": "json_object"}
    
    response = client.chat.completions.create(**call_args)
    try:
        content = response.choices[0].message.content
        print(f"Raw dedup response: {content[:200]}...")  # Log para debug
        
        # Usar la función de extracción mejorada para dedup también
        deduped_claims_obj = extract_json_from_response(content)
        # Verificar estructura básica para dedup
        if not isinstance(deduped_claims_obj, dict):
            print(f"Unexpected dedup structure: {deduped_claims_obj}")
            deduped_claims_obj = {"nesting": {}}
        print(f"Successfully parsed dedup JSON with keys: {list(deduped_claims_obj.keys())}")
        
    except Exception as e:
        print("Step 3: no deduped claims: ", response)
        print("Dedup parse error:", str(e))
        deduped_claims_obj = {"nesting": {}}  # Estructura por defecto para dedup
    return {"dedup_claims": deduped_claims_obj, "usage": response.usage}


#####################################
# Step 3: Sort & deduplicate claims #
# -----------------------------------#
@app.put("/sort_claims_tree/")
def sort_claims_tree(
    req: ClaimTreeLLMConfig, x_openai_api_key: str = Header(..., alias="X-OpenAI-API-Key"), log_to_wandb: str = config.WANDB_GROUP_LOG_NAME, dry_run = False
) -> dict:
    """Sort the topic/subtopic tree so that the most popular claims, subtopics, and topics
    all appear first. Deduplicate claims within each subtopic so that any near-duplicates appear as
    nested/child objects of a first-in parent claim, under the key "duplicates"

    Input format:
    - ClaimTree object: JSON/dictionary with the following fields
      - tree: the topic tree / full taxonomy of topics, subtopics, and claims (each with their full schema,
              including the claim, quote, topic, and subtopic)
    Example input tree:
    {
     "tree" : {
      "Pets": {
          "total": 5,
          "subtopics": {
              "Cats": {
                  "total": 2,
                  "claims": [
                      {
                          "claim": "Cats are the best pets.",
                          "commentId":"c1",
                          "quote": "I love cats.",
                          "topicName": "Pets",
                          "subtopicName": "Cats"
                      },
                      {
                          "claim": "Cats are the best pets.",
                          "commentId":"c1",
                          "quote": "I really really love cats",
                          "topicName": "Pets",
                          "subtopicName": "Cats"
                      }
                  ]
              },
              "Dogs": {
                  "total": 1,
                  "claims": [
                      {
                          "claim": "Dogs are superior pets.",
                          "commentId":"c2",
                          "quote": "dogs are great",
                          "topicName": "Pets",
                          "subtopicName": "Dogs"
                      }
                  ]
              },
              "Birds": {
                  "total": 2,
                  "claims": [
                      {
                          "claim": "Birds are not ideal pets for everyone.",
                          "commentId":"c3",
                          "quote": "I'm not sure about birds.",
                          "topicName": "Pets",
                          "subtopicName": "Birds"
                      },
                      {
                          "claim": "Birds are not suitable pets for everyone.",
                          "commentId":"c3",
                          "quote": "I don't know about birds.",
                          "topicName": "Pets",
                          "subtopicName": "Birds"
                      }
                  ]
              }
          }
      }
     }
    }
    Output format:
    - response object: JSON/dictionary with the following fields
      - data: the deduplicated claims & correctly sorted topic tree / full taxonomy of topics, subtopics,
              and claims, where the most popular topics/subtopics/claims (by near-duplicate count) appear
              first within each level of nesting
      - usage: token counts for the LLM calls of the deduplication step of the pipeline
        - completion_tokens
        - prompt_tokens
        - total_tokens

    Example output tree:
    [
      [
          "Pets",
          {
              "num_speakers" : 5,
              "speakers" : [
                  "Alice",
                  "Bob",
                  "Charles",
                  "Dany",
                  "Elinor"
              ],
              "num_claims": 5,
              "topics": [
                  [
                      "Cats",
                      {
                          "num_claims": 2,
                          "claims": [
                              {
                                  "claim": "Cats are the best pets.",
                                  "commentId":"c1",
                                  "quote": "I love cats.",
                                  "speaker" : "Alice",
                                  "topicName": "Pets",
                                  "subtopicName": "Cats",
                                  "duplicates": [
                                      {
                                          "claim": "Cats are the best pets.",
                                          "commendId:"c1"
                                          "quote": "I really really love cats",
                                          "speaker" : "Elinor",
                                          "topicName": "Pets",
                                          "subtopicName": "Cats",
                                          "duplicated": true
                                      }
                                  ]
                              }
                          ]
                          "num_speakers" : 2,
                          "speakers" : [
                              "Alice",
                              "Elinor"
                          ]
                      }
                  ],
                  [
                      "Birds",
                      {
                          "num_claims": 2,
                          "claims": [
                              {
                                  "claim": "Birds are not ideal pets for everyone.",
                                  "commentId:"c3",
                                  "quote": "I'm not sure about birds.",
                                  "speaker" : "Charles",
                                  "topicName": "Pets",
                                  "subtopicName": "Birds",
                                  "duplicates": [
                                      {
                                          "claim": "Birds are not suitable pets for everyone.",
                                          "commentId" "c3",
                                          "quote": "I don't know about birds.",
                                          "speaker": "Dany",
                                          "topicName": "Pets",
                                          "subtopicName": "Birds",
                                          "duplicated": true
                                      }
                                  ]
                              }
                          ]
                          "num_speakers" : 2,
                          "speakers" : [
                              "Charles",
                              "Dany"
                          ]
                      }
                  ],
                  [
                      "Dogs",
                      {
                          "num_claims": 1,
                          "claims": [
                              {
                                  "claim": "Dogs are superior pets.",
                                  "commentId": "c2",
                                  "quote": "dogs are great",
                                  "speaker" : "Bob",
                                  "topicName": "Pets",
                                  "subtopicName": "Dogs"
                              }
                          ]
                          "num_speakers" : 1,
                          "speakers" : [
                              "Bob"
                          ]

                      }
                  ]
              ]
          }
      ]
    ]

    For each subtopic, send the contained claims to an LLM to detect near-duplicates.
    These will be returned as dictionaries, where the keys are all the claims for the subtopic,
    numbered with relative ids (claimId0, claimId1, claimId2...claimIdN-1 for N claims), and the
    value for each claim id is a list of the relative claim ids of any near-duplicates.
    Note that this mapping is not guaranteed to be symmetric: claimId0 may have an empty list,
    but claimId1 may have claimId0 and claimId2 in the list. Hence we build a dictionary of
    all the relative ids encountered, and return near duplicates accounting for this asymmetry.

    After deduplication, the full tree of topics, subtopics, and their claims is sorted:
    - more frequent topics appear first
    - within each topic, more frequent subtopics appear first
    - within each subtopic, claims with the most duplicates (ie most supporting quotes) appear first
    Note that currently these duplicates are not counted towards the total claims in a subtopic/topic
    for sorting at the higher levels.

    For now, "near-duplicates" have similar meanings—this is not exact/identical claims and
    we may want to refine this in the future.

    We may also want to allow for other sorting/filtering styles, where the number of duplicates
    DOES matter, or where we want to sum the claims by a particular speaker or by other metadata
    towards the total for a subtopic/topic.
    """
    # skip calling an LLM
    if dry_run or config.DRY_RUN:
       print("dry_run sort tree")
       return config.MOCK_RESPONSE["sort_claims_tree"]
       
    claims_tree = req.tree
    llm = req.llm
    TK_IN = 0
    TK_OUT = 0
    TK_TOT = 0
    dupe_logs = []
    sorted_tree = {}
    
    # Validar estructura de entrada
    if not isinstance(claims_tree, dict):
        print("Warning: Invalid claims_tree structure, using empty tree")
        claims_tree = {}

    for topic, topic_data in claims_tree.items():
        per_topic_total = 0
        per_topic_list = {}
        # consider the empty top-level topic
        if not topic_data["subtopics"]:
            print("NO SUBTOPICS: ", topic)
        for subtopic, subtopic_data in topic_data["subtopics"].items():
            per_topic_total += subtopic_data["total"]
            per_topic_speakers = set()
            # canonical order of claims: as they appear in subtopic_data["claims"]
            # no need to deduplicate single claims
            if subtopic_data["total"] > 1:
                try:
                    response = dedup_claims(subtopic_data["claims"], llm=llm, api_key=x_openai_api_key)
                except Exception:
                    print(
                        "Step 3: no deduped claims response for: ",
                        subtopic_data["claims"],
                    )
                    continue
                deduped = response["dedup_claims"]
                usage = response["usage"]

                # check for duplicates bidirectionally, as we may get either of these scenarios
                # for the same pair of claims:
                # {'nesting': {'claimId0': [], 'claimId1': ['claimId0']}} => {0: [1], 1: [0]}
                # {'nesting': {'claimId0': ['claimId1'], 'claimId1': []}} => {0: [1], 1: [0]}
                # anecdata: recent models may be better about this?

                claim_set = {}
                if "nesting" in deduped:
                    # implementation notes:
                    # - MOST claims should NOT be near-duplicates
                    # - nesting where |claim_vals| > 0 should be a smaller set than |subtopic_data["claims"]|
                    # - but also we won't have duplicate info bidirectionally — A may be dupe of B, but B not dupe of A
                    for claim_key, claim_vals in deduped["nesting"].items():
                        # this claim_key has some duplicates
                        if len(claim_vals) > 0:
                            # extract relative index
                            claim_id = int(claim_key.split("Id")[1])
                            dupe_ids = [
                                int(dupe_claim_key.split("Id")[1])
                                for dupe_claim_key in claim_vals
                            ]
                            # assume duplication is symmetric: add claim_id to dupe_ids, check that each of these maps to the others
                            all_dupes = [claim_id]
                            all_dupes.extend(dupe_ids)
                            for curr_id, dupe in enumerate(all_dupes):
                                other_ids = [
                                    d for i, d in enumerate(all_dupes) if i != curr_id
                                ]
                                if dupe in claim_set:
                                    for other_id in other_ids:
                                        if other_id not in claim_set[dupe]:
                                            claim_set[dupe].append(other_id)
                                else:
                                    claim_set[dupe] = other_ids

                accounted_for_ids = {}
                deduped_claims = []
                # for each claim in our original list
                for claim_id, claim in enumerate(subtopic_data["claims"]):
                    # add speakers of all claims
                    if "speaker" in claim:
                        speaker = claim["speaker"]
                    else:
                        print("no speaker provided:", claim)
                        speaker = "unknown"
                    per_topic_speakers.add(speaker)

                    # only create a new claim if we haven't visited this one already
                    if claim_id not in accounted_for_ids:
                        clean_claim = {k: v for k, v in claim.items()}
                        clean_claim["duplicates"] = []

                        # if this claim has some duplicates
                        if claim_id in claim_set:
                            dupe_ids = claim_set[claim_id]
                            for dupe_id in dupe_ids:
                                if dupe_id not in accounted_for_ids:
                                    # Find the claim by ID in the claims list
                                    dupe_claim = None
                                    for claim_item in subtopic_data["claims"]:
                                        if claim_item.get("claimId") == dupe_id:
                                            dupe_claim = {k: v for k, v in claim_item.items()}
                                            break
                                    
                                    if dupe_claim:
                                        dupe_claim["duplicated"] = True
                                        # add all duplicates as children of main claim
                                        clean_claim["duplicates"].append(dupe_claim)
                                    else:
                                        print(f"Warning: Could not find duplicate claim with ID {dupe_id}")

                                    accounted_for_ids[dupe_id] = 1

                        # add verified claim (may be identical if it has no dupes, except for duplicates: [] field)
                        deduped_claims.append(clean_claim)
                        accounted_for_ids[claim_id] = 1

                # sort so the most duplicated claims are first
                sorted_deduped_claims = sorted(
                    deduped_claims, key=lambda x: len(x["duplicates"]), reverse=True,
                )
                if log_to_wandb:
                    dupe_logs.append(
                        [
                            json.dumps(subtopic_data["claims"], indent=1),
                            json.dumps(sorted_deduped_claims, indent=1),
                        ],
                    )

                TK_TOT += usage.total_tokens
                TK_IN += usage.prompt_tokens
                TK_OUT += usage.completion_tokens
            else:
                sorted_deduped_claims = subtopic_data["claims"]
                # there may be one unique claim or no claims if this is an empty subtopic
                if subtopic_data["claims"]:
                    if "speaker" in subtopic_data["claims"][0]:
                        speaker = subtopic_data["claims"][0]["speaker"]
                    else:
                        print("no speaker provided:", claim)
                        speaker = "unknown"
                    per_topic_speakers.add(speaker)
                else:
                    print("EMPTY SUBTOPIC AFTER CLAIMS: ", subtopic)

            # track how many claims and distinct speakers per subtopic
            tree_counts = {
                "claims": subtopic_data["total"],
                "speakers": len(per_topic_speakers),
            }
            # add list of sorted, deduplicated claims to the right subtopic node in the tree
            per_topic_list[subtopic] = {
                "claims": sorted_deduped_claims,
                "speakers": list(per_topic_speakers),
                "counts": tree_counts,
            }

        # sort all the subtopics in a given topic
        # two ways of sorting 1/16:
        # - (default) numPeople: count the distinct speakers per subtopic/topic
        # - numClaims: count the total claims per subtopic/topic
        set_topic_speakers = set()
        for k, c in per_topic_list.items():
            set_topic_speakers = set_topic_speakers.union(c["speakers"])

        if req.sort == "numPeople":
            sorted_subtopics = sorted(
                per_topic_list.items(),
                key=lambda x: x[1]["counts"]["speakers"],
                reverse=True,
            )
        elif req.sort == "numClaims":
            sorted_subtopics = sorted(
                per_topic_list.items(),
                key=lambda x: x[1]["counts"]["claims"],
                reverse=True,
            )
        # track how many claims and distinct speakers per subtopic
        tree_counts = {"claims": per_topic_total, "speakers": len(set_topic_speakers)}
        # we have to add all the speakers
        sorted_tree[topic] = {
            "topics": sorted_subtopics,
            "speakers": list(set_topic_speakers),
            "counts": tree_counts,
        }

    # sort all the topics in the tree
    if req.sort == "numPeople":
        full_sort_tree = sorted(
            sorted_tree.items(), key=lambda x: x[1]["counts"]["speakers"], reverse=True,
        )
    elif req.sort == "numClaims":
        full_sort_tree = sorted(
            sorted_tree.items(), key=lambda x: x[1]["counts"]["claims"], reverse=True,
        )

    # compute LLM costs for this step's tokens
    s3_total_cost = token_cost(req.llm.model_name, TK_IN, TK_OUT)

    if log_to_wandb:
        try:
            exp_group_name = str(log_to_wandb)
            wandb.init(
                project=config.WANDB_PROJECT_NAME, group=exp_group_name, resume="allow",
            )
            wandb.config.update(
                {
                    "s3_dedup/model": req.llm.model_name,
                    "s3_dedup/user_prompt": req.llm.user_prompt,
                    "s3_dedup/system_prompt": req.llm.system_prompt,
                },
            )

            report_data = [[json.dumps(full_sort_tree, indent=2)]]
            wandb.log(
                {
                    "U_tok_N/dedup": TK_TOT,
                    "U_tok_in/dedup": TK_IN,
                    "U_tok_out/dedup": TK_OUT,
                    "deduped_claims": wandb.Table(
                        data=dupe_logs, columns=["full_flat_claims", "deduped_claims"],
                    ),
                    "t3c_report": wandb.Table(data=report_data, columns=["t3c_report"]),
                    "cost/s3_dedup": s3_total_cost,
                },
            )
            # W&B run completion
            wandb.run.finish()
        except Exception:
            print("Failed to create wandb run")
    net_usage = {
        "total_tokens": TK_TOT,
        "prompt_tokens": TK_IN,
        "completion_tokens": TK_OUT,
    }

    return {"data": full_sort_tree, "usage": net_usage, "cost": s3_total_cost}


###########################################
# Optional / New Feature & Research Steps #
# -----------------------------------------#
# Steps below are optional/exploratory components of the T3C LLM pipeline.


########################################
# Crux claims and controversy analysis #
# --------------------------------------#
# Our first research feature finds "crux claims" to distill the perspectives
# on each subtopic into the core controversy — summary statements on which speakers
# are most evenly split into "agree" or "disagree" sides.
# We prompt an LLM for a crux claim with an explanation, given all the speakers' claims
# on each subtopic (along with the parent topic and a short description). We anonymize
# the claims before sending them to the LLM to protect PII and minimize any potential bias
# based on known speaker identity (e.g. when processing claims made by popular writers)
def controversy_matrix(cont_mat: list) -> list:
    """Compute a controversy matrix from individual speaker opinions on crux claims,
    as predicted by an LLM. For each pair of cruxes, for each speaker:
    # - add 0 only if the speaker agrees with both cruxes
    # - add 0.5 if the speaker has an opinion on one crux, but no known opinion on the other
    # - add 1 if the speaker has a known different opinion on each crux (agree/disagree or disagree/agree)
    # Sum the totals for each pair of cruxes in the corresponding cell in the cross-product
    # and return the matrix of scores.
    """
    cm = [[0 for a in range(len(cont_mat))] for b in range(len(cont_mat))]

    # loop through all the crux statements,
    for claim_index, row in enumerate(cont_mat):
        # these are the scores for each speaker
        per_speaker_scores = row[1:]
        for score_index, score in enumerate(per_speaker_scores):
            # we want this speaker's scores for all statements except current one
            other_scores = [
                item[score_index + 1] for item in cont_mat[claim_index + 1 :]
            ]
            for other_index, other_score in enumerate(other_scores):
                # if the scores match, there is no controversy — do not add anything
                if score != other_score:
                    # we only know one of the opinions
                    if score == 0 or other_score == 0:
                        cm[claim_index][claim_index + other_index + 1] += 0.5
                        cm[claim_index + other_index + 1][claim_index] += 0.5
                    # these opinions are different — max controversy
                    else:
                        cm[claim_index][claim_index + other_index + 1] += 1
                        cm[claim_index + other_index + 1][claim_index] += 1
    return cm


def cruxes_for_topic(
    llm: LLMConfig, topic: str, topic_desc: str, claims: list, speaker_map: dict, api_key: str
) -> dict:
    """For each fully-described subtopic, provide all the relevant claims with an anonymized
    numeric speaker id, and ask the LLM for a crux claim that best splits the speakers' opinions
    on this topic (ideally into two groups of equal size for agreement vs disagreement with the crux claim).
    Requires an explicit API key in api_key.
    """
    # Obtener cliente LLM (OpenAI o Ollama)
    client, actual_model = get_llm_client(api_key, llm.model_name)
    
    claims_anon = []
    speaker_set = set()
    for claim in claims:
        if "speaker" in claim:
            speaker_anon = speaker_map[claim["speaker"]]
            speaker_set.add(speaker_anon)
            speaker_claim = speaker_anon + ":" + claim["claim"]
            claims_anon.append(speaker_claim)

    # TODO: if speaker set is too small / all one person, do not generate cruxes
    if len(speaker_set) < 2:
        print("fewer than 2 speakers: ", topic)
        return None

    full_prompt = llm.user_prompt
    full_prompt += "\nTopic: " + topic + ": " + topic_desc
    full_prompt += "\nParticipant claims: \n" + json.dumps(claims_anon)

    # Para Ollama, modificar prompts para asegurar salida JSON
    system_prompt = llm.system_prompt
    if ollama_config.should_use_ollama():
        # Prompts optimizados para Llama3.2
        system_prompt = "You are a JSON generator. You MUST respond with ONLY valid JSON. No text before or after the JSON."
        full_prompt += f"\n\n<JSON_OUTPUT_REQUIRED>\nAnalyze cruxes and respond with valid JSON in this EXACT format:\n{{\n  \"crux\": {{\n    \"cruxClaim\": \"string\",\n    \"agree\": [\"speaker_list\"],\n    \"disagree\": [\"speaker_list\"],\n    \"explanation\": \"string\"\n  }}\n}}\n</JSON_OUTPUT_REQUIRED>"

    # Preparar argumentos para la llamada
    call_args = {
        "model": actual_model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": full_prompt},
        ],
        "temperature": 0.0,
    }
    
    # Configuraciones específicas según el backend
    if ollama_config.should_use_ollama():
        # Para Ollama: deshabilitar thinking para mayor velocidad
        call_args["think"] = False
    else:
        # Para OpenAI: usar response_format JSON
        call_args["response_format"] = {"type": "json_object"}
    
    response = client.chat.completions.create(**call_args)
    try:
        content = response.choices[0].message.content
        print(f"Raw crux response: {content[:200]}...")  # Log para debug
        
        # Usar la función de extracción mejorada para cruxes también
        crux_obj = extract_json_from_response(content)
        # Verificar estructura básica para crux
        if not isinstance(crux_obj, dict):
            print(f"Unexpected crux structure: type {type(crux_obj)}")
            crux_obj = {"crux": {"cruxClaim": "", "agree": [], "disagree": [], "explanation": ""}}
        print(f"Successfully parsed crux JSON with keys: {list(crux_obj.keys())}")
        
    except Exception as e:
        print("Crux parse error:", str(e))
        crux_obj = {"crux": "Error parsing response"}  # Estructura por defecto para cruxes
        
    return {"crux": crux_obj, "usage": response.usage}


def top_k_cruxes(cont_mat: list, cruxes: list, top_k: int = 0) -> list:
    """Return the top K most controversial crux pairs.
    Optionally let the caller set K, otherwise default
    to the ceiling of the square root of the number of crux claims.
    """
    if top_k == 0:
        K = min(math.ceil(math.sqrt(len(cruxes))), 10)
    else:
        K = top_k
    # let's sort a triangular half of the symmetrical matrix (diagonal is all zeros)
    scores = []
    for x in range(len(cont_mat)):
        for y in range(x + 1, len(cont_mat)):
            scores.append([cont_mat[x][y], x, y])
    all_scored_cruxes = sorted(scores, key=lambda x: x[0], reverse=True)
    top_cruxes = [
        {"score": score, "cruxA": cruxes[x], "cruxB": cruxes[y]}
        for score, x, y in all_scored_cruxes[:K]
    ]
    return top_cruxes


@app.post("/cruxes")
def cruxes_from_tree(
    req: CruxesLLMConfig, x_openai_api_key: str = Header(..., alias="X-OpenAI-API-Key"), log_to_wandb: str = config.WANDB_GROUP_LOG_NAME, dry_run = False,
) -> dict:
    """Given a topic, description, and corresponding list of claims with numerical speaker ids, extract the
    crux claims that would best split the claims into agree/disagree sides.
    Return a crux for each subtopic which contains at least 2 claims and at least 2 speakers.
    """
    if dry_run or config.DRY_RUN:
        print("dry_run cruxes")
        return config.MOCK_RESPONSE["cruxes"]
    cruxes_main = []
    crux_claims = []
    TK_IN = 0
    TK_OUT = 0
    TK_TOT = 0
    topic_desc = topic_desc_map(req.topics)

    # TODO: can we get this from client?
    speaker_map = full_speaker_map(req.crux_tree)
    # print("speaker ids: ", speaker_map)
    for topic, topic_details in req.crux_tree.items():
        subtopics = topic_details["subtopics"]
        for subtopic, subtopic_details in subtopics.items():
            # all claims for subtopic
            # TODO: reduce how many subtopics we analyze for cruxes, based on minimum representation
            # in known speaker comments?
            claims = subtopic_details["claims"]
            if len(claims) < 2:
                print("fewer than 2 claims: ", subtopic)
                continue

            if subtopic in topic_desc:
                subtopic_desc = topic_desc[subtopic]
            else:
                print("no description for subtopic:", subtopic)
                subtopic_desc = "No further details"

            topic_title = topic + ", " + subtopic
            llm_response = cruxes_for_topic(
                req.llm, topic_title, subtopic_desc, claims, speaker_map, x_openai_api_key,
            )
            if not llm_response:
                print("warning: no crux response from LLM")
                continue
            try:
                # Manejar estructura de crux correctamente
                crux_data = llm_response["crux"]
                if isinstance(crux_data, dict) and "crux" in crux_data:
                    # Estructura anidada {"crux": {"crux": {...}}}
                    crux = crux_data["crux"]
                elif isinstance(crux_data, dict) and "cruxClaim" in crux_data:
                    # Estructura directa {"cruxClaim": "...", "agree": [...], ...}
                    crux = crux_data
                else:
                    # Estructura no reconocida
                    print(f"Unexpected crux structure: {crux_data}")
                    continue
                    
                usage = llm_response["usage"]
            except Exception as e:
                print(f"warning: crux response parsing failed: {str(e)}")
                continue

            ids_to_speakers = {v: k for k, v in speaker_map.items()}
            spoken_claims = [c["speaker"] + ": " + c["claim"] for c in claims]

            # create more readable table: crux only, named speakers who agree, named speakers who disagree
            crux_claim = crux["cruxClaim"]
            agree = crux["agree"]
            disagree = crux["disagree"]
            try:
                explanation = crux["explanation"]
            except Exception:
                explanation = "N/A"

            # let's add back the names to the sanitized/speaker-ids-only
            # in the agree/disagree claims
            agree = [a.split(":")[0] for a in agree]
            disagree = [a.split(":")[0] for a in disagree]
            named_agree = [a + ":" + ids_to_speakers[a] for a in agree]
            named_disagree = [d + ":" + ids_to_speakers[d] for d in disagree]
            crux_claims.append([crux_claim, named_agree, named_disagree, explanation])

            # most readable form:
            # - crux claim, explanation, agree, disagree
            # - all claims prepended with speaker names
            # - topic & subctopic, description
            cruxes_main.append(
                [
                    crux_claim,
                    explanation,
                    named_agree,
                    named_disagree,
                    json.dumps(spoken_claims, indent=1),
                    topic_title,
                    subtopic_desc,
                ],
            )

            TK_TOT += usage.total_tokens
            TK_IN += usage.prompt_tokens
            TK_OUT += usage.completion_tokens

    # convert agree/disagree to numeric scores:
    # for each crux claim, for each speaker:
    # - assign 1 if the speaker agrees with the crux
    # - assign 0.5 if the speaker disagrees
    # - assign 0 if the speaker's opinion is unknown/unspecified
    speaker_labels = sorted(speaker_map.keys())
    cont_mat = []
    for row in crux_claims:
        claim_scores = []
        for sl in speaker_labels:
            # associate the numeric id with the speaker so the LLM explanation
            # is more easily interpretable (by cross-referencing adjacent columns which have the
            # full speaker name, which is withheld from the LLM)
            labeled_speaker = speaker_map[sl] + ":" + sl
            if labeled_speaker in row[1]:
                claim_scores.append(1)
            elif labeled_speaker in row[2]:
                claim_scores.append(0.5)
            else:
                claim_scores.append(0)
        cm = [row[0]]
        cm.extend(claim_scores)
        cont_mat.append(cm)
    full_controversy_matrix = controversy_matrix(cont_mat)

    crux_claims_only = [row[0] for row in crux_claims]
    top_cruxes = top_k_cruxes(full_controversy_matrix, crux_claims_only, req.top_k)
    # compute LLM costs for this step's tokens
    s4_total_cost = token_cost(req.llm.model_name, TK_IN, TK_OUT)

    # Note: we will now be sending speaker names to W&B
    # (still not to external LLM providers, to avoid bias on crux detection and better preserve PII)
    if log_to_wandb:
        try:
            exp_group_name = str(log_to_wandb)
            wandb.init(
                project=config.WANDB_PROJECT_NAME, group=exp_group_name, resume="allow",
            )
            wandb.config.update(
                {
                    "s4_cruxes/model": req.llm.model_name,
                    "s4_cruxes/prompt": req.llm.user_prompt,
                },
            )
            log_top_cruxes = [[c["score"], c["cruxA"], c["cruxB"]] for c in top_cruxes]
            wandb.log(
                {
                    "U_tok_N/cruxes": TK_TOT,
                    "U_tok_in/cruxes": TK_IN,
                    "U_tok_out/cruxes": TK_OUT,
                    "cost/s4_cruxes": s4_total_cost,
                    "crux_details": wandb.Table(
                        data=cruxes_main,
                        columns=[
                            "crux",
                            "reason",
                            "agree",
                            "disagree",
                            "original_claims",
                            "topic, subtopic",
                            "description",
                        ],
                    ),
                    "crux_top_scores": wandb.Table(
                        data=log_top_cruxes, columns=["score", "cruxA", "cruxB"],
                    ),
                },
            )
            cols = ["crux"]
            cols.extend(speaker_labels)
            wandb.log(
                {
                    "crux_binary_scores": wandb.Table(data=cont_mat, columns=cols),
                    "crux_cmat_scores": wandb.Table(
                        data=full_controversy_matrix,
                        columns=[
                            "Crux " + str(i)
                            for i in range(len(full_controversy_matrix))
                        ],
                    ),
                    # TODO: render a visual of the controversy matrix
                    # currently matplotlib requires a GUI to generate the plot, which is incompatible with pyserver config
                    # filename = show_confusion_matrix(full_confusion_matrix, claims_only, "Test Conf Mat", "conf_mat_test.jpg")
                    # "cont_mat_img" : wandb.Image(filename)
                },
            )
        except Exception:
            print("Failed to log wandb run")

    # wrap and name fields before returning
    net_usage = {
        "total_tokens": TK_TOT,
        "prompt_tokens": TK_IN,
        "completion_tokens": TK_OUT,
    }
    cruxes = [
        {"cruxClaim": c[0], "agree": c[1], "disagree": c[2], "explanation": c[3]}
        for c in crux_claims
    ]
    crux_response = {
        "cruxClaims": cruxes,
        "controversyMatrix": full_controversy_matrix,
        "topCruxes": top_cruxes,
        "usage": net_usage,
        "cost": s4_total_cost,
    }
    return crux_response


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port, reload=True)
