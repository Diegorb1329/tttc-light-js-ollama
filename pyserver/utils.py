#! /usr/bin env python
import json
import wandb
from openai import OpenAI

import config

def get_openai_client(api_key: str):
    """
    Initializes and returns an OpenAI client.
    If OPENAI_API_BASE_URL is set in the config, it configures the client
    to use the specified base URL (e.g., for OpenRouter) and uses
    OPENROUTER_API_KEY for authentication. Otherwise, it uses the default
    OpenAI configuration with the provided api_key.
    """
    if config.OPENAI_API_BASE_URL:
        if not config.OPENROUTER_API_KEY:
            raise ValueError("OPENROUTER_API_KEY must be set when using a custom base_url")
        return OpenAI(
            api_key=config.OPENROUTER_API_KEY,
            base_url=config.OPENAI_API_BASE_URL,
        )
    return OpenAI(api_key=api_key)

def build_extra_body():
    """
    Constructs the extra_body dictionary for OpenRouter specific features
    based on environment variables.
    """
    extra_body = {}
    if config.OPENROUTER_MODELS:
        extra_body["models"] = [model.strip() for model in config.OPENROUTER_MODELS.split(',')]
    if config.OPENROUTER_TRANSFORMS:
        extra_body["transforms"] = [transform.strip() for transform in config.OPENROUTER_TRANSFORMS.split(',')]
    return extra_body if extra_body else None


def comment_is_meaningful(raw_comment:str):
  """ Check whether the raw comment contains enough words/characters
  to be meaningful in web app mode. Only check word count for short comments.
  TODO: add config for other modes like elicitation/direct response
  """
  if len(raw_comment) >= config.MIN_CHAR_COUNT_FOR_MEANING or len(raw_comment.split(" ")) >= config.MIN_WORD_COUNT_FOR_MEANING:
    return True
  else:
    return False


def token_cost(model_name:str, tok_in:int, tok_out:int):
  """ Returns the cost for the current model running the given numbers of
  tokens in/out for this call """
  if model_name not in config.COST_BY_MODEL:
    print(f"Warning: Model '{model_name}' not found in COST_BY_MODEL. Cost calculation may be inaccurate.")
    # Fallback to a default or zero cost if model is not in the dictionary
    return 0.0
  return 0.001 * (tok_in  *  config.COST_BY_MODEL[model_name]["in_per_1K"] + tok_out * config.COST_BY_MODEL[model_name]["out_per_1K"])

def cute_print(json_obj):
  """Returns a pretty version of a dictionary as properly-indented and scaled
  json in html for at-a-glance review in W&B"""
  str_json = json.dumps(json_obj, indent=1)
  cute_html = '<pre id="json"><font size=2>' + str_json + "</font></pre>"
  return wandb.Html(cute_html)

def topic_desc_map(topics:list)->dict:
  """ Convert a list of topics into a dictionary returning the short description for 
      each topic name. Note this currently assumes we have no duplicate topic/subtopic
      names, which ideally we shouldn't :)
  """
  topic_desc = {}
  for topic in topics:
    topic_desc[topic["topicName"]] = topic["topicShortDescription"]
    if "subtopics" in topic:
      for subtopic in topic["subtopics"]:
        topic_desc[subtopic["subtopicName"]] = subtopic["subtopicShortDescription"]
  return topic_desc

def full_speaker_map(tree:dict):
  """ Given a full topic tree, collect all distinct speakers for all claims into one set,
  sort alphabetically, then enumerate (so the numerical id of the speaker is deterministic
  from the composition of any particular dataset """
  speakers = set()
  for topic, topic_details in tree.items():
    for subtopic, subtopic_details in topic_details["subtopics"].items():
      # all claims for subtopic
      claims = subtopic_details["claims"]
      for claim in claims:
        speakers.add(claim["speaker"])
  speaker_list = list(speakers)
  speaker_list.sort()
  speaker_map = {}
  for i, s in enumerate(speaker_list):
    speaker_map[s] = str(i)
  return speaker_map


def show_confusion_matrix(cm, class_names, title, filename):
  """Returns a matplotlib plot of the confusion matrix for W&B.
  cm: a matrix of scores
  class_names: the names of the classes for the x and y axes
  """
  import matplotlib.pyplot as plt
  import numpy as np

  # Normalize the confusion matrix.
  # cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

  figure = plt.figure(figsize=(8, 8))
  plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Wistia)
  plt.title(title)
  plt.colorbar()
  tick_marks = np.arange(len(class_names))
  plt.xticks(tick_marks, class_names, rotation=45)
  plt.yticks(tick_marks, class_names)

  # Use white text if squares are dark; otherwise black.
  threshold = cm.max() / 2.
  
  for i, j in product(range(cm.shape[0]), range(cm.shape[1])):
    color = "white" if cm[i, j] < threshold else "black"
    plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)
    
  plt.tight_layout()
  plt.ylabel('True label')
  plt.xlabel('Predicted label')
  plt.savefig(filename)
  return filename