"""
JSON Schemas for Structured Outputs
Supports both OpenAI and OpenRouter structured outputs
"""

def get_json_schemas():
    """Define JSON schemas for structured outputs"""
    
    # Schema for topic tree (taxonomy)
    topic_tree_schema = {
        "name": "topic_tree",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "taxonomy": {
                    "type": "array",
                    "description": "Array of topics with their subtopics",
                    "items": {
                        "type": "object",
                        "properties": {
                            "topicName": {
                                "type": "string",
                                "description": "Name of the main topic"
                            },
                            "topicShortDescription": {
                                "type": "string", 
                                "description": "Brief description of the topic"
                            },
                            "subtopics": {
                                "type": "array",
                                "description": "Array of subtopics under this topic",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "subtopicName": {
                                            "type": "string",
                                            "description": "Name of the subtopic"
                                        },
                                        "subtopicShortDescription": {
                                            "type": "string",
                                            "description": "Brief description of the subtopic"
                                        }
                                    },
                                    "required": ["subtopicName", "subtopicShortDescription"],
                                    "additionalProperties": False
                                }
                            }
                        },
                        "required": ["topicName", "topicShortDescription", "subtopics"],
                        "additionalProperties": False
                    }
                }
            },
            "required": ["taxonomy"],
            "additionalProperties": False
        }
    }
    
    # Schema for claims extraction
    claims_schema = {
        "name": "claims",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "claims": {
                    "type": "array",
                    "description": "Array of extracted claims from the comment",
                    "items": {
                        "type": "object",
                        "properties": {
                            "claim": {
                                "type": "string",
                                "description": "The extracted claim statement"
                            },
                            "quote": {
                                "type": "string",
                                "description": "The relevant quote from the original comment"
                            },
                            "topicName": {
                                "type": "string",
                                "description": "The topic this claim belongs to"
                            },
                            "subtopicName": {
                                "type": "string",
                                "description": "The subtopic this claim belongs to"
                            }
                        },
                        "required": ["claim", "quote", "topicName", "subtopicName"],
                        "additionalProperties": False
                    }
                }
            },
            "required": ["claims"],
            "additionalProperties": False
        }
    }
    
    # Schema for deduplication results
    dedup_schema = {
        "name": "deduplication",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "duplicates": {
                    "type": "array",
                    "description": "Array of claim IDs that are duplicates of other claims",
                    "items": {
                        "type": "object",
                        "properties": {
                            "claimId": {
                                "type": "integer",
                                "description": "ID of the claim that is a duplicate"
                            },
                            "parentClaimId": {
                                "type": "integer",
                                "description": "ID of the parent claim this is a duplicate of"
                            },
                            "similarity": {
                                "type": "string",
                                "description": "Description of why these claims are similar"
                            }
                        },
                        "required": ["claimId", "parentClaimId", "similarity"],
                        "additionalProperties": False
                    }
                }
            },
            "required": ["duplicates"],
            "additionalProperties": False
        }
    }
    
    # Schema for crux generation
    crux_schema = {
        "name": "crux",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "crux": {
                    "type": "object",
                    "description": "A controversial statement designed to divide participants",
                    "properties": {
                        "cruxClaim": {
                            "type": "string",
                            "description": "The controversial claim statement"
                        },
                        "agree": {
                            "type": "array",
                            "description": "List of participant IDs who would agree with the crux",
                            "items": {
                                "type": "string",
                                "description": "Participant identifier"
                            }
                        },
                        "disagree": {
                            "type": "array", 
                            "description": "List of participant IDs who would disagree with the crux",
                            "items": {
                                "type": "string",
                                "description": "Participant identifier"
                            }
                        },
                        "explanation": {
                            "type": "string",
                            "description": "Explanation of why participants are divided this way"
                        }
                    },
                    "required": ["cruxClaim", "agree", "disagree", "explanation"],
                    "additionalProperties": False
                }
            },
            "required": ["crux"],
            "additionalProperties": False
        }
    }
    
    return {
        "topic_tree": topic_tree_schema,
        "claims": claims_schema,
        "dedup": dedup_schema,
        "crux": crux_schema
    }

def get_structured_response_format(schema_type: str, use_structured_outputs: bool = True):
    """
    Get the response_format parameter for structured outputs
    
    Args:
        schema_type: Type of schema ('topic_tree', 'claims', 'dedup')
        use_structured_outputs: Whether to use structured outputs (True) or fallback to json_object (False)
    
    Returns:
        dict: Response format configuration
    """
    if not use_structured_outputs:
        # Fallback to basic JSON object for compatibility
        return {"type": "json_object"}
    
    schemas = get_json_schemas()
    if schema_type not in schemas:
        # Fallback to json_object for unsupported schema types
        return {"type": "json_object"}
    
    return {
        "type": "json_schema",
        "json_schema": schemas[schema_type]
    } 