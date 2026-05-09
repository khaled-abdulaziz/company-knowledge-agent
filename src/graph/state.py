# ==============================================================
# LangGraph Agent State
# ==============================================================

from typing import TypedDict, List, Optional
from llama_index.core.schema import NodeWithScore


class AgentState(TypedDict):
    """
    Shared state passed between all LangGraph nodes.

    Flow:
        question → router → (docs_node OR sql_node) → answer

    Each field is updated by the relevant node.
    """

    
    # INPUT — set by the user
    
    question: str

    
    # ROUTING — set by router_node

    route: str

    is_sensitive: bool


    # CONTEXT — set by docs_node or sql_node
    
    retrieved_nodes: List[NodeWithScore]

    sql_results: Optional[List[dict]]


  
    # OUTPUT — set by the final answering node
    answer: str
   
    llm_used: str
  