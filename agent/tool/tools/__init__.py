"""
Specific tool implementations
"""

from agent.tool.tools.search_tool import SearchTool
from agent.tool.tools.calculator_tool import CalculatorTool
from agent.tool.tools.wiki_search_tool import WikiSearchTool
__all__ = [
    'SearchTool',
    'CalculatorTool',
    'WikiSearchTool',
] 

def _default_tools(env, working_dir="expr/2WikiMultiHopQA", embedding_model=None):
    if env == 'search':
        return [SearchTool()]
    elif env == 'fedsearch':
        from fedgraphr1.client.federated_search_tool import FederatedSearchTool
        if embedding_model is None:
            try:
                from FlagEmbedding import FlagAutoModel
                embedding_model = FlagAutoModel.from_finetuned(
                    'BAAI/bge-large-en-v1.5',
                    query_instruction_for_retrieval=(
                        "Represent this sentence for searching relevant passages: "
                    ),
                )
            except ImportError:
                pass  # embedding_model stays None; tool will warn on first query
        tool = FederatedSearchTool(working_dir=working_dir, embedding_model=embedding_model)
        tool.load()  # load pre-built Stage 1 FAISS indices from working_dir
        return [tool]
    elif env == 'calculator':
        return [CalculatorTool()]
    elif env == 'wikisearch':
        return [WikiSearchTool()]
    else:
        raise NotImplementedError
