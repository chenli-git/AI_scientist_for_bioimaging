"""
core/router.py
---------------
Hybrid router that decides which agent should handle a user query
based on content, keywords, or LLM classification.
"""

from typing import Dict
from core.llm_client import get_llm
from agents.AI_scientist_agent import AIScientistAgent

class Router:
    """
    Simple intelligent router for the multi-agent system.
    Routes queries to the most appropriate agent.
    """
    def __init__(self, agent_map: Dict[str, object]):
        """
        agent_map: dictionary of agent_name → agent_class
        e.g. {"scientist": AIScientistAgent, "reviewer": PaperReviewerAgent}
        """
        self.agent_map = agent_map
        self.llm = get_llm(temperature=0.0)
    
    def _rule_based_route(self, query: str) -> str:
        """Basic keyword-based routing."""
        q = query.lower()
        if any(k in q for k in ["paper", "review", "experiment summary", "criticize"]):
            return "reviewer"
        elif any(k in q for k in ["analyze data", "segmentation", "radiomics", "statistical"]):
            return "analyst"
        elif any(k in q for k in ["microscopy", "imaging", "neuron", "astrocyte", "adaptive optics"]):
            return "scientist"
        else:
            return "scientist"  # default fallback

    def _llm_based_route(self, query: str) -> str:
        """Ask an LLM to classify the task."""
        prompt = f"""
        You are a router that decides which AI agent should answer a query.

        Available agents (waiting to implement):
        - scientist: for microscopy, imaging, or biological research questions.
        - reviewer: for paper critique, literature analysis, or summarization.
        - analyst: for quantitative or data analysis questions.

        User query:
        {query}

        Respond with only one word: 'scientist', 'reviewer', or 'analyst'.
        """
        try:
            response = self.llm.invoke(prompt)
            label = response.content.strip().lower()
            if label not in self.agent_map:
                label = self._rule_based_route(query)
            return label
        except Exception:
            return self._rule_based_route(query)
        
    # ------------------------------------------------------------------
    # Main dispatch
    # ------------------------------------------------------------------
    def route_query(self, query: str, use_llm: bool = True):
        """Return an instantiated agent based on the query."""
        label = self._llm_based_route(query) if use_llm else self._rule_based_route(query)
        agent_cls = self.agent_map.get(label)
        if not agent_cls:
            raise ValueError(f"No agent found for route: {label}")
        return agent_cls()

if __name__ == "__main__":
    agent_map = {
        "scientist": AIScientistAgent,
    }
    router = Router(agent_map)
    while True:
            query = input(">>> ").strip()
            if query.lower() in ["exit", "quit"]:
                break
            agent = router.route_query(query, use_llm=True)
            response = agent.run(query)
            print(f"\n🧠 [{agent.__class__.__name__}] → {response}\n")
