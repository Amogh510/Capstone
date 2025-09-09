"""
Python Client for Knowledge Graph Retrieval Service
Easy-to-use client for LLM integration
"""

import requests
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import json

@dataclass
class RetrievalClient:
    """Client for the KG Retrieval Service"""
    base_url: str = "http://localhost:8000"
    timeout: int = 30
    
    def search(
        self,
        query: str,
        search_type: str = "fulltext",
        max_results: int = 10,
        expand_depth: int = 1,
        file_filter: Optional[str] = None,
        component_filter: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Main search interface
        
        Args:
            query: Search query string
            search_type: One of 'fulltext', 'component', 'file', 'semantic'
            max_results: Maximum number of results to return
            expand_depth: How many relationship hops to expand (0-3)
            file_filter: Filter results to specific file path
            component_filter: Filter results to specific component
            
        Returns:
            Dictionary with search results and LLM-formatted context
        """
        payload = {
            "query": query,
            "search_type": search_type,
            "max_results": max_results,
            "expand_depth": expand_depth,
            "include_code": False
        }
        
        if file_filter:
            payload["file_filter"] = file_filter
        if component_filter:
            payload["component_filter"] = component_filter
            
        response = requests.post(
            f"{self.base_url}/search",
            json=payload,
            timeout=self.timeout
        )
        response.raise_for_status()
        return response.json()
    
    def get_llm_context(
        self,
        query: str,
        search_type: str = "fulltext",
        max_results: int = 10,
        expand_depth: int = 1
    ) -> str:
        """
        Get LLM-ready context string for a query
        
        Returns:
            Formatted string ready to be used as LLM context
        """
        result = self.search(
            query=query,
            search_type=search_type,
            max_results=max_results,
            expand_depth=expand_depth
        )
        return result.get("llm_formatted_context", "")
    
    def search_component(
        self,
        component_name: str,
        expand_depth: int = 1,
        max_results: int = 20
    ) -> Dict[str, Any]:
        """Search for a specific component and its context"""
        response = requests.get(
            f"{self.base_url}/search/component/{component_name}",
            params={
                "expand_depth": expand_depth,
                "max_results": max_results
            },
            timeout=self.timeout
        )
        response.raise_for_status()
        return response.json()
    
    def search_file(
        self,
        file_path: str,
        max_results: int = 50
    ) -> Dict[str, Any]:
        """Get all context from a specific file"""
        response = requests.get(
            f"{self.base_url}/search/file",
            params={
                "file_path": file_path,
                "max_results": max_results
            },
            timeout=self.timeout
        )
        response.raise_for_status()
        return response.json()
    
    def search_fulltext(
        self,
        query: str,
        max_results: int = 10,
        expand_depth: int = 0
    ) -> Dict[str, Any]:
        """Simple fulltext search"""
        response = requests.get(
            f"{self.base_url}/search/fulltext",
            params={
                "q": query,
                "max_results": max_results,
                "expand_depth": expand_depth
            },
            timeout=self.timeout
        )
        response.raise_for_status()
        return response.json()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        response = requests.get(f"{self.base_url}/stats", timeout=self.timeout)
        response.raise_for_status()
        return response.json()
    
    def health_check(self) -> bool:
        """Check if the service is healthy"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200
        except:
            return False

# LLM Integration Helpers
class LLMContextBuilder:
    """Helper class for building LLM contexts with KG data"""
    
    def __init__(self, client: RetrievalClient):
        self.client = client
    
    def build_context_for_question(
        self,
        user_question: str,
        max_context_length: int = 6000
    ) -> str:
        """
        Build context for a user question about the codebase
        
        Args:
            user_question: The user's question
            max_context_length: Maximum length of context
            
        Returns:
            Formatted context string
        """
        # Try different search strategies and combine results
        contexts = []
        
        # 1. Fulltext search for direct matches
        fulltext_result = self.client.search(
            query=user_question,
            search_type="fulltext",
            max_results=5,
            expand_depth=1
        )
        if fulltext_result.get("llm_formatted_context"):
            contexts.append(fulltext_result["llm_formatted_context"])
        
        # 2. If question mentions specific component, search for it
        # Simple heuristic: look for capitalized words that might be components
        words = user_question.split()
        for word in words:
            if word[0].isupper() and len(word) > 2:
                try:
                    comp_result = self.client.search_component(word, expand_depth=1)
                    if comp_result.get("llm_formatted_context"):
                        contexts.append(comp_result["llm_formatted_context"])
                        break  # Only add one component context
                except:
                    continue
        
        # Combine contexts
        combined_context = "\n\n".join(contexts)
        
        # Truncate if too long
        if len(combined_context) > max_context_length:
            combined_context = combined_context[:max_context_length]
            last_section = combined_context.rfind("\n## ")
            if last_section > 0:
                combined_context = combined_context[:last_section] + "\n\n[Context truncated]"
        
        return combined_context
    
    def build_context_for_file_question(
        self,
        file_path: str,
        question: str = ""
    ) -> str:
        """Build context focused on a specific file"""
        result = self.client.search_file(file_path)
        return result.get("llm_formatted_context", "")
    
    def build_context_for_component_question(
        self,
        component_name: str,
        question: str = "",
        include_related: bool = True
    ) -> str:
        """Build context focused on a specific component"""
        expand_depth = 2 if include_related else 0
        result = self.client.search_component(component_name, expand_depth=expand_depth)
        return result.get("llm_formatted_context", "")

# Example usage functions
def example_usage():
    """Example of how to use the retrieval client"""
    
    # Initialize client
    client = RetrievalClient()
    
    # Check if service is running
    if not client.health_check():
        print("❌ Retrieval service is not running!")
        print("Start it with: python main.py")
        return
    
    print("✅ Retrieval service is running")
    
    # Get database stats
    stats = client.get_stats()
    print(f"📊 Database contains:")
    for stat in stats["node_statistics"][:5]:
        print(f"   {stat['node_type']}: {stat['count']} nodes")
    
    # Example searches
    print("\n🔍 Example searches:")
    
    # 1. Search for Alert components
    print("\n1. Searching for Alert components...")
    result = client.search_component("Alert", expand_depth=1)
    print(f"   Found {len(result['results'])} nodes")
    
    # 2. Search for specific file
    print("\n2. Searching in Alert files...")
    result = client.search_fulltext("Alert", max_results=5)
    print(f"   Found {len(result['results'])} nodes")
    
    # 3. Get LLM context for a question
    print("\n3. Getting LLM context for 'How does the Alert component work?'")
    llm_builder = LLMContextBuilder(client)
    context = llm_builder.build_context_for_question("How does the Alert component work?")
    print(f"   Generated {len(context)} characters of context")
    print(f"   Preview: {context[:200]}...")

if __name__ == "__main__":
    example_usage()
