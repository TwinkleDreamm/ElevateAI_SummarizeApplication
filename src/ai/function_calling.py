"""
Function calling module for ElevateAI.
Handles Azure OpenAI function calling capabilities.
"""
import json
from typing import List, Dict, Any, Optional, Callable, Union
from dataclasses import dataclass
from datetime import datetime

from config.settings import settings
from src.utils.logger import logger
from src.utils.exceptions import LLMError


@dataclass
class FunctionDefinition:
    """Definition of a callable function."""
    name: str
    description: str
    parameters: Dict[str, Any]
    function: Callable
    required_params: List[str]


class FunctionCaller:
    """Handles function calling for LLM interactions."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the function caller.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.settings = settings
        self.logger = logger
        
        # Registry of available functions
        self.functions: Dict[str, FunctionDefinition] = {}
        
        # Register built-in functions
        self._register_builtin_functions()
    
    def _register_builtin_functions(self) -> None:
        """Register built-in functions."""
        
        # Search function
        self.register_function(
            name="search_documents",
            description="Search through the document database for relevant information",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query to find relevant documents"
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results to return",
                        "default": 5
                    },
                    "threshold": {
                        "type": "number",
                        "description": "Minimum similarity threshold",
                        "default": 0.7
                    }
                },
                "required": ["query"]
            },
            function=self._search_documents
        )
        
        # Get document info function
        self.register_function(
            name="get_document_info",
            description="Get detailed information about a specific document",
            parameters={
                "type": "object",
                "properties": {
                    "document_id": {
                        "type": "string",
                        "description": "ID of the document to get information about"
                    }
                },
                "required": ["document_id"]
            },
            function=self._get_document_info
        )
        
        # Calculate statistics function
        self.register_function(
            name="calculate_statistics",
            description="Calculate basic statistics for a list of numbers",
            parameters={
                "type": "object",
                "properties": {
                    "numbers": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "List of numbers to calculate statistics for"
                    },
                    "include_median": {
                        "type": "boolean",
                        "description": "Whether to include median in calculations",
                        "default": True
                    }
                },
                "required": ["numbers"]
            },
            function=self._calculate_statistics
        )
        
        # Get current time function
        self.register_function(
            name="get_current_time",
            description="Get the current date and time",
            parameters={
                "type": "object",
                "properties": {
                    "format": {
                        "type": "string",
                        "description": "Time format (iso, readable, timestamp)",
                        "default": "readable"
                    }
                },
                "required": []
            },
            function=self._get_current_time
        )
    
    def register_function(self, name: str, description: str, parameters: Dict[str, Any], 
                         function: Callable) -> None:
        """
        Register a new function for calling.
        
        Args:
            name: Function name
            description: Function description
            parameters: JSON schema for parameters
            function: Callable function
        """
        required_params = parameters.get("required", [])
        
        func_def = FunctionDefinition(
            name=name,
            description=description,
            parameters=parameters,
            function=function,
            required_params=required_params
        )
        
        self.functions[name] = func_def
        self.logger.info(f"Registered function: {name}")
    
    def get_function_definitions(self) -> List[Dict[str, Any]]:
        """
        Get function definitions in OpenAI format.
        
        Returns:
            List of function definitions
        """
        definitions = []
        
        for func_def in self.functions.values():
            definition = {
                "name": func_def.name,
                "description": func_def.description,
                "parameters": func_def.parameters
            }
            definitions.append(definition)
        
        return definitions
    
    def call_function(self, function_name: str, arguments: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Call a registered function.
        
        Args:
            function_name: Name of function to call
            arguments: Function arguments (JSON string or dict)
            
        Returns:
            Function result
            
        Raises:
            LLMError: If function call fails
        """
        if function_name not in self.functions:
            raise LLMError(f"Unknown function: {function_name}")
        
        func_def = self.functions[function_name]
        
        try:
            # Parse arguments if string
            if isinstance(arguments, str):
                args = json.loads(arguments)
            else:
                args = arguments
            
            # Validate required parameters
            for param in func_def.required_params:
                if param not in args:
                    raise LLMError(f"Missing required parameter: {param}")
            
            # Call function
            self.logger.info(f"Calling function: {function_name} with args: {args}")
            result = func_def.function(**args)
            
            return {
                "success": True,
                "result": result,
                "function_name": function_name
            }
            
        except json.JSONDecodeError as e:
            raise LLMError(f"Invalid JSON arguments: {e}")
        except Exception as e:
            self.logger.error(f"Function call failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "function_name": function_name
            }
    
    def process_function_calls(self, function_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process multiple function calls.
        
        Args:
            function_calls: List of function calls
            
        Returns:
            List of function results
        """
        results = []
        
        for call in function_calls:
            function_name = call.get('name')
            arguments = call.get('arguments')
            
            if not function_name:
                results.append({
                    "success": False,
                    "error": "Missing function name"
                })
                continue
            
            result = self.call_function(function_name, arguments)
            results.append(result)
        
        return results
    
    # Built-in function implementations
    
    def _search_documents(self, query: str, max_results: int = 5, threshold: float = 0.7) -> Dict[str, Any]:
        """Search documents function implementation."""
        try:
            # This would integrate with the actual search engine
            # For now, return a mock response
            return {
                "query": query,
                "results": [
                    {
                        "id": "doc_1",
                        "title": "Sample Document",
                        "content": "This is a sample document that matches your query.",
                        "score": 0.85
                    }
                ],
                "total_results": 1,
                "search_time": "0.1s"
            }
        except Exception as e:
            return {"error": f"Search failed: {e}"}
    
    def _get_document_info(self, document_id: str) -> Dict[str, Any]:
        """Get document info function implementation."""
        try:
            # This would integrate with the actual document database
            return {
                "document_id": document_id,
                "title": "Sample Document",
                "type": "text",
                "size": "1.2 MB",
                "created_at": "2024-01-01T00:00:00Z",
                "word_count": 1500,
                "language": "en"
            }
        except Exception as e:
            return {"error": f"Failed to get document info: {e}"}
    
    def _calculate_statistics(self, numbers: List[float], include_median: bool = True) -> Dict[str, Any]:
        """Calculate statistics function implementation."""
        try:
            if not numbers:
                return {"error": "Empty number list"}
            
            import statistics
            
            stats = {
                "count": len(numbers),
                "sum": sum(numbers),
                "mean": statistics.mean(numbers),
                "min": min(numbers),
                "max": max(numbers)
            }
            
            if len(numbers) > 1:
                stats["stdev"] = statistics.stdev(numbers)
            
            if include_median:
                stats["median"] = statistics.median(numbers)
            
            return stats
            
        except Exception as e:
            return {"error": f"Statistics calculation failed: {e}"}
    
    def _get_current_time(self, format: str = "readable") -> Dict[str, Any]:
        """Get current time function implementation."""
        try:
            now = datetime.now()
            
            if format == "iso":
                time_str = now.isoformat()
            elif format == "timestamp":
                time_str = str(int(now.timestamp()))
            else:  # readable
                time_str = now.strftime("%Y-%m-%d %H:%M:%S")
            
            return {
                "current_time": time_str,
                "format": format,
                "timezone": "local"
            }
            
        except Exception as e:
            return {"error": f"Failed to get current time: {e}"}
    
    def create_function_call_message(self, function_results: List[Dict[str, Any]]) -> Dict[str, str]:
        """
        Create a message with function call results.
        
        Args:
            function_results: Results from function calls
            
        Returns:
            Message dictionary
        """
        results_text = []
        
        for result in function_results:
            if result.get("success"):
                function_name = result.get("function_name", "unknown")
                function_result = result.get("result", {})
                results_text.append(f"Function '{function_name}' returned: {json.dumps(function_result, indent=2)}")
            else:
                function_name = result.get("function_name", "unknown")
                error = result.get("error", "Unknown error")
                results_text.append(f"Function '{function_name}' failed: {error}")
        
        return {
            "role": "function",
            "content": "\n\n".join(results_text)
        }
    
    def get_available_functions(self) -> List[str]:
        """
        Get list of available function names.
        
        Returns:
            List of function names
        """
        return list(self.functions.keys())
    
    def get_function_help(self, function_name: str) -> Optional[Dict[str, Any]]:
        """
        Get help information for a function.
        
        Args:
            function_name: Name of function
            
        Returns:
            Function help information or None
        """
        if function_name not in self.functions:
            return None
        
        func_def = self.functions[function_name]
        
        return {
            "name": func_def.name,
            "description": func_def.description,
            "parameters": func_def.parameters,
            "required_parameters": func_def.required_params
        }
