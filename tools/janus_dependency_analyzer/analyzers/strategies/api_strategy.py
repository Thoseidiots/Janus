"""
API endpoint analysis strategy.

This strategy scans for REST API endpoints, GraphQL schemas, and RPC interfaces
to identify programmatic capabilities.
"""

import logging
import re
import json
from typing import List, Optional, Dict, Any
from pathlib import Path
import subprocess

from .base_strategy import BaseAnalysisStrategy
from ...core.models import Application, Capability, CapabilityCategory, InterfaceType, Parameter


logger = logging.getLogger(__name__)


class APIEndpointStrategy(BaseAnalysisStrategy):
    """
    Strategy for analyzing API endpoints to extract capabilities.
    
    This strategy looks for:
    - REST API endpoints
    - GraphQL schemas
    - RPC interfaces
    - OpenAPI/Swagger documentation
    - Configuration files with API definitions
    """
    
    def __init__(self):
        """Initialize the API endpoint analysis strategy."""
        super().__init__("api_endpoint_analysis", confidence_factor=0.9)
        
        # Common API-related file patterns
        self.api_file_patterns = [
            "*.json",
            "*.yaml", 
            "*.yml",
            "swagger.*",
            "openapi.*",
            "api.*",
            "schema.*",
            "*.graphql",
            "*.gql"
        ]
        
        # API-related directories
        self.api_directories = [
            "api",
            "apis", 
            "swagger",
            "openapi",
            "schema",
            "schemas",
            "graphql",
            "docs",
            "documentation"
        ]
        
        # Common API ports to check
        self.common_api_ports = [
            3000, 8000, 8080, 8081, 8090, 9000, 9090,
            5000, 5001, 4000, 3001, 7000, 8888
        ]
    
    def can_analyze(self, app: Application) -> bool:
        """
        Check if this strategy can analyze the given application.
        
        Args:
            app: Application to check
            
        Returns:
            bool: True if API-related files or indicators are found
        """
        if not app.is_accessible:
            return False
        
        # Check for API-related files
        api_files = self._find_api_files(app.installation_path)
        if api_files:
            return True
        
        # Check if the application might be a web server/API server
        if self._appears_to_be_api_server(app):
            return True
        
        return False
    
    def extract_capabilities(self, app: Application) -> List[Capability]:
        """
        Extract capabilities from API endpoint analysis.
        
        Args:
            app: Application to analyze
            
        Returns:
            List[Capability]: Extracted capabilities
        """
        capabilities = []
        
        # Analyze API documentation files
        api_files = self._find_api_files(app.installation_path)
        for api_file in api_files:
            file_capabilities = self._analyze_api_file(app, api_file)
            capabilities.extend(file_capabilities)
        
        # Try to detect running API endpoints
        if self._appears_to_be_api_server(app):
            runtime_capabilities = self._analyze_runtime_api(app)
            capabilities.extend(runtime_capabilities)
        
        # Analyze configuration files for API settings
        config_capabilities = self._analyze_api_configuration(app)
        capabilities.extend(config_capabilities)
        
        return capabilities
    
    def _find_api_files(self, installation_path: Path) -> List[Path]:
        """
        Find API-related files in the installation directory.
        
        Args:
            installation_path: Path to search
            
        Returns:
            List[Path]: Found API files
        """
        api_files = []
        
        if not installation_path.exists():
            return api_files
        
        try:
            # Search in main directory
            for pattern in self.api_file_patterns:
                api_files.extend(installation_path.glob(pattern))
            
            # Search in API-related subdirectories
            for api_dir_name in self.api_directories:
                api_dir = installation_path / api_dir_name
                if api_dir.exists() and api_dir.is_dir():
                    for pattern in self.api_file_patterns:
                        api_files.extend(api_dir.glob(pattern))
            
            # Filter to only include files that look like API definitions
            api_files = [f for f in api_files if f.is_file() and self._is_api_file(f)]
            
        except (OSError, PermissionError) as e:
            self.logger.debug(f"Error searching for API files in {installation_path}: {e}")
        
        return api_files[:20]  # Limit to prevent excessive processing
    
    def _is_api_file(self, file_path: Path) -> bool:
        """
        Check if a file is likely an API definition file.
        
        Args:
            file_path: Path to check
            
        Returns:
            bool: True if the file appears to be API-related
        """
        name_lower = file_path.name.lower()
        
        # Check for API-related names
        api_indicators = [
            'api', 'swagger', 'openapi', 'schema', 'graphql',
            'endpoints', 'routes', 'service', 'spec'
        ]
        
        if any(indicator in name_lower for indicator in api_indicators):
            return True
        
        # Check file content for API indicators
        content = self._safe_read_file(file_path, max_size=1024 * 100)  # 100KB limit
        if content:
            content_lower = content.lower()
            api_content_indicators = [
                'swagger', 'openapi', 'paths:', 'endpoints',
                'graphql', 'query', 'mutation', 'subscription',
                'rest', 'api', '"get":', '"post":', '"put":', '"delete":'
            ]
            
            return any(indicator in content_lower for indicator in api_content_indicators)
        
        return False
    
    def _analyze_api_file(self, app: Application, api_file: Path) -> List[Capability]:
        """
        Analyze a single API file to extract capabilities.
        
        Args:
            app: Application being analyzed
            api_file: API file to analyze
            
        Returns:
            List[Capability]: Capabilities extracted from the file
        """
        capabilities = []
        
        content = self._safe_read_file(api_file)
        if not content:
            return capabilities
        
        # Determine file type and analyze accordingly
        if api_file.suffix.lower() in ['.json', '.yaml', '.yml']:
            capabilities.extend(self._analyze_openapi_spec(app, content, api_file))
            capabilities.extend(self._analyze_json_api_config(app, content, api_file))
        elif api_file.suffix.lower() in ['.graphql', '.gql']:
            capabilities.extend(self._analyze_graphql_schema(app, content, api_file))
        else:
            # Generic text-based API analysis
            capabilities.extend(self._analyze_generic_api_file(app, content, api_file))
        
        return capabilities
    
    def _analyze_openapi_spec(self, app: Application, content: str, file_path: Path) -> List[Capability]:
        """
        Analyze OpenAPI/Swagger specification.
        
        Args:
            app: Application being analyzed
            content: File content
            file_path: Path to the file
            
        Returns:
            List[Capability]: OpenAPI-based capabilities
        """
        capabilities = []
        
        try:
            # Try to parse as JSON first, then YAML
            spec_data = None
            try:
                spec_data = json.loads(content)
            except json.JSONDecodeError:
                try:
                    import yaml
                    spec_data = yaml.safe_load(content)
                except (ImportError, yaml.YAMLError):
                    pass
            
            if not spec_data or not isinstance(spec_data, dict):
                return capabilities
            
            # Check if it's an OpenAPI/Swagger spec
            if 'swagger' in spec_data or 'openapi' in spec_data:
                capabilities.extend(self._extract_openapi_capabilities(app, spec_data))
            
        except Exception as e:
            self.logger.debug(f"Error parsing OpenAPI spec {file_path}: {e}")
        
        return capabilities
    
    def _extract_openapi_capabilities(self, app: Application, spec_data: Dict[str, Any]) -> List[Capability]:
        """
        Extract capabilities from OpenAPI specification data.
        
        Args:
            app: Application being analyzed
            spec_data: Parsed OpenAPI specification
            
        Returns:
            List[Capability]: Extracted capabilities
        """
        capabilities = []
        
        # Extract API info
        info = spec_data.get('info', {})
        api_title = info.get('title', 'API')
        api_description = info.get('description', '')
        
        # Extract paths and operations
        paths = spec_data.get('paths', {})
        for path, path_data in paths.items():
            if isinstance(path_data, dict):
                for method, operation_data in path_data.items():
                    if method.lower() in ['get', 'post', 'put', 'delete', 'patch', 'options', 'head']:
                        capability = self._create_api_endpoint_capability(
                            app, method.upper(), path, operation_data
                        )
                        capabilities.append(capability)
        
        # Create a general API capability
        if capabilities:
            general_capability = self._create_capability(
                app=app,
                name=f"{api_title} API",
                category=CapabilityCategory.NETWORK_OPERATIONS,
                interface_type=InterfaceType.REST_API,
                description=api_description or f"REST API with {len(capabilities)} endpoints",
                confidence=0.9
            )
            capabilities.insert(0, general_capability)
        
        return capabilities
    
    def _create_api_endpoint_capability(
        self, 
        app: Application, 
        method: str, 
        path: str, 
        operation_data: Dict[str, Any]
    ) -> Capability:
        """
        Create a capability for an API endpoint.
        
        Args:
            app: Application being analyzed
            method: HTTP method
            path: API path
            operation_data: Operation specification data
            
        Returns:
            Capability: API endpoint capability
        """
        operation_id = operation_data.get('operationId', f"{method}_{path.replace('/', '_')}")
        summary = operation_data.get('summary', f"{method} {path}")
        description = operation_data.get('description', summary)
        
        # Extract parameters
        parameters = []
        for param_data in operation_data.get('parameters', []):
            if isinstance(param_data, dict):
                param = Parameter(
                    name=param_data.get('name', ''),
                    type=param_data.get('type', 'string'),
                    description=param_data.get('description', ''),
                    required=param_data.get('required', False)
                )
                parameters.append(param)
        
        return self._create_capability(
            app=app,
            name=f"{method} {path}",
            category=CapabilityCategory.NETWORK_OPERATIONS,
            interface_type=InterfaceType.REST_API,
            description=description,
            confidence=0.9,
            parameters=parameters,
            examples=[f"{method} {path}"]
        )
    
    def _analyze_graphql_schema(self, app: Application, content: str, file_path: Path) -> List[Capability]:
        """
        Analyze GraphQL schema file.
        
        Args:
            app: Application being analyzed
            content: Schema content
            file_path: Path to the file
            
        Returns:
            List[Capability]: GraphQL-based capabilities
        """
        capabilities = []
        
        # Extract GraphQL types and operations
        type_matches = re.findall(r'type\s+(\w+)\s*{([^}]+)}', content, re.MULTILINE)
        for type_name, type_body in type_matches:
            if type_name in ['Query', 'Mutation', 'Subscription']:
                # Extract operations from Query/Mutation/Subscription types
                operations = re.findall(r'(\w+)\s*(?:\([^)]*\))?\s*:\s*([^,\n]+)', type_body)
                for op_name, return_type in operations:
                    capability = self._create_capability(
                        app=app,
                        name=f"GraphQL {type_name}: {op_name}",
                        category=CapabilityCategory.NETWORK_OPERATIONS,
                        interface_type=InterfaceType.GRAPHQL_API,
                        description=f"{type_name} operation returning {return_type.strip()}",
                        confidence=0.8
                    )
                    capabilities.append(capability)
        
        # Create general GraphQL capability
        if capabilities:
            general_capability = self._create_capability(
                app=app,
                name="GraphQL API",
                category=CapabilityCategory.NETWORK_OPERATIONS,
                interface_type=InterfaceType.GRAPHQL_API,
                description=f"GraphQL API with {len(capabilities)} operations",
                confidence=0.9
            )
            capabilities.insert(0, general_capability)
        
        return capabilities
    
    def _analyze_json_api_config(self, app: Application, content: str, file_path: Path) -> List[Capability]:
        """
        Analyze JSON configuration files for API settings.
        
        Args:
            app: Application being analyzed
            content: File content
            file_path: Path to the file
            
        Returns:
            List[Capability]: Configuration-based capabilities
        """
        capabilities = []
        
        try:
            config_data = json.loads(content)
            if not isinstance(config_data, dict):
                return capabilities
            
            # Look for API-related configuration
            api_indicators = ['port', 'host', 'routes', 'endpoints', 'middleware', 'cors']
            
            if any(key in config_data for key in api_indicators):
                capability = self._create_capability(
                    app=app,
                    name="API Configuration",
                    category=CapabilityCategory.NETWORK_OPERATIONS,
                    interface_type=InterfaceType.REST_API,
                    description=f"API configuration from {file_path.name}",
                    confidence=0.6
                )
                capabilities.append(capability)
        
        except json.JSONDecodeError:
            pass
        
        return capabilities
    
    def _analyze_generic_api_file(self, app: Application, content: str, file_path: Path) -> List[Capability]:
        """
        Analyze generic API-related files.
        
        Args:
            app: Application being analyzed
            content: File content
            file_path: Path to the file
            
        Returns:
            List[Capability]: Generic API capabilities
        """
        capabilities = []
        
        # Look for API patterns in the content
        api_patterns = [
            (r'@app\.route\([\'"]([^\'"]+)[\'"]', 'Flask Route'),
            (r'app\.(get|post|put|delete)\([\'"]([^\'"]+)[\'"]', 'Express Route'),
            (r'@RequestMapping\([\'"]([^\'"]+)[\'"]', 'Spring Route'),
            (r'@Path\([\'"]([^\'"]+)[\'"]', 'JAX-RS Endpoint'),
            (r'router\.(get|post|put|delete)\([\'"]([^\'"]+)[\'"]', 'Router Endpoint')
        ]
        
        for pattern, capability_type in api_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    path = match[1] if len(match) > 1 else match[0]
                    method = match[0] if len(match) > 1 else 'GET'
                else:
                    path = match
                    method = 'GET'
                
                capability = self._create_capability(
                    app=app,
                    name=f"{capability_type}: {method.upper()} {path}",
                    category=CapabilityCategory.NETWORK_OPERATIONS,
                    interface_type=InterfaceType.REST_API,
                    description=f"API endpoint defined in {file_path.name}",
                    confidence=0.7,
                    examples=[f"{method.upper()} {path}"]
                )
                capabilities.append(capability)
        
        return capabilities
    
    def _appears_to_be_api_server(self, app: Application) -> bool:
        """
        Check if the application appears to be an API server.
        
        Args:
            app: Application to check
            
        Returns:
            bool: True if it appears to be an API server
        """
        # Check application name for server indicators
        name_lower = app.name.lower()
        server_indicators = [
            'server', 'api', 'service', 'daemon', 'web', 'http',
            'rest', 'graphql', 'endpoint', 'gateway'
        ]
        
        if any(indicator in name_lower for indicator in server_indicators):
            return True
        
        # Check if it's a known web framework executable
        known_servers = [
            'node', 'python', 'java', 'ruby', 'php', 'go',
            'nginx', 'apache', 'tomcat', 'jetty', 'gunicorn',
            'uvicorn', 'flask', 'django', 'express'
        ]
        
        executable_name = app.executable_path.name.lower()
        if any(server in executable_name for server in known_servers):
            return True
        
        return False
    
    def _analyze_runtime_api(self, app: Application) -> List[Capability]:
        """
        Analyze running API by attempting to connect to common ports.
        
        Args:
            app: Application being analyzed
            
        Returns:
            List[Capability]: Runtime API capabilities
        """
        capabilities = []
        
        # This is a placeholder for runtime API detection
        # In a real implementation, you might:
        # 1. Check if the application is running
        # 2. Scan common ports for HTTP services
        # 3. Try to fetch API documentation endpoints
        # 4. Analyze HTTP responses for API patterns
        
        # For now, we'll create a generic capability if it appears to be a server
        if self._appears_to_be_api_server(app):
            capability = self._create_capability(
                app=app,
                name="Runtime API Service",
                category=CapabilityCategory.NETWORK_OPERATIONS,
                interface_type=InterfaceType.REST_API,
                description="Application appears to provide API services at runtime",
                confidence=0.5
            )
            capabilities.append(capability)
        
        return capabilities
    
    def _analyze_api_configuration(self, app: Application) -> List[Capability]:
        """
        Analyze configuration files for API-related settings.
        
        Args:
            app: Application being analyzed
            
        Returns:
            List[Capability]: Configuration-based capabilities
        """
        capabilities = []
        
        # Look for common configuration files
        config_patterns = [
            "config.*",
            "*.conf",
            "*.ini",
            "*.properties",
            "package.json",
            "requirements.txt",
            "Gemfile",
            "pom.xml"
        ]
        
        config_files = []
        for pattern in config_patterns:
            config_files.extend(app.installation_path.glob(pattern))
        
        for config_file in config_files[:10]:  # Limit analysis
            if config_file.is_file():
                content = self._safe_read_file(config_file)
                if content and self._contains_api_config(content):
                    capability = self._create_capability(
                        app=app,
                        name=f"API Configuration ({config_file.name})",
                        category=CapabilityCategory.NETWORK_OPERATIONS,
                        interface_type=InterfaceType.REST_API,
                        description=f"API configuration found in {config_file.name}",
                        confidence=0.6
                    )
                    capabilities.append(capability)
        
        return capabilities
    
    def _contains_api_config(self, content: str) -> bool:
        """
        Check if content contains API-related configuration.
        
        Args:
            content: File content to check
            
        Returns:
            bool: True if API configuration is detected
        """
        content_lower = content.lower()
        api_config_indicators = [
            'port', 'host', 'server', 'api', 'endpoint', 'route',
            'cors', 'middleware', 'express', 'flask', 'django',
            'spring', 'fastapi', 'gin', 'echo', 'fiber'
        ]
        
        return any(indicator in content_lower for indicator in api_config_indicators)