"""
Base analysis strategy implementation.

This module provides a common base class for all analysis strategies
with shared functionality and utilities.
"""

import logging
import re
from pathlib import Path
from typing import List, Optional, Dict, Any
from abc import ABC

from ...core.interfaces import AnalysisStrategy
from ...core.models import Application, Capability, CapabilityCategory, InterfaceType, Parameter


logger = logging.getLogger(__name__)


class BaseAnalysisStrategy(AnalysisStrategy, ABC):
    """
    Base implementation for analysis strategies.
    
    Provides common functionality and utilities that all analysis
    strategies can inherit and extend.
    """
    
    def __init__(self, strategy_name: str, confidence_factor: float = 1.0):
        """
        Initialize the base analysis strategy.
        
        Args:
            strategy_name: Name of this strategy
            confidence_factor: Base confidence factor for this strategy (0.0 to 1.0)
        """
        self.strategy_name = strategy_name
        self.confidence_factor = max(0.0, min(1.0, confidence_factor))
        self.logger = logging.getLogger(f"{__name__}.{strategy_name}")
    
    def get_strategy_name(self) -> str:
        """Get the name of this analysis strategy."""
        return self.strategy_name
    
    def get_confidence_factor(self) -> float:
        """Get the base confidence factor for this strategy."""
        return self.confidence_factor
    
    def _create_capability(
        self,
        app: Application,
        name: str,
        category: CapabilityCategory,
        interface_type: InterfaceType,
        description: str = "",
        confidence: float = 0.5,
        parameters: Optional[List[Parameter]] = None,
        examples: Optional[List[str]] = None,
        supported_formats: Optional[List[str]] = None
    ) -> Capability:
        """
        Create a capability with common fields populated.
        
        Args:
            app: Application this capability belongs to
            name: Capability name
            category: Capability category
            interface_type: Interface type
            description: Capability description
            confidence: Confidence score (0.0 to 1.0)
            parameters: List of parameters
            examples: List of usage examples
            supported_formats: List of supported file formats
            
        Returns:
            Capability: Created capability
        """
        return Capability(
            application_id=app.id,
            name=name,
            category=category,
            interface_type=interface_type,
            description=description,
            confidence_score=max(0.0, min(1.0, confidence)),
            parameters=parameters or [],
            examples=examples or [],
            supported_formats=supported_formats or []
        )
    
    def _extract_file_formats(self, text: str) -> List[str]:
        """
        Extract file format extensions from text.
        
        Args:
            text: Text to search for file formats
            
        Returns:
            List[str]: List of file format extensions found
        """
        # Common file format patterns
        format_patterns = [
            r'\b\w+\.(jpg|jpeg|png|gif|bmp|tiff|svg|webp)\b',  # Images
            r'\b\w+\.(mp4|avi|mov|wmv|flv|mkv|webm)\b',        # Videos
            r'\b\w+\.(mp3|wav|flac|aac|ogg|m4a)\b',            # Audio
            r'\b\w+\.(pdf|doc|docx|txt|rtf|odt)\b',            # Documents
            r'\b\w+\.(zip|rar|7z|tar|gz|bz2)\b',               # Archives
            r'\b\w+\.(json|xml|yaml|yml|csv|tsv)\b',           # Data
            r'\b\w+\.(html|htm|css|js|php|py|java|cpp|c)\b'    # Code
        ]
        
        formats = set()
        text_lower = text.lower()
        
        for pattern in format_patterns:
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            formats.update(matches)
        
        # Also look for explicit format mentions
        format_keywords = [
            'jpeg', 'jpg', 'png', 'gif', 'bmp', 'tiff', 'svg', 'webp',
            'mp4', 'avi', 'mov', 'wmv', 'flv', 'mkv', 'webm',
            'mp3', 'wav', 'flac', 'aac', 'ogg', 'm4a',
            'pdf', 'doc', 'docx', 'txt', 'rtf', 'odt',
            'zip', 'rar', '7z', 'tar', 'gz', 'bz2',
            'json', 'xml', 'yaml', 'yml', 'csv', 'tsv',
            'html', 'htm', 'css', 'js', 'php', 'py', 'java', 'cpp'
        ]
        
        for keyword in format_keywords:
            if keyword in text_lower:
                formats.add(keyword)
        
        return sorted(list(formats))
    
    def _categorize_capability(self, name: str, description: str) -> CapabilityCategory:
        """
        Categorize a capability based on its name and description.
        
        Args:
            name: Capability name
            description: Capability description
            
        Returns:
            CapabilityCategory: Inferred category
        """
        text = f"{name} {description}".lower()
        
        # File processing keywords
        if any(keyword in text for keyword in [
            'file', 'convert', 'transform', 'process', 'parse', 'read', 'write',
            'import', 'export', 'format', 'encode', 'decode', 'compress', 'extract'
        ]):
            return CapabilityCategory.FILE_PROCESSING
        
        # Network operations keywords
        if any(keyword in text for keyword in [
            'http', 'https', 'url', 'download', 'upload', 'request', 'response',
            'api', 'rest', 'soap', 'web', 'network', 'socket', 'tcp', 'udp'
        ]):
            return CapabilityCategory.NETWORK_OPERATIONS
        
        # Data transformation keywords
        if any(keyword in text for keyword in [
            'data', 'transform', 'filter', 'sort', 'query', 'search', 'index',
            'database', 'sql', 'json', 'xml', 'csv', 'parse', 'serialize'
        ]):
            return CapabilityCategory.DATA_TRANSFORMATION
        
        # User interface keywords
        if any(keyword in text for keyword in [
            'gui', 'window', 'dialog', 'menu', 'button', 'interface', 'ui',
            'display', 'show', 'view', 'render', 'draw', 'graphics'
        ]):
            return CapabilityCategory.USER_INTERFACE
        
        # Development tools keywords
        if any(keyword in text for keyword in [
            'compile', 'build', 'debug', 'test', 'lint', 'format', 'refactor',
            'git', 'version', 'deploy', 'package', 'bundle', 'minify'
        ]):
            return CapabilityCategory.DEVELOPMENT_TOOLS
        
        # Multimedia keywords
        if any(keyword in text for keyword in [
            'image', 'video', 'audio', 'media', 'play', 'record', 'edit',
            'resize', 'crop', 'filter', 'effect', 'codec', 'stream'
        ]):
            return CapabilityCategory.MULTIMEDIA
        
        # Security keywords
        if any(keyword in text for keyword in [
            'encrypt', 'decrypt', 'hash', 'sign', 'verify', 'certificate',
            'key', 'password', 'auth', 'security', 'crypto', 'ssl', 'tls'
        ]):
            return CapabilityCategory.SECURITY
        
        # Database keywords
        if any(keyword in text for keyword in [
            'database', 'db', 'sql', 'query', 'table', 'record', 'schema',
            'migrate', 'backup', 'restore', 'index', 'transaction'
        ]):
            return CapabilityCategory.DATABASE
        
        # Communication keywords
        if any(keyword in text for keyword in [
            'email', 'mail', 'message', 'chat', 'send', 'receive', 'notify',
            'alert', 'sms', 'push', 'webhook', 'callback'
        ]):
            return CapabilityCategory.COMMUNICATION
        
        # Default to system integration
        return CapabilityCategory.SYSTEM_INTEGRATION
    
    def _infer_interface_type(self, app: Application, capability_info: Dict[str, Any]) -> InterfaceType:
        """
        Infer the interface type based on application and capability information.
        
        Args:
            app: Application being analyzed
            capability_info: Information about the capability
            
        Returns:
            InterfaceType: Inferred interface type
        """
        # Check if it's a GUI application
        if any(keyword in app.name.lower() for keyword in ['gui', 'desktop', 'window']):
            return InterfaceType.GUI
        
        # Check for web interface indicators
        if any(keyword in str(capability_info).lower() for keyword in [
            'http', 'web', 'browser', 'html', 'port', 'server'
        ]):
            return InterfaceType.WEB_INTERFACE
        
        # Check for API indicators
        if any(keyword in str(capability_info).lower() for keyword in [
            'api', 'rest', 'json', 'endpoint', 'service'
        ]):
            return InterfaceType.REST_API
        
        # Check for library indicators
        if any(keyword in str(capability_info).lower() for keyword in [
            'library', 'lib', 'module', 'import', 'package'
        ]):
            return InterfaceType.LIBRARY
        
        # Default to command line
        return InterfaceType.COMMAND_LINE
    
    def _parse_parameters_from_text(self, text: str) -> List[Parameter]:
        """
        Parse parameter information from text.
        
        Args:
            text: Text containing parameter information
            
        Returns:
            List[Parameter]: Parsed parameters
        """
        parameters = []
        
        # Common parameter patterns
        patterns = [
            r'--(\w+)(?:\s+<(\w+)>)?(?:\s+(.+?)(?=\s+--|$))?',  # --param <type> description
            r'-(\w)(?:\s+<(\w+)>)?(?:\s+(.+?)(?=\s+-|$))?',     # -p <type> description
            r'(\w+):\s*<(\w+)>\s*-\s*(.+?)(?=\n|\s{2,}|$)',     # param: <type> - description
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.MULTILINE | re.IGNORECASE)
            for match in matches:
                if len(match) >= 1:
                    name = match[0]
                    param_type = match[1] if len(match) > 1 and match[1] else "string"
                    description = match[2] if len(match) > 2 and match[2] else ""
                    
                    # Clean up description
                    description = description.strip().rstrip('.,;')
                    
                    # Determine if parameter is required
                    required = not any(keyword in description.lower() for keyword in [
                        'optional', 'default', 'if not specified'
                    ])
                    
                    parameter = Parameter(
                        name=name,
                        type=param_type,
                        description=description,
                        required=required
                    )
                    parameters.append(parameter)
        
        return parameters
    
    def _extract_examples_from_text(self, text: str) -> List[str]:
        """
        Extract usage examples from text.
        
        Args:
            text: Text containing examples
            
        Returns:
            List[str]: Extracted examples
        """
        examples = []
        
        # Look for example patterns
        example_patterns = [
            r'(?:example|usage|sample):\s*(.+?)(?=\n\n|\n[A-Z]|$)',
            r'(?:e\.g\.|for example):\s*(.+?)(?=\n\n|\n[A-Z]|$)',
            r'`([^`]+)`',  # Code in backticks
            r'^\s*\$\s*(.+?)$',  # Shell commands starting with $
        ]
        
        for pattern in example_patterns:
            matches = re.findall(pattern, text, re.MULTILINE | re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    match = match[0] if match else ""
                
                example = match.strip()
                if example and len(example) > 5:  # Filter out very short matches
                    examples.append(example)
        
        return examples[:5]  # Limit to 5 examples
    
    def _is_text_file(self, file_path: Path) -> bool:
        """
        Check if a file is likely a text file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            bool: True if the file is likely text
        """
        text_extensions = {
            '.txt', '.md', '.rst', '.doc', '.docx', '.pdf',
            '.html', '.htm', '.xml', '.json', '.yaml', '.yml',
            '.ini', '.cfg', '.conf', '.log', '.readme'
        }
        
        return file_path.suffix.lower() in text_extensions
    
    def _safe_read_file(self, file_path: Path, max_size: int = 1024 * 1024) -> Optional[str]:
        """
        Safely read a text file with size and encoding protection.
        
        Args:
            file_path: Path to the file
            max_size: Maximum file size to read in bytes
            
        Returns:
            Optional[str]: File content if successful, None otherwise
        """
        try:
            if not file_path.exists() or not file_path.is_file():
                return None
            
            # Check file size
            if file_path.stat().st_size > max_size:
                self.logger.debug(f"File {file_path} too large ({file_path.stat().st_size} bytes)")
                return None
            
            # Try different encodings
            encodings = ['utf-8', 'utf-16', 'latin-1', 'cp1252']
            
            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        return f.read()
                except UnicodeDecodeError:
                    continue
            
            self.logger.debug(f"Could not decode file {file_path} with any encoding")
            return None
            
        except (OSError, PermissionError) as e:
            self.logger.debug(f"Could not read file {file_path}: {e}")
            return None