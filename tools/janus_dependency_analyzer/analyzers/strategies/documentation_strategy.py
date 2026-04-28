"""
Documentation analysis strategy.

This strategy analyzes application documentation files (README, man pages, etc.)
to extract capability information.
"""

import logging
from pathlib import Path
from typing import List, Optional
import re

from .base_strategy import BaseAnalysisStrategy
from ...core.models import Application, Capability, CapabilityCategory, InterfaceType


logger = logging.getLogger(__name__)


class DocumentationAnalysisStrategy(BaseAnalysisStrategy):
    """
    Strategy for analyzing application documentation to extract capabilities.
    
    This strategy searches for and analyzes documentation files such as:
    - README files
    - Man pages
    - Help files
    - Documentation directories
    """
    
    def __init__(self):
        """Initialize the documentation analysis strategy."""
        super().__init__("documentation_analysis", confidence_factor=0.8)
        
        # Common documentation file patterns
        self.doc_patterns = [
            "README*",
            "readme*",
            "HELP*",
            "help*",
            "MANUAL*",
            "manual*",
            "USAGE*",
            "usage*",
            "*.md",
            "*.rst",
            "*.txt"
        ]
        
        # Documentation directories to search
        self.doc_directories = [
            "doc",
            "docs",
            "documentation",
            "help",
            "man",
            "manual"
        ]
    
    def can_analyze(self, app: Application) -> bool:
        """
        Check if this strategy can analyze the given application.
        
        Args:
            app: Application to check
            
        Returns:
            bool: True if documentation files are found
        """
        if not app.is_accessible:
            return False
        
        # Look for documentation files in the installation directory
        doc_files = self._find_documentation_files(app.installation_path)
        return len(doc_files) > 0
    
    def extract_capabilities(self, app: Application) -> List[Capability]:
        """
        Extract capabilities from application documentation.
        
        Args:
            app: Application to analyze
            
        Returns:
            List[Capability]: Extracted capabilities
        """
        capabilities = []
        
        # Find all documentation files
        doc_files = self._find_documentation_files(app.installation_path)
        
        for doc_file in doc_files:
            try:
                file_capabilities = self._analyze_documentation_file(app, doc_file)
                capabilities.extend(file_capabilities)
            except Exception as e:
                self.logger.debug(f"Error analyzing documentation file {doc_file}: {e}")
        
        return capabilities
    
    def _find_documentation_files(self, installation_path: Path) -> List[Path]:
        """
        Find documentation files in the installation directory.
        
        Args:
            installation_path: Path to search for documentation
            
        Returns:
            List[Path]: Found documentation files
        """
        doc_files = []
        
        if not installation_path.exists():
            return doc_files
        
        try:
            # Search in the main installation directory
            for pattern in self.doc_patterns:
                doc_files.extend(installation_path.glob(pattern))
            
            # Search in documentation subdirectories
            for doc_dir_name in self.doc_directories:
                doc_dir = installation_path / doc_dir_name
                if doc_dir.exists() and doc_dir.is_dir():
                    for pattern in self.doc_patterns:
                        doc_files.extend(doc_dir.glob(pattern))
            
            # Filter to only include text files
            doc_files = [f for f in doc_files if f.is_file() and self._is_text_file(f)]
            
        except (OSError, PermissionError) as e:
            self.logger.debug(f"Error searching for documentation in {installation_path}: {e}")
        
        return doc_files[:10]  # Limit to prevent excessive processing
    
    def _analyze_documentation_file(self, app: Application, doc_file: Path) -> List[Capability]:
        """
        Analyze a single documentation file to extract capabilities.
        
        Args:
            app: Application being analyzed
            doc_file: Documentation file to analyze
            
        Returns:
            List[Capability]: Capabilities extracted from the file
        """
        capabilities = []
        
        # Read the documentation file
        content = self._safe_read_file(doc_file)
        if not content:
            return capabilities
        
        # Extract capabilities using various patterns
        capabilities.extend(self._extract_feature_capabilities(app, content))
        capabilities.extend(self._extract_command_capabilities(app, content))
        capabilities.extend(self._extract_format_capabilities(app, content))
        
        return capabilities
    
    def _extract_feature_capabilities(self, app: Application, content: str) -> List[Capability]:
        """
        Extract capabilities from feature descriptions in documentation.
        
        Args:
            app: Application being analyzed
            content: Documentation content
            
        Returns:
            List[Capability]: Feature-based capabilities
        """
        capabilities = []
        
        # Look for feature sections
        feature_patterns = [
            r'(?:features?|capabilities?|functions?):\s*\n((?:[-*•]\s*.+\n?)+)',
            r'(?:what (?:it|this) (?:does|can do)):\s*\n((?:[-*•]\s*.+\n?)+)',
            r'(?:supports?):\s*\n((?:[-*•]\s*.+\n?)+)'
        ]
        
        for pattern in feature_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                features = re.findall(r'[-*•]\s*(.+)', match)
                for feature in features:
                    feature = feature.strip()
                    if len(feature) > 10:  # Filter out very short features
                        capability = self._create_capability(
                            app=app,
                            name=self._extract_capability_name(feature),
                            category=self._categorize_capability(feature, feature),
                            interface_type=self._infer_interface_type(app, {"description": feature}),
                            description=feature,
                            confidence=0.7,
                            supported_formats=self._extract_file_formats(feature)
                        )
                        capabilities.append(capability)
        
        return capabilities
    
    def _extract_command_capabilities(self, app: Application, content: str) -> List[Capability]:
        """
        Extract capabilities from command descriptions in documentation.
        
        Args:
            app: Application being analyzed
            content: Documentation content
            
        Returns:
            List[Capability]: Command-based capabilities
        """
        capabilities = []
        
        # Look for command patterns
        command_patterns = [
            r'(?:usage|syntax):\s*(.+?)(?:\n\n|\n[A-Z]|$)',
            r'(?:commands?):\s*\n((?:\s+\w+.+\n?)+)',
            r'(?:options?):\s*\n((?:\s+[-\w]+.+\n?)+)'
        ]
        
        for pattern in command_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE | re.MULTILINE | re.DOTALL)
            for match in matches:
                # Extract individual commands or options
                lines = match.split('\n')
                for line in lines:
                    line = line.strip()
                    if line and len(line) > 5:
                        # Try to extract command name and description
                        command_match = re.match(r'(\w+)\s+(.+)', line)
                        if command_match:
                            command_name = command_match.group(1)
                            description = command_match.group(2)
                            
                            capability = self._create_capability(
                                app=app,
                                name=command_name,
                                category=self._categorize_capability(command_name, description),
                                interface_type=InterfaceType.COMMAND_LINE,
                                description=description,
                                confidence=0.6,
                                parameters=self._parse_parameters_from_text(line),
                                examples=self._extract_examples_from_text(line)
                            )
                            capabilities.append(capability)
        
        return capabilities
    
    def _extract_format_capabilities(self, app: Application, content: str) -> List[Capability]:
        """
        Extract capabilities related to file format support.
        
        Args:
            app: Application being analyzed
            content: Documentation content
            
        Returns:
            List[Capability]: Format-based capabilities
        """
        capabilities = []
        
        # Look for format support mentions
        format_patterns = [
            r'(?:supports?|handles?|processes?|converts?)\s+(?:the\s+)?(?:following\s+)?(?:file\s+)?(?:formats?|types?):\s*(.+?)(?:\n\n|\n[A-Z]|$)',
            r'(?:input|output)\s+(?:file\s+)?(?:formats?|types?):\s*(.+?)(?:\n\n|\n[A-Z]|$)',
            r'(?:compatible\s+with|works\s+with):\s*(.+?)(?:\n\n|\n[A-Z]|$)'
        ]
        
        for pattern in format_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE | re.MULTILINE | re.DOTALL)
            for match in matches:
                formats = self._extract_file_formats(match)
                if formats:
                    capability = self._create_capability(
                        app=app,
                        name="File Format Support",
                        category=CapabilityCategory.FILE_PROCESSING,
                        interface_type=self._infer_interface_type(app, {"formats": formats}),
                        description=f"Supports file formats: {', '.join(formats)}",
                        confidence=0.8,
                        supported_formats=formats
                    )
                    capabilities.append(capability)
        
        return capabilities
    
    def _extract_capability_name(self, description: str) -> str:
        """
        Extract a concise capability name from a description.
        
        Args:
            description: Full capability description
            
        Returns:
            str: Extracted capability name
        """
        # Remove common prefixes
        description = re.sub(r'^(?:can\s+|able\s+to\s+|supports?\s+)', '', description, flags=re.IGNORECASE)
        
        # Take the first few words as the name
        words = description.split()
        if len(words) <= 3:
            return description
        
        # Try to find a good breaking point
        for i in range(2, min(5, len(words))):
            if words[i].endswith(('.', ',', ';', ':')):
                return ' '.join(words[:i])
        
        # Default to first 3 words
        return ' '.join(words[:3])
    
    def _is_documentation_file(self, file_path: Path) -> bool:
        """
        Check if a file is likely a documentation file.
        
        Args:
            file_path: Path to check
            
        Returns:
            bool: True if the file is likely documentation
        """
        name_lower = file_path.name.lower()
        
        # Check for documentation file names
        doc_names = [
            'readme', 'help', 'manual', 'usage', 'doc', 'docs',
            'changelog', 'changes', 'news', 'history', 'license',
            'copying', 'install', 'authors', 'contributors'
        ]
        
        for doc_name in doc_names:
            if doc_name in name_lower:
                return True
        
        # Check for documentation extensions
        doc_extensions = {'.md', '.rst', '.txt', '.doc', '.docx', '.pdf'}
        if file_path.suffix.lower() in doc_extensions:
            return True
        
        return False