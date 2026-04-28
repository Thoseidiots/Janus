"""
Help text analysis strategy.

This strategy executes applications with help flags (--help, -h, etc.)
to extract capability information from their help output.
"""

import logging
import subprocess
from typing import List, Optional, Dict, Any
import re
import shlex

from .base_strategy import BaseAnalysisStrategy
from ...core.models import Application, Capability, CapabilityCategory, InterfaceType


logger = logging.getLogger(__name__)


class HelpTextAnalysisStrategy(BaseAnalysisStrategy):
    """
    Strategy for analyzing application help text to extract capabilities.
    
    This strategy executes applications with various help flags and analyzes
    the output to identify supported operations and parameters.
    """
    
    def __init__(self):
        """Initialize the help text analysis strategy."""
        super().__init__("help_text_analysis", confidence_factor=0.9)
        
        # Common help flags to try
        self.help_flags = [
            "--help",
            "-h",
            "-?",
            "help",
            "--usage",
            "-u",
            "--version",  # Sometimes includes usage info
            "-v"
        ]
        
        # Adaptive timeout: start fast, increase if needed
        self.timeout_seconds = 0.5  # Start with 500ms
        self.max_timeout_seconds = 3.0  # Maximum 3 seconds
    
    def can_analyze(self, app: Application) -> bool:
        """
        Check if this strategy can analyze the given application.
        
        Args:
            app: Application to check
            
        Returns:
            bool: True if the application is executable and accessible
        """
        if not app.is_accessible:
            return False
        
        # Check if the executable exists and is executable
        return (app.executable_path.exists() and 
                app.executable_path.is_file() and
                self._is_executable(app.executable_path))
    
    def extract_capabilities(self, app: Application) -> List[Capability]:
        """
        Extract capabilities from application help text.
        
        Args:
            app: Application to analyze
            
        Returns:
            List[Capability]: Extracted capabilities
        """
        capabilities = []
        
        # Try different help flags
        for help_flag in self.help_flags:
            try:
                help_text = self._get_help_text(app, help_flag)
                if help_text:
                    flag_capabilities = self._analyze_help_text(app, help_text, help_flag)
                    capabilities.extend(flag_capabilities)
                    
                    # If we got good results, we can stop trying more flags
                    if len(flag_capabilities) > 0:
                        break
                        
            except Exception as e:
                self.logger.debug(f"Error getting help text with {help_flag} for {app.name}: {e}")
        
        return capabilities
    
    def _get_help_text(self, app: Application, help_flag: str) -> Optional[str]:
        """
        Execute the application with a help flag and capture output.
        Uses subprocess cache to avoid redundant executions.
        
        Args:
            app: Application to execute
            help_flag: Help flag to use
            
        Returns:
            Optional[str]: Help text output if successful
        """
        from ...cache.subprocess_cache import get_subprocess_cache
        
        # Check cache first
        cache = get_subprocess_cache()
        command_args = (help_flag,)
        cached_result = cache.get(app.executable_path, command_args)
        
        if cached_result is not None:
            output = cached_result['stdout'] or cached_result['stderr']
            if output and len(output.strip()) > 20:
                return output
            return None
        
        # Not in cache, execute subprocess
        try:
            cmd = [str(app.executable_path), help_flag]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='ignore',
                timeout=self.timeout_seconds,
                cwd=app.installation_path
            )
            
            # Cache the result
            cache.put(
                app.executable_path,
                command_args,
                result.stdout or '',
                result.stderr or '',
                result.returncode
            )
            
            output = result.stdout or result.stderr
            
            if output and len(output.strip()) > 20:
                return output
            
        except subprocess.TimeoutExpired:
            self.logger.debug(f"Timeout executing {app.name} {help_flag}")
            # Cache the timeout result to avoid retrying
            cache.put(app.executable_path, command_args, '', '', -1)
        except subprocess.SubprocessError as e:
            self.logger.debug(f"Error executing {app.name} {help_flag}: {e}")
            cache.put(app.executable_path, command_args, '', '', -1)
        except Exception as e:
            self.logger.debug(f"Unexpected error executing {app.name} {help_flag}: {e}")
        
        return None
    
    def _analyze_help_text(self, app: Application, help_text: str, help_flag: str) -> List[Capability]:
        """
        Analyze help text to extract capabilities.
        
        Args:
            app: Application being analyzed
            help_text: Help text to analyze
            help_flag: Help flag that was used
            
        Returns:
            List[Capability]: Extracted capabilities
        """
        capabilities = []
        
        # Extract different types of capabilities
        capabilities.extend(self._extract_command_capabilities(app, help_text))
        capabilities.extend(self._extract_option_capabilities(app, help_text))
        capabilities.extend(self._extract_usage_capabilities(app, help_text))
        capabilities.extend(self._extract_description_capabilities(app, help_text))
        
        return capabilities
    
    def _extract_command_capabilities(self, app: Application, help_text: str) -> List[Capability]:
        """
        Extract capabilities from command descriptions in help text.
        
        Args:
            app: Application being analyzed
            help_text: Help text to analyze
            
        Returns:
            List[Capability]: Command-based capabilities
        """
        capabilities = []
        
        # Look for command sections
        command_patterns = [
            r'(?:commands?|subcommands?):\s*\n((?:\s+\w+.+\n?)+)',
            r'(?:available\s+commands?):\s*\n((?:\s+\w+.+\n?)+)',
            r'(?:usage|use):\s*\n((?:\s+\w+.+\n?)+)'
        ]
        
        for pattern in command_patterns:
            matches = re.findall(pattern, help_text, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                commands = self._parse_command_list(match)
                for command_name, description in commands:
                    capability = self._create_capability(
                        app=app,
                        name=command_name,
                        category=self._categorize_capability(command_name, description),
                        interface_type=InterfaceType.COMMAND_LINE,
                        description=description,
                        confidence=0.8,
                        examples=[f"{app.executable_path.name} {command_name}"]
                    )
                    capabilities.append(capability)
        
        return capabilities
    
    def _extract_option_capabilities(self, app: Application, help_text: str) -> List[Capability]:
        """
        Extract capabilities from option descriptions in help text.
        
        Args:
            app: Application being analyzed
            help_text: Help text to analyze
            
        Returns:
            List[Capability]: Option-based capabilities
        """
        capabilities = []
        
        # Look for option sections
        option_patterns = [
            r'(?:options?|flags?):\s*\n((?:\s+[-\w].+\n?)+)',
            r'(?:available\s+options?):\s*\n((?:\s+[-\w].+\n?)+)'
        ]
        
        for pattern in option_patterns:
            matches = re.findall(pattern, help_text, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                options = self._parse_option_list(match)
                
                # Group related options into capabilities
                grouped_options = self._group_options_by_functionality(options)
                
                for group_name, group_options in grouped_options.items():
                    descriptions = [desc for _, desc in group_options if desc]
                    combined_description = "; ".join(descriptions[:3])  # Limit description length
                    
                    parameters = []
                    for option, desc in group_options:
                        param = self._parse_option_parameter(option, desc)
                        if param:
                            parameters.append(param)
                    
                    capability = self._create_capability(
                        app=app,
                        name=group_name,
                        category=self._categorize_capability(group_name, combined_description),
                        interface_type=InterfaceType.COMMAND_LINE,
                        description=combined_description,
                        confidence=0.7,
                        parameters=parameters
                    )
                    capabilities.append(capability)
        
        return capabilities
    
    def _extract_usage_capabilities(self, app: Application, help_text: str) -> List[Capability]:
        """
        Extract capabilities from usage patterns in help text.
        
        Args:
            app: Application being analyzed
            help_text: Help text to analyze
            
        Returns:
            List[Capability]: Usage-based capabilities
        """
        capabilities = []
        
        # Look for usage patterns
        usage_patterns = [
            r'(?:usage|syntax):\s*(.+?)(?:\n\n|\n[A-Z]|$)',
            r'(?:use|run):\s*(.+?)(?:\n\n|\n[A-Z]|$)'
        ]
        
        for pattern in usage_patterns:
            matches = re.findall(pattern, help_text, re.IGNORECASE | re.MULTILINE | re.DOTALL)
            for match in matches:
                usage_text = match.strip()
                if len(usage_text) > 10:
                    # Extract the main functionality from usage
                    functionality = self._extract_functionality_from_usage(usage_text)
                    if functionality:
                        capability = self._create_capability(
                            app=app,
                            name=functionality,
                            category=self._categorize_capability(functionality, usage_text),
                            interface_type=InterfaceType.COMMAND_LINE,
                            description=f"Usage: {usage_text}",
                            confidence=0.6,
                            examples=[usage_text],
                            supported_formats=self._extract_file_formats(usage_text)
                        )
                        capabilities.append(capability)
        
        return capabilities
    
    def _extract_description_capabilities(self, app: Application, help_text: str) -> List[Capability]:
        """
        Extract capabilities from general description in help text.
        
        Args:
            app: Application being analyzed
            help_text: Help text to analyze
            
        Returns:
            List[Capability]: Description-based capabilities
        """
        capabilities = []
        
        # Look for description sections
        description_patterns = [
            r'(?:description|about):\s*(.+?)(?:\n\n|\n[A-Z]|$)',
            r'^(.+?)(?:\n\n|\nusage|\noptions|\ncommands)',  # First paragraph
        ]
        
        for pattern in description_patterns:
            matches = re.findall(pattern, help_text, re.IGNORECASE | re.MULTILINE | re.DOTALL)
            for match in matches:
                description = match.strip()
                if len(description) > 20 and len(description) < 500:  # Reasonable description length
                    # Extract key functionalities from description
                    functionalities = self._extract_functionalities_from_description(description)
                    for functionality in functionalities:
                        capability = self._create_capability(
                            app=app,
                            name=functionality,
                            category=self._categorize_capability(functionality, description),
                            interface_type=self._infer_interface_type(app, {"description": description}),
                            description=description,
                            confidence=0.5,
                            supported_formats=self._extract_file_formats(description)
                        )
                        capabilities.append(capability)
                    break  # Only use the first good description
        
        return capabilities
    
    def _parse_command_list(self, command_text: str) -> List[tuple]:
        """
        Parse a list of commands from help text.
        
        Args:
            command_text: Text containing command list
            
        Returns:
            List[tuple]: List of (command_name, description) tuples
        """
        commands = []
        
        for line in command_text.split('\n'):
            line = line.strip()
            if line:
                # Try to parse command and description
                match = re.match(r'(\w+)\s+(.+)', line)
                if match:
                    command_name = match.group(1)
                    description = match.group(2).strip()
                    commands.append((command_name, description))
        
        return commands
    
    def _parse_option_list(self, option_text: str) -> List[tuple]:
        """
        Parse a list of options from help text.
        
        Args:
            option_text: Text containing option list
            
        Returns:
            List[tuple]: List of (option, description) tuples
        """
        options = []
        
        for line in option_text.split('\n'):
            line = line.strip()
            if line and line.startswith('-'):
                # Try to parse option and description
                parts = line.split(None, 1)
                if len(parts) >= 2:
                    option = parts[0]
                    description = parts[1]
                    options.append((option, description))
                else:
                    options.append((parts[0], ""))
        
        return options
    
    def _group_options_by_functionality(self, options: List[tuple]) -> Dict[str, List[tuple]]:
        """
        Group related options by their functionality.
        
        Args:
            options: List of (option, description) tuples
            
        Returns:
            Dict[str, List[tuple]]: Grouped options by functionality
        """
        groups = {}
        
        for option, description in options:
            # Determine the functionality group
            group_name = self._determine_option_group(option, description)
            
            if group_name not in groups:
                groups[group_name] = []
            
            groups[group_name].append((option, description))
        
        return groups
    
    def _determine_option_group(self, option: str, description: str) -> str:
        """
        Determine the functionality group for an option.
        
        Args:
            option: Option flag
            description: Option description
            
        Returns:
            str: Functionality group name
        """
        text = f"{option} {description}".lower()
        
        # Common option groups
        if any(keyword in text for keyword in ['output', 'format', 'export', 'save']):
            return "Output Control"
        elif any(keyword in text for keyword in ['input', 'file', 'read', 'load']):
            return "Input Control"
        elif any(keyword in text for keyword in ['verbose', 'quiet', 'debug', 'log']):
            return "Logging Control"
        elif any(keyword in text for keyword in ['config', 'setting', 'option']):
            return "Configuration"
        elif any(keyword in text for keyword in ['help', 'version', 'info']):
            return "Information"
        elif any(keyword in text for keyword in ['filter', 'select', 'match']):
            return "Filtering"
        elif any(keyword in text for keyword in ['process', 'transform', 'convert']):
            return "Processing"
        else:
            return "General Options"
    
    def _parse_option_parameter(self, option: str, description: str) -> Optional[Any]:
        """
        Parse parameter information from an option.
        
        Args:
            option: Option flag
            description: Option description
            
        Returns:
            Optional[Parameter]: Parsed parameter if applicable
        """
        from ...core.models import Parameter
        
        # Extract parameter name from option
        param_name = option.lstrip('-')
        
        # Determine parameter type from description
        param_type = "string"
        if any(keyword in description.lower() for keyword in ['number', 'count', 'size']):
            param_type = "integer"
        elif any(keyword in description.lower() for keyword in ['true', 'false', 'enable', 'disable']):
            param_type = "boolean"
        
        # Determine if required
        required = not any(keyword in description.lower() for keyword in ['optional', 'default'])
        
        return Parameter(
            name=param_name,
            type=param_type,
            description=description,
            required=required
        )
    
    def _extract_functionality_from_usage(self, usage_text: str) -> Optional[str]:
        """
        Extract the main functionality from a usage pattern.
        
        Args:
            usage_text: Usage pattern text
            
        Returns:
            Optional[str]: Extracted functionality name
        """
        # Remove the executable name and common patterns
        cleaned = re.sub(r'^\w+\s+', '', usage_text)  # Remove executable name
        cleaned = re.sub(r'\[.*?\]', '', cleaned)      # Remove optional parts
        cleaned = re.sub(r'<.*?>', '', cleaned)        # Remove placeholders
        
        # Extract the first meaningful word or phrase
        words = cleaned.split()
        if words:
            return words[0].capitalize()
        
        return None
    
    def _extract_functionalities_from_description(self, description: str) -> List[str]:
        """
        Extract key functionalities from a description.
        
        Args:
            description: Application description
            
        Returns:
            List[str]: List of functionality names
        """
        functionalities = []
        
        # Look for action verbs that indicate functionality
        action_patterns = [
            r'\b(convert|transform|process|analyze|generate|create|build|compile|parse|extract|compress|decompress|encrypt|decrypt|download|upload|sync|backup|restore|monitor|scan|search|find|replace|edit|modify|update|install|remove|configure|manage|control)\b'
        ]
        
        for pattern in action_patterns:
            matches = re.findall(pattern, description, re.IGNORECASE)
            for match in matches:
                functionalities.append(match.capitalize())
        
        # Remove duplicates and limit
        return list(set(functionalities))[:5]