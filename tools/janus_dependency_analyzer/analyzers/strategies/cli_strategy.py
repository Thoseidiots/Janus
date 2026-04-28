"""
Command-line interface analysis strategy.

This strategy analyzes command-line argument patterns and subcommands
to identify application capabilities.
"""

import logging
import subprocess
import re
from typing import List, Optional, Dict, Set
from pathlib import Path

from .base_strategy import BaseAnalysisStrategy
from ...core.models import Application, Capability, CapabilityCategory, InterfaceType, Parameter


logger = logging.getLogger(__name__)


class CommandLineInterfaceStrategy(BaseAnalysisStrategy):
    """
    Strategy for analyzing command-line interfaces to extract capabilities.
    
    This strategy examines command-line argument patterns, subcommands,
    and parameter structures to identify application capabilities.
    """
    
    def __init__(self):
        """Initialize the CLI analysis strategy."""
        super().__init__("cli_analysis", confidence_factor=0.85)
        
        # Common CLI patterns to analyze
        self.cli_patterns = [
            "--help",
            "-h", 
            "help",
            "--version",
            "-v",
            "--list",
            "--info"
        ]
        
        # Adaptive timeout: start fast
        self.timeout_seconds = 0.5  # Start with 500ms
        self.max_timeout_seconds = 3.0  # Maximum 3 seconds
    
    def can_analyze(self, app: Application) -> bool:
        """
        Check if this strategy can analyze the given application.
        
        Args:
            app: Application to check
            
        Returns:
            bool: True if the application appears to have a CLI interface
        """
        if not app.is_accessible:
            return False
        
        # Check if it's an executable file
        if not (app.executable_path.exists() and 
                app.executable_path.is_file() and
                self._is_executable(app.executable_path)):
            return False
        
        # Try a quick help command to see if it responds like a CLI tool
        return self._has_cli_interface(app)
    
    def extract_capabilities(self, app: Application) -> List[Capability]:
        """
        Extract capabilities from command-line interface analysis.
        
        Args:
            app: Application to analyze
            
        Returns:
            List[Capability]: Extracted capabilities
        """
        capabilities = []
        
        # Analyze help output for CLI structure
        help_output = self._get_comprehensive_help(app)
        if help_output:
            capabilities.extend(self._analyze_cli_structure(app, help_output))
        
        # Analyze subcommands if present
        subcommands = self._discover_subcommands(app)
        for subcommand in subcommands:
            subcmd_capabilities = self._analyze_subcommand(app, subcommand)
            capabilities.extend(subcmd_capabilities)
        
        # Analyze argument patterns
        arg_capabilities = self._analyze_argument_patterns(app, help_output or "")
        capabilities.extend(arg_capabilities)
        
        return capabilities
    
    def _has_cli_interface(self, app: Application) -> bool:
        """
        Check if the application has a command-line interface.
        """
        try:
            result = subprocess.run(
                [str(app.executable_path), "--help"],
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='ignore',
                timeout=5
            )
            
            output = result.stdout or result.stderr
            if output and len(output.strip()) > 20:
                cli_indicators = [
                    'usage:', 'options:', 'commands:', 'arguments:',
                    'flags:', 'parameters:', 'syntax:'
                ]
                output_lower = output.lower()
                return any(indicator in output_lower for indicator in cli_indicators)
        
        except (subprocess.TimeoutExpired, subprocess.SubprocessError, UnicodeDecodeError):
            pass
        
        return False
    
    def _get_comprehensive_help(self, app: Application) -> Optional[str]:
        """
        Get comprehensive help output from the application.
        Tries flags in order and returns the first meaningful response.
        """
        help_commands = [
            ["--help"],
            ["-h"],
            ["help"],
            ["--usage"],
            ["-?"]
        ]
        
        for cmd_args in help_commands:
            try:
                result = subprocess.run(
                    [str(app.executable_path)] + cmd_args,
                    capture_output=True,
                    text=True,
                    encoding='utf-8',
                    errors='ignore',
                    timeout=self.timeout_seconds,
                    cwd=app.installation_path
                )
                
                output = result.stdout or result.stderr
                if output and len(output.strip()) > 50:
                    return output
                    
            except (subprocess.TimeoutExpired, subprocess.SubprocessError, UnicodeDecodeError):
                continue
        
        return None
    
    def _analyze_cli_structure(self, app: Application, help_output: str) -> List[Capability]:
        """
        Analyze the overall CLI structure from help output.
        
        Args:
            app: Application being analyzed
            help_output: Help text output
            
        Returns:
            List[Capability]: CLI structure capabilities
        """
        capabilities = []
        
        # Extract main usage patterns
        usage_patterns = self._extract_usage_patterns(help_output)
        for pattern in usage_patterns:
            capability = self._create_capability(
                app=app,
                name=f"CLI Usage: {pattern['name']}",
                category=CapabilityCategory.SYSTEM_INTEGRATION,
                interface_type=InterfaceType.COMMAND_LINE,
                description=pattern['description'],
                confidence=0.8,
                parameters=pattern.get('parameters', []),
                examples=pattern.get('examples', [])
            )
            capabilities.append(capability)
        
        # Extract option groups
        option_groups = self._extract_option_groups(help_output)
        for group_name, options in option_groups.items():
            parameters = []
            for option in options:
                param = self._create_parameter_from_option(option)
                if param:
                    parameters.append(param)
            
            capability = self._create_capability(
                app=app,
                name=group_name,
                category=self._categorize_capability(group_name, ""),
                interface_type=InterfaceType.COMMAND_LINE,
                description=f"Command-line options for {group_name.lower()}",
                confidence=0.7,
                parameters=parameters
            )
            capabilities.append(capability)
        
        return capabilities
    
    def _discover_subcommands(self, app: Application) -> List[str]:
        """
        Discover available subcommands declared in the application's help output.
        Does NOT probe common subcommand names — that's too slow (up to 100s per app).
        """
        subcommands = set()
        
        help_output = self._get_comprehensive_help(app)
        if not help_output:
            return []
        
        subcommand_patterns = [
            r'(?:commands?|subcommands?):\s*\n((?:\s+\w+.+\n?)+)',
            r'(?:available\s+commands?):\s*\n((?:\s+\w+.+\n?)+)',
        ]
        
        for pattern in subcommand_patterns:
            matches = re.findall(pattern, help_output, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                for line in match.split('\n'):
                    line = line.strip()
                    if line:
                        cmd_match = re.match(r'(\w+)', line)
                        if cmd_match:
                            subcommands.add(cmd_match.group(1))
        
        return list(subcommands)[:10]
    
    def _analyze_subcommand(self, app: Application, subcommand: str) -> List[Capability]:
        """
        Analyze a specific subcommand to extract its capabilities.
        
        Args:
            app: Application being analyzed
            subcommand: Subcommand to analyze
            
        Returns:
            List[Capability]: Subcommand capabilities
        """
        capabilities = []
        
        # Get help for the subcommand
        subcmd_help = self._get_subcommand_help(app, subcommand)
        if not subcmd_help:
            return capabilities
        
        # Extract subcommand functionality
        functionality = self._extract_subcommand_functionality(subcmd_help)
        if functionality:
            parameters = self._parse_parameters_from_text(subcmd_help)
            examples = self._extract_examples_from_text(subcmd_help)
            
            capability = self._create_capability(
                app=app,
                name=f"{subcommand.capitalize()} Command",
                category=self._categorize_capability(subcommand, functionality),
                interface_type=InterfaceType.COMMAND_LINE,
                description=functionality,
                confidence=0.8,
                parameters=parameters,
                examples=examples,
                supported_formats=self._extract_file_formats(subcmd_help)
            )
            capabilities.append(capability)
        
        return capabilities
    
    def _get_subcommand_help(self, app: Application, subcommand: str) -> Optional[str]:
        """Get help output for a specific subcommand."""
        help_variants = [
            [subcommand, "--help"],
            [subcommand, "-h"],
            ["help", subcommand]
        ]
        
        for variant in help_variants:
            try:
                result = subprocess.run(
                    [str(app.executable_path)] + variant,
                    capture_output=True,
                    text=True,
                    encoding='utf-8',
                    errors='ignore',
                    timeout=self.timeout_seconds
                )
                
                output = result.stdout or result.stderr
                if output and len(output.strip()) > 20:
                    return output
                    
            except (subprocess.TimeoutExpired, subprocess.SubprocessError, UnicodeDecodeError):
                continue
        
        return None
    
    def _extract_usage_patterns(self, help_output: str) -> List[Dict]:
        """
        Extract usage patterns from help output.
        
        Args:
            help_output: Help text to analyze
            
        Returns:
            List[Dict]: Usage patterns with metadata
        """
        patterns = []
        
        # Look for usage lines
        usage_matches = re.findall(
            r'(?:usage|syntax):\s*(.+?)(?:\n\n|\n[A-Z]|$)',
            help_output,
            re.IGNORECASE | re.MULTILINE
        )
        
        for usage_line in usage_matches:
            usage_line = usage_line.strip()
            if len(usage_line) > 10:
                pattern = {
                    'name': self._extract_pattern_name(usage_line),
                    'description': f"Usage pattern: {usage_line}",
                    'examples': [usage_line],
                    'parameters': self._parse_parameters_from_text(usage_line)
                }
                patterns.append(pattern)
        
        return patterns
    
    def _extract_option_groups(self, help_output: str) -> Dict[str, List[str]]:
        """
        Extract and group command-line options.
        
        Args:
            help_output: Help text to analyze
            
        Returns:
            Dict[str, List[str]]: Grouped options
        """
        groups = {}
        
        # Find option sections
        option_sections = re.findall(
            r'(?:options?|flags?):\s*\n((?:\s+[-\w].+\n?)+)',
            help_output,
            re.IGNORECASE | re.MULTILINE
        )
        
        for section in option_sections:
            options = []
            for line in section.split('\n'):
                line = line.strip()
                if line and line.startswith('-'):
                    options.append(line)
            
            if options:
                groups['Command Options'] = options
        
        return groups
    
    def _extract_pattern_name(self, usage_line: str) -> str:
        """
        Extract a descriptive name from a usage pattern.
        
        Args:
            usage_line: Usage pattern line
            
        Returns:
            str: Extracted pattern name
        """
        # Remove executable name and extract key operation
        parts = usage_line.split()
        if len(parts) > 1:
            # Look for the main command or operation
            for part in parts[1:]:
                if not part.startswith('-') and not part.startswith('<') and not part.startswith('['):
                    return part.capitalize()
        
        return "Main Operation"
    
    def _extract_subcommand_functionality(self, help_text: str) -> Optional[str]:
        """
        Extract the main functionality description from subcommand help.
        
        Args:
            help_text: Help text for subcommand
            
        Returns:
            Optional[str]: Functionality description
        """
        # Look for description patterns
        desc_patterns = [
            r'(?:description|summary):\s*(.+?)(?:\n\n|\n[A-Z]|$)',
            r'^(.+?)(?:\n\n|\nusage|\noptions)',  # First line/paragraph
        ]
        
        for pattern in desc_patterns:
            matches = re.findall(pattern, help_text, re.IGNORECASE | re.MULTILINE | re.DOTALL)
            for match in matches:
                desc = match.strip()
                if 20 <= len(desc) <= 200:  # Reasonable description length
                    return desc
        
        return None
    
    def _create_parameter_from_option(self, option_line: str) -> Optional[Parameter]:
        """
        Create a Parameter object from an option line.
        
        Args:
            option_line: Option line from help text
            
        Returns:
            Optional[Parameter]: Created parameter
        """
        # Parse option format: -f, --flag [description]
        match = re.match(r'(-\w+)(?:,\s*(--\w+))?\s+(.+)', option_line)
        if not match:
            return None
        
        short_flag = match.group(1)
        long_flag = match.group(2)
        description = match.group(3)
        
        # Use long flag name if available, otherwise short flag
        param_name = (long_flag or short_flag).lstrip('-')
        
        # Determine parameter type
        param_type = "boolean"  # Most CLI flags are boolean
        if any(indicator in description.lower() for indicator in ['<', 'value', 'number', 'file', 'path']):
            param_type = "string"
        
        return Parameter(
            name=param_name,
            type=param_type,
            description=description,
            required=False  # CLI options are typically optional
        )
    
    def _analyze_argument_patterns(self, app: Application, help_output: str) -> List[Capability]:
        """
        Analyze command-line argument patterns to infer capabilities.
        
        Args:
            app: Application being analyzed
            help_output: Help output to analyze
            
        Returns:
            List[Capability]: Argument-based capabilities
        """
        capabilities = []
        
        # Look for positional arguments
        arg_patterns = re.findall(
            r'<(\w+)>|(\w+)\s+\.\.\.',
            help_output
        )
        
        for pattern in arg_patterns:
            arg_name = pattern[0] or pattern[1]
            if arg_name:
                capability = self._create_capability(
                    app=app,
                    name=f"{arg_name.capitalize()} Processing",
                    category=self._categorize_capability(arg_name, ""),
                    interface_type=InterfaceType.COMMAND_LINE,
                    description=f"Processes {arg_name} arguments",
                    confidence=0.6
                )
                capabilities.append(capability)
        
        return capabilities