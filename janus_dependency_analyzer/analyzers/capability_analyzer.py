"""
Main capability analyzer implementation.

This module provides the primary CapabilityAnalyzer implementation that
orchestrates multiple analysis strategies to identify application capabilities.
"""

import logging
from typing import List, Dict, Set, Any
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..core.interfaces import CapabilityAnalyzer, AnalysisStrategy
from ..core.models import Application, Capability, CapabilityCategory


logger = logging.getLogger(__name__)


class CapabilityAnalyzerImpl(CapabilityAnalyzer):
    """
    Main capability analyzer implementation.
    
    This class orchestrates multiple analysis strategies to comprehensively
    analyze applications and identify their capabilities with confidence scoring.
    """
    
    def __init__(self, enable_cache: bool = True):
        """Initialize the capability analyzer with default strategies."""
        self.logger = logging.getLogger(__name__)
        self._strategies: List[AnalysisStrategy] = []
        self._cache = None
        self._cache_enabled = enable_cache
        self._initialize_strategies()
        
        if enable_cache:
            try:
                from ..cache.cache_manager import CacheManager
                self._cache = CacheManager()
                self.logger.info("Cache enabled")
            except Exception as e:
                self.logger.warning(f"Failed to initialize cache: {e}")
                self._cache = None
    
    def _initialize_strategies(self) -> None:
        """Initialize the default analysis strategies."""
        try:
            from .strategies import (
                DocumentationAnalysisStrategy,
                HelpTextAnalysisStrategy,
                CommandLineInterfaceStrategy,
                APIEndpointStrategy,
                AppxManifestStrategy,
                OxpeckerStrategy,
                FileAssociationStrategy,
            )
            
            self._strategies = [
                # Ground-truth registry sources first (highest confidence)
                AppxManifestStrategy(),       # Windows Store manifest
                FileAssociationStrategy(),    # HKCR shell associations (Win32/traditional)
                # Source-code analysis
                OxpeckerStrategy(),
                # General-purpose strategies
                DocumentationAnalysisStrategy(),
                HelpTextAnalysisStrategy(),
                CommandLineInterfaceStrategy(),
                APIEndpointStrategy(),
            ]
            
            self.logger.info(f"Initialized {len(self._strategies)} analysis strategies")
            
        except ImportError as e:
            self.logger.error(f"Failed to import analysis strategies: {e}")
            self._strategies = []
    
    def analyze_application(self, app: Application, enable_early_exit: bool = True) -> List[Capability]:
        """
        Analyze an application to identify its capabilities.
        Strategies run in parallel for performance.
        Uses caching to avoid re-analyzing unchanged applications.
        
        Args:
            app: Application to analyze
            enable_early_exit: If True, stop analyzing when high-confidence results found
        """
        self.logger.info(f"Analyzing application: {app.name}")
        
        if not app.is_accessible:
            self.logger.warning(f"Application {app.name} is not accessible: {app.access_error}")
            return []
        
        # Check cache first
        if self._cache:
            cached_capabilities = self._cache.get(app)
            if cached_capabilities is not None:
                self.logger.info(f"Using cached results for {app.name} ({len(cached_capabilities)} capabilities)")
                return cached_capabilities
        
        # Determine which strategies can analyze this app first (fast check)
        applicable = [s for s in self._strategies if self._can_analyze_safe(s, app)]
        
        if not applicable:
            return []
        
        all_capabilities: List[Capability] = []
        
        # Run applicable strategies in parallel (I/O bound — subprocess + filesystem)
        with ThreadPoolExecutor(max_workers=min(4, len(applicable))) as executor:
            future_to_strategy = {
                executor.submit(self._run_strategy, s, app): s
                for s in applicable
            }
            for future in as_completed(future_to_strategy):
                strategy = future_to_strategy[future]
                try:
                    capabilities = future.result(timeout=30)
                    for capability in capabilities:
                        capability.detection_method = strategy.get_strategy_name()
                        capability.confidence_score *= strategy.get_confidence_factor()
                    all_capabilities.extend(capabilities)
                    self.logger.debug(
                        f"{strategy.get_strategy_name()} found {len(capabilities)} capabilities"
                    )
                    
                    # Early exit optimization: stop if we have high-confidence results
                    if enable_early_exit and len(all_capabilities) >= 3:
                        high_conf = [c for c in all_capabilities if c.confidence_score > 0.8]
                        if len(high_conf) >= 3:
                            self.logger.info(
                                f"Early exit for {app.name}: {len(high_conf)} high-confidence "
                                f"capabilities found"
                            )
                            # Cancel remaining futures
                            for f in future_to_strategy:
                                if not f.done():
                                    f.cancel()
                            break
                    
                except Exception as e:
                    self.logger.error(
                        f"Error applying {strategy.get_strategy_name()} to {app.name}: {e}"
                    )
        
        merged = self.merge_capabilities(all_capabilities)
        self.logger.info(f"Analysis complete for {app.name}: {len(merged)} capabilities identified")
        
        # Store in cache
        if self._cache:
            self._cache.put(app, merged)
        
        return merged
    
    def _can_analyze_safe(self, strategy: AnalysisStrategy, app: Application) -> bool:
        """Call can_analyze with error isolation."""
        try:
            return strategy.can_analyze(app)
        except Exception as e:
            self.logger.debug(f"{strategy.get_strategy_name()}.can_analyze failed: {e}")
            return False
    
    def _run_strategy(self, strategy: AnalysisStrategy, app: Application) -> List[Capability]:
        """Run a single strategy with error isolation."""
        try:
            return strategy.extract_capabilities(app)
        except Exception as e:
            self.logger.error(f"{strategy.get_strategy_name()} failed on {app.name}: {e}")
            return []
    
    def get_analysis_strategies(self) -> List[AnalysisStrategy]:
        """
        Get all available analysis strategies.
        
        Returns:
            List[AnalysisStrategy]: Available analysis strategies
        """
        return self._strategies.copy()
    
    def merge_capabilities(self, capabilities: List[Capability]) -> List[Capability]:
        """
        Merge overlapping capabilities and calculate confidence scores.
        
        This method identifies capabilities that are likely describing the same
        functionality and merges them into a single capability with combined
        confidence and information.
        
        Args:
            capabilities: List of capabilities to merge
            
        Returns:
            List[Capability]: Merged and scored capabilities
        """
        if not capabilities:
            return []
        
        # Group capabilities by name and category for potential merging
        capability_groups = defaultdict(list)
        
        for capability in capabilities:
            # Create a key based on normalized name and category
            key = (
                capability.name.lower().strip(),
                capability.category,
                capability.interface_type
            )
            capability_groups[key].append(capability)
        
        merged_capabilities = []
        
        for group_key, group_capabilities in capability_groups.items():
            if len(group_capabilities) == 1:
                # Single capability, no merging needed
                merged_capabilities.append(group_capabilities[0])
            else:
                # Multiple capabilities with same name/category, merge them
                merged_capability = self._merge_capability_group(group_capabilities)
                merged_capabilities.append(merged_capability)
        
        # Sort by confidence score (highest first)
        merged_capabilities.sort(key=lambda c: c.confidence_score, reverse=True)
        
        return merged_capabilities
    
    def _merge_capability_group(self, capabilities: List[Capability]) -> Capability:
        """
        Merge a group of similar capabilities into a single capability.
        
        Args:
            capabilities: List of capabilities to merge
            
        Returns:
            Capability: Merged capability
        """
        if not capabilities:
            raise ValueError("Cannot merge empty capability group")
        
        if len(capabilities) == 1:
            return capabilities[0]
        
        # Use the first capability as the base
        base_capability = capabilities[0]
        
        # Combine confidence scores using weighted average
        total_confidence = sum(c.confidence_score for c in capabilities)
        avg_confidence = total_confidence / len(capabilities)
        
        # Boost confidence slightly for multiple detections
        confidence_boost = min(0.2, (len(capabilities) - 1) * 0.05)
        final_confidence = min(1.0, avg_confidence + confidence_boost)
        
        # Combine descriptions — preserve insertion order, deduplicate
        seen_desc: set = set()
        unique_descriptions = []
        for c in capabilities:
            if c.description and c.description not in seen_desc:
                seen_desc.add(c.description)
                unique_descriptions.append(c.description)
        combined_description = "; ".join(unique_descriptions) if unique_descriptions else base_capability.description
        
        # Combine parameters (remove duplicates)
        all_parameters = []
        seen_params = set()
        
        for capability in capabilities:
            for param in capability.parameters:
                param_key = (param.name, param.type)
                if param_key not in seen_params:
                    all_parameters.append(param)
                    seen_params.add(param_key)
        
        # Combine examples
        all_examples = []
        for capability in capabilities:
            all_examples.extend(capability.examples)
        unique_examples = list(set(all_examples))
        
        # Combine supported formats
        all_formats = []
        for capability in capabilities:
            all_formats.extend(capability.supported_formats)
        unique_formats = list(set(all_formats))
        
        # Combine detection methods
        detection_methods = [c.detection_method for c in capabilities if c.detection_method]
        combined_detection_method = ", ".join(set(detection_methods))
        
        # Create merged capability
        merged_capability = Capability(
            id=base_capability.id,  # Keep the first ID
            application_id=base_capability.application_id,
            name=base_capability.name,
            category=base_capability.category,
            description=combined_description,
            interface_type=base_capability.interface_type,
            parameters=all_parameters,
            confidence_score=final_confidence,
            detection_method=combined_detection_method,
            examples=unique_examples,
            documentation_url=base_capability.documentation_url,
            supported_formats=unique_formats
        )
        
        return merged_capability
    
    def add_strategy(self, strategy: AnalysisStrategy) -> None:
        """
        Add a new analysis strategy.
        
        Args:
            strategy: Analysis strategy to add
        """
        if strategy not in self._strategies:
            self._strategies.append(strategy)
            self.logger.info(f"Added analysis strategy: {strategy.get_strategy_name()}")
    
    def remove_strategy(self, strategy_name: str) -> bool:
        """
        Remove an analysis strategy by name.
        
        Args:
            strategy_name: Name of the strategy to remove
            
        Returns:
            bool: True if strategy was removed, False if not found
        """
        for i, strategy in enumerate(self._strategies):
            if strategy.get_strategy_name() == strategy_name:
                removed_strategy = self._strategies.pop(i)
                self.logger.info(f"Removed analysis strategy: {removed_strategy.get_strategy_name()}")
                return True
        
        self.logger.warning(f"Strategy not found for removal: {strategy_name}")
        return False
    
    def get_capability_statistics(self, capabilities: List[Capability]) -> Dict[str, int]:
        """
        Get statistics about detected capabilities.
        
        Args:
            capabilities: List of capabilities to analyze
            
        Returns:
            Dict[str, int]: Statistics about capabilities
        """
        stats = {
            "total_capabilities": len(capabilities),
            "high_confidence": len([c for c in capabilities if c.confidence_score >= 0.8]),
            "medium_confidence": len([c for c in capabilities if 0.5 <= c.confidence_score < 0.8]),
            "low_confidence": len([c for c in capabilities if c.confidence_score < 0.5]),
        }
        
        # Count by category
        category_counts = defaultdict(int)
        for capability in capabilities:
            category_counts[capability.category.value] += 1
        
        stats.update({f"category_{k}": v for k, v in category_counts.items()})
        
        # Count by interface type
        interface_counts = defaultdict(int)
        for capability in capabilities:
            interface_counts[capability.interface_type.value] += 1
        
        stats.update({f"interface_{k}": v for k, v in interface_counts.items()})
        
        return stats
    
    def save_cache(self) -> None:
        """Save the cache to disk."""
        if self._cache:
            self._cache.save()
            self.logger.info("Cache saved")
        
        # Also save subprocess cache
        try:
            from ..cache.subprocess_cache import get_subprocess_cache
            subprocess_cache = get_subprocess_cache()
            subprocess_cache.save()
        except Exception as e:
            self.logger.warning(f"Failed to save subprocess cache: {e}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dict with cache statistics or empty dict if cache disabled
        """
        if self._cache:
            return self._cache.get_stats()
        return {}
    
    def clear_cache(self) -> None:
        """Clear the cache."""
        if self._cache:
            self._cache.clear()
            self.logger.info("Cache cleared")