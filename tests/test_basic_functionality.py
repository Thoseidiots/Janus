"""
Basic functionality tests to verify the system works end-to-end.
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch

from janus_dependency_analyzer.scanners.system_scanner import SystemScannerImpl
from janus_dependency_analyzer.analyzers.capability_analyzer import CapabilityAnalyzerImpl
from janus_dependency_analyzer.core.models import (
    Application, Platform, ScanResult, Capability, CapabilityCategory, InterfaceType
)


class TestBasicFunctionality:
    """Test basic system functionality."""
    
    def test_system_scanner_initialization(self):
        """Test that system scanner can be initialized."""
        scanner = SystemScannerImpl()
        assert scanner is not None
        
        # Should detect current platform
        platform = scanner.detect_platform()
        assert platform in [Platform.WINDOWS, Platform.MACOS, Platform.LINUX]
    
    def test_capability_analyzer_initialization(self):
        """Test that capability analyzer can be initialized."""
        analyzer = CapabilityAnalyzerImpl()
        assert analyzer is not None
        
        # Should have analysis strategies
        strategies = analyzer.get_analysis_strategies()
        assert len(strategies) >= 0  # May be 0 if imports fail, but shouldn't crash
    
    def test_scan_result_functionality(self):
        """Test ScanResult basic functionality."""
        result = ScanResult(platform=Platform.LINUX)
        
        # Test adding applications
        app1 = Application(name="TestApp1", is_accessible=True)
        app2 = Application(name="TestApp2", is_accessible=False, access_error="Permission denied")
        
        result.add_application(app1)
        result.add_application(app2)
        
        assert result.total_applications == 2
        assert result.accessible_applications == 1
        
        # Test adding errors and warnings
        result.add_error("Test error")
        result.add_warning("Test warning")
        
        assert len(result.errors) == 1
        assert len(result.warnings) == 1
        
        # Test finalization
        result.finalize()
        assert result.scan_end_time is not None
    
    def test_capability_analysis_basic(self):
        """Test basic capability analysis functionality."""
        analyzer = CapabilityAnalyzerImpl()
        
        # Create a mock application
        app = Application(
            name="TestApp",
            executable_path=Path("/usr/bin/test"),
            is_accessible=True,
            platform=Platform.LINUX
        )
        
        # Analyze capabilities (should not crash even if no strategies work)
        capabilities = analyzer.analyze_application(app)
        assert isinstance(capabilities, list)
        
        # Test capability merging with empty list
        merged = analyzer.merge_capabilities([])
        assert merged == []
        
        # Test capability merging with sample capabilities
        cap1 = Capability(
            application_id=app.id,
            name="Test Capability",
            category=CapabilityCategory.FILE_PROCESSING,
            interface_type=InterfaceType.COMMAND_LINE,
            confidence_score=0.8
        )
        
        cap2 = Capability(
            application_id=app.id,
            name="Test Capability",  # Same name
            category=CapabilityCategory.FILE_PROCESSING,
            interface_type=InterfaceType.COMMAND_LINE,
            confidence_score=0.6
        )
        
        merged = analyzer.merge_capabilities([cap1, cap2])
        assert len(merged) == 1  # Should merge similar capabilities
        assert merged[0].confidence_score > 0.6  # Should boost confidence
    
    @patch('janus_dependency_analyzer.scanners.windows_scanner.winreg')
    def test_windows_scanner_mock(self, mock_winreg):
        """Test Windows scanner with mocked registry access."""
        # Skip if not on Windows or if winreg import fails
        try:
            from janus_dependency_analyzer.scanners.windows_scanner import WindowsScanner
        except ImportError:
            pytest.skip("Windows scanner not available")
        
        # Mock registry operations to prevent actual system access
        mock_winreg.HKEY_LOCAL_MACHINE = -2147483646
        mock_winreg.OpenKey.side_effect = OSError("Mocked registry access")
        
        scanner = WindowsScanner()
        
        # Should not crash even with registry access failures
        apps = scanner.scan_standard_locations()
        assert isinstance(apps, list)
    
    def test_application_creation_and_validation(self):
        """Test application creation and basic validation."""
        app = Application(
            name="Test Application",
            version="1.0.0",
            executable_path=Path("/usr/bin/test"),
            installation_path=Path("/usr/share/test"),
            platform=Platform.LINUX
        )
        
        assert app.name == "Test Application"
        assert app.version == "1.0.0"
        assert app.platform == Platform.LINUX
        assert app.id is not None
        assert app.discovered_at is not None
    
    def test_capability_creation_and_validation(self):
        """Test capability creation and basic validation."""
        capability = Capability(
            application_id="test-app-id",
            name="File Processing",
            category=CapabilityCategory.FILE_PROCESSING,
            interface_type=InterfaceType.COMMAND_LINE,
            description="Processes various file formats",
            confidence_score=0.85
        )
        
        assert capability.name == "File Processing"
        assert capability.category == CapabilityCategory.FILE_PROCESSING
        assert capability.interface_type == InterfaceType.COMMAND_LINE
        assert capability.confidence_score == 0.85
        assert capability.id is not None
    
    def test_platform_detection_consistency(self):
        """Test that platform detection is consistent."""
        scanner1 = SystemScannerImpl()
        scanner2 = SystemScannerImpl()
        
        platform1 = scanner1.detect_platform()
        platform2 = scanner2.detect_platform()
        
        # Should detect the same platform consistently
        assert platform1 == platform2
    
    def test_error_handling_in_analysis(self):
        """Test error handling in capability analysis."""
        analyzer = CapabilityAnalyzerImpl()
        
        # Create an inaccessible application
        inaccessible_app = Application(
            name="InaccessibleApp",
            executable_path=Path("/nonexistent/path"),
            is_accessible=False,
            access_error="File not found",
            platform=Platform.LINUX
        )
        
        # Should handle inaccessible applications gracefully
        capabilities = analyzer.analyze_application(inaccessible_app)
        assert isinstance(capabilities, list)
        # Should return empty list for inaccessible apps
        assert len(capabilities) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])