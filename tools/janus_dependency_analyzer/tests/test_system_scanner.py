"""
Unit tests for the SystemScanner orchestrator class.

This module tests the SystemScanner implementation to ensure it correctly:
- Detects the current platform
- Loads appropriate platform-specific scanners
- Performs full and incremental scans
- Handles errors gracefully
"""

import pytest
import platform
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

from ..scanners.system_scanner import SystemScannerImpl
from ..core.models import Platform, ScanResult, Application, ApplicationMetadata


class TestSystemScanner:
    """Test cases for SystemScanner implementation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.scanner = SystemScannerImpl()
    
    def test_platform_detection_windows(self):
        """Test platform detection for Windows."""
        with patch('platform.system', return_value='Windows'):
            platform_detected = self.scanner.detect_platform()
            assert platform_detected == Platform.WINDOWS
    
    def test_platform_detection_macos(self):
        """Test platform detection for macOS."""
        with patch('platform.system', return_value='Darwin'):
            platform_detected = self.scanner.detect_platform()
            assert platform_detected == Platform.MACOS
    
    def test_platform_detection_linux(self):
        """Test platform detection for Linux."""
        with patch('platform.system', return_value='Linux'):
            platform_detected = self.scanner.detect_platform()
            assert platform_detected == Platform.LINUX
    
    def test_platform_detection_unsupported(self):
        """Test platform detection for unsupported platform."""
        with patch('platform.system', return_value='FreeBSD'):
            with pytest.raises(RuntimeError, match="Unsupported platform"):
                self.scanner.detect_platform()
    
    def test_platform_detection_caching(self):
        """Test that platform detection results are cached."""
        with patch('platform.system', return_value='Linux') as mock_system:
            # First call
            platform1 = self.scanner.detect_platform()
            # Second call
            platform2 = self.scanner.detect_platform()
            
            assert platform1 == platform2 == Platform.LINUX
            # platform.system() should only be called once due to caching
            assert mock_system.call_count == 1
    
    @patch('janus_dependency_analyzer.scanners.system_scanner.SystemScannerImpl.detect_platform')
    def test_get_platform_scanner_windows(self, mock_detect):
        """Test getting Windows platform scanner."""
        mock_detect.return_value = Platform.WINDOWS
        
        with patch('janus_dependency_analyzer.scanners.windows_scanner.WindowsScanner') as mock_scanner:
            scanner_instance = Mock()
            mock_scanner.return_value = scanner_instance
            
            result = self.scanner.get_platform_scanner()
            
            assert result == scanner_instance
            mock_scanner.assert_called_once()
    
    @patch('janus_dependency_analyzer.scanners.system_scanner.SystemScannerImpl.detect_platform')
    def test_get_platform_scanner_macos(self, mock_detect):
        """Test getting macOS platform scanner."""
        mock_detect.return_value = Platform.MACOS
        
        with patch('janus_dependency_analyzer.scanners.macos_scanner.MacOSScanner') as mock_scanner:
            scanner_instance = Mock()
            mock_scanner.return_value = scanner_instance
            
            result = self.scanner.get_platform_scanner()
            
            assert result == scanner_instance
            mock_scanner.assert_called_once()
    
    @patch('janus_dependency_analyzer.scanners.system_scanner.SystemScannerImpl.detect_platform')
    def test_get_platform_scanner_linux(self, mock_detect):
        """Test getting Linux platform scanner."""
        mock_detect.return_value = Platform.LINUX
        
        with patch('janus_dependency_analyzer.scanners.linux_scanner.LinuxScanner') as mock_scanner:
            scanner_instance = Mock()
            mock_scanner.return_value = scanner_instance
            
            result = self.scanner.get_platform_scanner()
            
            assert result == scanner_instance
            mock_scanner.assert_called_once()
    
    @patch('janus_dependency_analyzer.scanners.system_scanner.SystemScannerImpl.detect_platform')
    def test_get_platform_scanner_fallback(self, mock_detect):
        """Test fallback to base scanner when platform scanner fails to import."""
        mock_detect.return_value = Platform.LINUX
        
        # Mock the Linux scanner import to fail
        with patch('janus_dependency_analyzer.scanners.linux_scanner.LinuxScanner', side_effect=ImportError("Module not found")):
            with patch('janus_dependency_analyzer.scanners.base.BasePlatformScanner') as mock_base:
                base_instance = Mock()
                mock_base.return_value = base_instance
                
                result = self.scanner.get_platform_scanner()
                
                assert result == base_instance
                mock_base.assert_called_once_with(Platform.LINUX)
    
    @patch('janus_dependency_analyzer.scanners.system_scanner.SystemScannerImpl.get_platform_scanner')
    @patch('janus_dependency_analyzer.scanners.system_scanner.SystemScannerImpl.detect_platform')
    def test_scan_full_success(self, mock_detect, mock_get_scanner):
        """Test successful full system scan."""
        mock_detect.return_value = Platform.LINUX
        
        # Create mock applications
        mock_app1 = Application(name="TestApp1", version="1.0")
        mock_app2 = Application(name="TestApp2", version="2.0")
        mock_applications = [mock_app1, mock_app2]
        
        # Mock platform scanner
        mock_platform_scanner = Mock()
        mock_platform_scanner.discover_applications.return_value = mock_applications
        mock_get_scanner.return_value = mock_platform_scanner
        
        # Perform scan
        result = self.scanner.scan_full()
        
        # Verify results
        assert isinstance(result, ScanResult)
        assert result.scan_type == "full"
        assert result.platform == Platform.LINUX
        assert result.total_applications == 2
        assert len(result.applications) == 2
        assert result.applications[0].name == "TestApp1"
        assert result.applications[1].name == "TestApp2"
        assert result.scan_end_time is not None
    
    @patch('janus_dependency_analyzer.scanners.system_scanner.SystemScannerImpl.get_platform_scanner')
    @patch('janus_dependency_analyzer.scanners.system_scanner.SystemScannerImpl.detect_platform')
    def test_scan_full_error_handling(self, mock_detect, mock_get_scanner):
        """Test error handling in full system scan."""
        mock_detect.return_value = Platform.LINUX
        
        # Mock platform scanner to raise an exception
        mock_platform_scanner = Mock()
        mock_platform_scanner.discover_applications.side_effect = Exception("Scanner error")
        mock_get_scanner.return_value = mock_platform_scanner
        
        # Perform scan
        result = self.scanner.scan_full()
        
        # Verify error handling
        assert isinstance(result, ScanResult)
        assert result.scan_type == "full"
        assert result.total_applications == 0
        assert len(result.errors) == 1
        assert "Scanner error" in result.errors[0]
    
    @patch('janus_dependency_analyzer.scanners.system_scanner.SystemScannerImpl.get_platform_scanner')
    @patch('janus_dependency_analyzer.scanners.system_scanner.SystemScannerImpl.detect_platform')
    @patch('janus_dependency_analyzer.scanners.system_scanner.SystemScannerImpl._has_application_changed')
    def test_scan_incremental_success(self, mock_has_changed, mock_detect, mock_get_scanner):
        """Test successful incremental system scan."""
        mock_detect.return_value = Platform.LINUX
        mock_has_changed.return_value = False  # No file system changes
        
        # Create mock applications with different timestamps
        now = datetime.now()
        old_time = now - timedelta(hours=2)
        recent_time = now - timedelta(minutes=30)
        
        mock_app1 = Application(name="OldApp", version="1.0")
        mock_app1.discovered_at = old_time
        
        mock_app2 = Application(name="RecentApp", version="2.0")
        mock_app2.discovered_at = recent_time
        
        mock_applications = [mock_app1, mock_app2]
        
        # Mock platform scanner
        mock_platform_scanner = Mock()
        mock_platform_scanner.discover_applications.return_value = mock_applications
        mock_get_scanner.return_value = mock_platform_scanner
        
        # Perform incremental scan with last scan time 1 hour ago
        last_scan_time = now - timedelta(hours=1)
        result = self.scanner.scan_incremental(last_scan_time)
        
        # Verify results - only recent app should be included
        assert isinstance(result, ScanResult)
        assert result.scan_type == "incremental"
        assert result.platform == Platform.LINUX
        assert result.total_applications == 1
        assert result.applications[0].name == "RecentApp"
    
    @patch('janus_dependency_analyzer.scanners.system_scanner.SystemScannerImpl.get_platform_scanner')
    @patch('janus_dependency_analyzer.scanners.system_scanner.SystemScannerImpl.detect_platform')
    def test_scan_incremental_error_handling(self, mock_detect, mock_get_scanner):
        """Test error handling in incremental system scan."""
        mock_detect.return_value = Platform.LINUX
        
        # Mock platform scanner to raise an exception
        mock_platform_scanner = Mock()
        mock_platform_scanner.discover_applications.side_effect = Exception("Scanner error")
        mock_get_scanner.return_value = mock_platform_scanner
        
        # Perform incremental scan
        last_scan_time = datetime.now() - timedelta(hours=1)
        result = self.scanner.scan_incremental(last_scan_time)
        
        # Verify error handling
        assert isinstance(result, ScanResult)
        assert result.scan_type == "incremental"
        assert result.total_applications == 0
        assert len(result.errors) == 1
        assert "Scanner error" in result.errors[0]
    
    @patch('janus_dependency_analyzer.scanners.system_scanner.SystemScannerImpl._has_application_changed')
    def test_detect_changes_new_applications(self, mock_has_changed):
        """Test change detection for newly discovered applications."""
        now = datetime.now()
        last_scan_time = now - timedelta(hours=1)
        
        # Mock the file system check to return False (no file changes)
        mock_has_changed.return_value = False
        
        # Create applications - one old, one new
        old_app = Application(name="OldApp", version="1.0")
        old_app.discovered_at = now - timedelta(hours=2)
        
        new_app = Application(name="NewApp", version="1.0")
        new_app.discovered_at = now - timedelta(minutes=30)
        
        applications = [old_app, new_app]
        
        # Test change detection
        changed = self.scanner._detect_changes(applications, last_scan_time)
        
        # Only the new app should be detected as changed
        assert len(changed) == 1
        assert changed[0].name == "NewApp"
    
    def test_detect_changes_inaccessible_applications(self):
        """Test change detection includes previously inaccessible applications."""
        now = datetime.now()
        last_scan_time = now - timedelta(hours=1)
        
        # Create an old but inaccessible application
        inaccessible_app = Application(name="InaccessibleApp", version="1.0")
        inaccessible_app.discovered_at = now - timedelta(hours=2)
        inaccessible_app.is_accessible = False
        inaccessible_app.access_error = "Permission denied"
        
        applications = [inaccessible_app]
        
        # Test change detection
        changed = self.scanner._detect_changes(applications, last_scan_time)
        
        # The inaccessible app should be included for re-checking
        assert len(changed) == 1
        assert changed[0].name == "InaccessibleApp"
    
    @patch('pathlib.Path.exists')
    @patch('pathlib.Path.stat')
    def test_has_application_changed_executable_modified(self, mock_stat, mock_exists):
        """Test application change detection based on executable modification time."""
        now = datetime.now()
        last_scan_time = now - timedelta(hours=1)
        
        # Mock file exists and modification time
        mock_exists.return_value = True
        mock_stat_result = Mock()
        mock_stat_result.st_mtime = (now - timedelta(minutes=30)).timestamp()
        mock_stat.return_value = mock_stat_result
        
        app = Application(name="TestApp", version="1.0")
        
        # Test change detection
        changed = self.scanner._has_application_changed(app, last_scan_time)
        
        # Should detect change since modification time is after last scan
        assert changed is True
    
    @patch('pathlib.Path.exists')
    @patch('pathlib.Path.stat')
    def test_has_application_changed_no_change(self, mock_stat, mock_exists):
        """Test application change detection when no changes occurred."""
        now = datetime.now()
        last_scan_time = now - timedelta(hours=1)
        
        # Mock file exists and old modification time
        mock_exists.return_value = True
        mock_stat_result = Mock()
        mock_stat_result.st_mtime = (now - timedelta(hours=2)).timestamp()
        mock_stat.return_value = mock_stat_result
        
        app = Application(name="TestApp", version="1.0")
        
        # Test change detection
        changed = self.scanner._has_application_changed(app, last_scan_time)
        
        # Should not detect change since modification time is before last scan
        assert changed is False
    
    @patch('pathlib.Path.exists')
    @patch('pathlib.Path.stat')
    def test_has_application_changed_permission_error(self, mock_stat, mock_exists):
        """Test application change detection handles permission errors gracefully."""
        now = datetime.now()
        last_scan_time = now - timedelta(hours=1)
        
        # Mock file exists but stat raises permission error
        mock_exists.return_value = True
        mock_stat.side_effect = PermissionError("Access denied")
        
        app = Application(name="TestApp", version="1.0")
        
        # Test change detection
        changed = self.scanner._has_application_changed(app, last_scan_time)
        
        # Should assume change when unable to check (safe default)
        assert changed is True