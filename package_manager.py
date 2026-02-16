“””
Janus Package Manager
Manages dependencies, libraries, and code modules autonomously
Works offline with local package registry
“””

import json
import os
import hashlib
import shutil
import tarfile
import zipfile
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, asdict
import subprocess

@dataclass
class Package:
“”“Represents a code package/library”””
name: str
version: str
description: str
author: str
language: str
files: Dict[str, str]  # relative_path -> content
dependencies: List[str]  # [“package_name@version”]
install_script: Optional[str] = None
created_at: Optional[str] = None
checksum: Optional[str] = None

@dataclass
class PackageMetadata:
“”“Metadata for installed package”””
name: str
version: str
install_path: str
installed_at: str
size_bytes: int
dependencies: List[str]

class LocalPackageRegistry:
“”“Local registry of available packages”””

```
def __init__(self, registry_path: str = "/tmp/janus_packages"):
    self.registry_path = Path(registry_path)
    self.registry_path.mkdir(exist_ok=True, parents=True)
    
    self.packages_dir = self.registry_path / "packages"
    self.packages_dir.mkdir(exist_ok=True)
    
    self.index_file = self.registry_path / "index.json"
    self.index = self._load_index()

def _load_index(self) -> Dict:
    """Loads package index"""
    if self.index_file.exists():
        with open(self.index_file, 'r') as f:
            return json.load(f)
    return {"packages": {}, "updated_at": datetime.now().isoformat()}

def _save_index(self):
    """Saves package index"""
    self.index["updated_at"] = datetime.now().isoformat()
    with open(self.index_file, 'w') as f:
        json.dump(self.index, f, indent=2)

def publish_package(self, package: Package) -> bool:
    """Publishes a package to the local registry"""
    package_key = f"{package.name}@{package.version}"
    
    # Create package directory
    package_dir = self.packages_dir / package.name / package.version
    package_dir.mkdir(exist_ok=True, parents=True)
    
    # Write package files
    for filepath, content in package.files.items():
        file_path = package_dir / filepath
        file_path.parent.mkdir(exist_ok=True, parents=True)
        with open(file_path, 'w') as f:
            f.write(content)
    
    # Calculate checksum
    checksum = self._calculate_package_checksum(package)
    package.checksum = checksum
    package.created_at = datetime.now().isoformat()
    
    # Write package.json metadata
    with open(package_dir / "package.json", 'w') as f:
        json.dump(asdict(package), f, indent=2)
    
    # Update index
    if package.name not in self.index["packages"]:
        self.index["packages"][package.name] = {}
    
    self.index["packages"][package.name][package.version] = {
        "description": package.description,
        "author": package.author,
        "language": package.language,
        "dependencies": package.dependencies,
        "checksum": checksum,
        "published_at": package.created_at,
        "path": str(package_dir)
    }
    
    self._save_index()
    return True

def get_package(self, name: str, version: str = "latest") -> Optional[Package]:
    """Retrieves a package from registry"""
    if name not in self.index["packages"]:
        return None
    
    versions = self.index["packages"][name]
    
    if version == "latest":
        # Get most recent version
        version = max(versions.keys())
    
    if version not in versions:
        return None
    
    package_info = versions[version]
    package_path = Path(package_info["path"])
    
    if not package_path.exists():
        return None
    
    # Load package.json
    with open(package_path / "package.json", 'r') as f:
        package_data = json.load(f)
    
    return Package(**package_data)

def search_packages(self, query: str) -> List[Dict]:
    """Searches for packages by name or description"""
    results = []
    
    query_lower = query.lower()
    
    for pkg_name, versions in self.index["packages"].items():
        # Check if query matches package name
        if query_lower in pkg_name.lower():
            for version, info in versions.items():
                results.append({
                    "name": pkg_name,
                    "version": version,
                    "description": info["description"],
                    "language": info["language"]
                })
        else:
            # Check descriptions
            for version, info in versions.items():
                if query_lower in info["description"].lower():
                    results.append({
                        "name": pkg_name,
                        "version": version,
                        "description": info["description"],
                        "language": info["language"]
                    })
    
    return results

def list_all_packages(self) -> List[Dict]:
    """Lists all available packages"""
    packages = []
    
    for pkg_name, versions in self.index["packages"].items():
        for version, info in versions.items():
            packages.append({
                "name": pkg_name,
                "version": version,
                "description": info["description"],
                "language": info["language"],
                "published_at": info["published_at"]
            })
    
    return packages

def _calculate_package_checksum(self, package: Package) -> str:
    """Calculates checksum for package integrity"""
    content = json.dumps(asdict(package), sort_keys=True)
    return hashlib.sha256(content.encode()).hexdigest()
```

class PackageInstaller:
“”“Installs and manages installed packages”””

```
def __init__(self, install_root: str = "/tmp/janus_installed"):
    self.install_root = Path(install_root)
    self.install_root.mkdir(exist_ok=True, parents=True)
    
    self.installed_file = self.install_root / "installed.json"
    self.installed = self._load_installed()

def _load_installed(self) -> Dict[str, PackageMetadata]:
    """Loads installed packages registry"""
    if self.installed_file.exists():
        with open(self.installed_file, 'r') as f:
            data = json.load(f)
            return {
                k: PackageMetadata(**v)
                for k, v in data.items()
            }
    return {}

def _save_installed(self):
    """Saves installed packages registry"""
    data = {k: asdict(v) for k, v in self.installed.items()}
    with open(self.installed_file, 'w') as f:
        json.dump(data, f, indent=2)

def install_package(self, 
                   package: Package, 
                   registry: LocalPackageRegistry,
                   force: bool = False) -> Tuple[bool, str]:
    """Installs a package and its dependencies"""
    package_key = f"{package.name}@{package.version}"
    
    # Check if already installed
    if package_key in self.installed and not force:
        return True, f"Package {package_key} already installed"
    
    # Resolve and install dependencies first
    for dep in package.dependencies:
        dep_name, dep_version = self._parse_dependency(dep)
        dep_package = registry.get_package(dep_name, dep_version)
        
        if not dep_package:
            return False, f"Dependency not found: {dep}"
        
        success, message = self.install_package(dep_package, registry, force)
        if not success:
            return False, f"Failed to install dependency {dep}: {message}"
    
    # Install the package
    install_path = self.install_root / package.name / package.version
    install_path.mkdir(exist_ok=True, parents=True)
    
    total_size = 0
    
    # Copy all files
    for filepath, content in package.files.items():
        file_path = install_path / filepath
        file_path.parent.mkdir(exist_ok=True, parents=True)
        
        with open(file_path, 'w') as f:
            f.write(content)
        
        total_size += len(content.encode())
    
    # Run install script if present
    if package.install_script:
        try:
            script_path = install_path / "install.sh"
            with open(script_path, 'w') as f:
                f.write(package.install_script)
            
            os.chmod(script_path, 0o755)
            
            result = subprocess.run(
                [str(script_path)],
                cwd=str(install_path),
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode != 0:
                return False, f"Install script failed: {result.stderr}"
        
        except Exception as e:
            return False, f"Install script error: {e}"
    
    # Record installation
    metadata = PackageMetadata(
        name=package.name,
        version=package.version,
        install_path=str(install_path),
        installed_at=datetime.now().isoformat(),
        size_bytes=total_size,
        dependencies=package.dependencies
    )
    
    self.installed[package_key] = metadata
    self._save_installed()
    
    return True, f"Successfully installed {package_key}"

def uninstall_package(self, name: str, version: str) -> Tuple[bool, str]:
    """Uninstalls a package"""
    package_key = f"{name}@{version}"
    
    if package_key not in self.installed:
        return False, f"Package {package_key} not installed"
    
    metadata = self.installed[package_key]
    install_path = Path(metadata.install_path)
    
    # Check if other packages depend on this
    dependents = []
    for pkg_key, pkg_meta in self.installed.items():
        if package_key in pkg_meta.dependencies:
            dependents.append(pkg_key)
    
    if dependents:
        return False, f"Cannot uninstall: required by {', '.join(dependents)}"
    
    # Remove installation directory
    if install_path.exists():
        shutil.rmtree(install_path)
    
    # Remove from registry
    del self.installed[package_key]
    self._save_installed()
    
    return True, f"Successfully uninstalled {package_key}"

def list_installed(self) -> List[Dict]:
    """Lists all installed packages"""
    return [
        {
            "name": meta.name,
            "version": meta.version,
            "installed_at": meta.installed_at,
            "size_mb": meta.size_bytes / (1024 * 1024),
            "dependencies": len(meta.dependencies)
        }
        for meta in self.installed.values()
    ]

def get_package_path(self, name: str, version: str = "latest") -> Optional[str]:
    """Gets installation path for a package"""
    if version == "latest":
        # Find latest installed version
        installed_versions = [
            (k, v) for k, v in self.installed.items()
            if v.name == name
        ]
        if not installed_versions:
            return None
        # Sort by version and get latest
        latest = max(installed_versions, key=lambda x: x[1].version)
        return latest[1].install_path
    
    package_key = f"{name}@{version}"
    if package_key in self.installed:
        return self.installed[package_key].install_path
    
    return None

def _parse_dependency(self, dep: str) -> Tuple[str, str]:
    """Parses dependency string like 'package@1.0.0'"""
    if '@' in dep:
        name, version = dep.rsplit('@', 1)
        return name, version
    return dep, "latest"
```

class PackageBuilder:
“”“Builds packages from source code”””

```
def __init__(self):
    self.templates = self._create_templates()

def _create_templates(self) -> Dict:
    """Creates package templates for different languages"""
    return {
        'python': {
            'structure': ['src/', 'tests/', 'README.md', 'setup.py'],
            'entry_file': 'src/__init__.py'
        },
        'javascript': {
            'structure': ['src/', 'test/', 'README.md', 'package.json'],
            'entry_file': 'src/index.js'
        },
        'rust': {
            'structure': ['src/', 'tests/', 'README.md', 'Cargo.toml'],
            'entry_file': 'src/lib.rs'
        }
    }

def create_package_from_code(self,
                             name: str,
                             version: str,
                             description: str,
                             author: str,
                             language: str,
                             source_files: Dict[str, str],
                             dependencies: List[str] = None) -> Package:
    """Creates a package from source code"""
    
    if dependencies is None:
        dependencies = []
    
    # Add any auto-detected dependencies
    detected_deps = self._detect_dependencies(source_files, language)
    all_deps = list(set(dependencies + detected_deps))
    
    package = Package(
        name=name,
        version=version,
        description=description,
        author=author,
        language=language,
        files=source_files,
        dependencies=all_deps
    )
    
    return package

def _detect_dependencies(self, source_files: Dict[str, str], language: str) -> List[str]:
    """Auto-detects dependencies from source code"""
    dependencies = set()
    
    if language == 'python':
        for content in source_files.values():
            # Find import statements
            lines = content.split('\n')
            for line in lines:
                line = line.strip()
                if line.startswith('import ') or line.startswith('from '):
                    # Extract module name
                    parts = line.split()
                    if len(parts) >= 2:
                        module = parts[1].split('.')[0]
                        # Filter out standard library
                        if module not in ['os', 'sys', 'json', 'math', 're', 'time', 'datetime']:
                            dependencies.add(module)
    
    elif language == 'javascript':
        for content in source_files.values():
            # Find require/import statements
            import_pattern = r'(?:require|import)\s*\([\'"]([^\'"]+)[\'"]\)'
            matches = __import__('re').findall(import_pattern, content)
            dependencies.update(matches)
    
    return list(dependencies)

def build_from_directory(self, directory: Path, metadata: Dict) -> Package:
    """Builds a package from a directory structure"""
    files = {}
    
    for root, dirs, filenames in os.walk(directory):
        for filename in filenames:
            filepath = Path(root) / filename
            relative_path = filepath.relative_to(directory)
            
            # Skip hidden files and build artifacts
            if any(part.startswith('.') for part in relative_path.parts):
                continue
            if '__pycache__' in str(relative_path):
                continue
            
            try:
                with open(filepath, 'r') as f:
                    files[str(relative_path)] = f.read()
            except UnicodeDecodeError:
                # Skip binary files
                continue
    
    return self.create_package_from_code(
        name=metadata['name'],
        version=metadata['version'],
        description=metadata['description'],
        author=metadata['author'],
        language=metadata['language'],
        source_files=files,
        dependencies=metadata.get('dependencies', [])
    )
```

class JanusPackageManager:
“”“Main package manager for Janus”””

```
def __init__(self):
    self.registry = LocalPackageRegistry()
    self.installer = PackageInstaller()
    self.builder = PackageBuilder()
    
    # Bootstrap with some built-in packages
    self._bootstrap_packages()

def _bootstrap_packages(self):
    """Creates some built-in utility packages"""
    # Math utilities package
    math_utils = Package(
        name="janus-math-utils",
        version="1.0.0",
        description="Mathematical utility functions for Janus",
        author="Janus Core",
        language="python",
        files={
            "math_utils.py": """
```

import math

def gcd(a, b):
‘’‘Greatest common divisor’’’
while b:
a, b = b, a % b
return a

def lcm(a, b):
‘’‘Least common multiple’’’
return abs(a * b) // gcd(a, b)

def is_prime(n):
‘’‘Check if number is prime’’’
if n < 2:
return False
for i in range(2, int(math.sqrt(n)) + 1):
if n % i == 0:
return False
return True

def factorial(n):
‘’‘Calculate factorial’’’
if n <= 1:
return 1
return n * factorial(n - 1)
“””
},
dependencies=[]
)

```
    # Only publish if not already in registry
    if "janus-math-utils" not in self.registry.index["packages"]:
        self.registry.publish_package(math_utils)
    
    # String utilities package
    string_utils = Package(
        name="janus-string-utils",
        version="1.0.0",
        description="String manipulation utilities for Janus",
        author="Janus Core",
        language="python",
        files={
            "string_utils.py": """
```

def reverse(s):
‘’‘Reverse a string’’’
return s[::-1]

def is_palindrome(s):
‘’‘Check if string is palindrome’’’
s = s.lower().replace(’ ’, ‘’)
return s == s[::-1]

def count_words(s):
‘’‘Count words in string’’’
return len(s.split())

def to_title_case(s):
‘’‘Convert to title case’’’
return ’ ’.join(word.capitalize() for word in s.split())
“””
},
dependencies=[]
)

```
    if "janus-string-utils" not in self.registry.index["packages"]:
        self.registry.publish_package(string_utils)

def search(self, query: str) -> List[Dict]:
    """Searches for packages"""
    return self.registry.search_packages(query)

def install(self, name: str, version: str = "latest") -> Tuple[bool, str]:
    """Installs a package"""
    package = self.registry.get_package(name, version)
    
    if not package:
        return False, f"Package {name}@{version} not found in registry"
    
    return self.installer.install_package(package, self.registry)

def uninstall(self, name: str, version: str = "latest") -> Tuple[bool, str]:
    """Uninstalls a package"""
    if version == "latest":
        # Find installed version
        installed = [k for k in self.installer.installed.keys() if k.startswith(f"{name}@")]
        if not installed:
            return False, f"Package {name} not installed"
        version = installed[0].split('@')[1]
    
    return self.installer.uninstall_package(name, version)

def publish(self, package: Package) -> bool:
    """Publishes a package to registry"""
    return self.registry.publish_package(package)

def list_available(self) -> List[Dict]:
    """Lists available packages"""
    return self.registry.list_all_packages()

def list_installed(self) -> List[Dict]:
    """Lists installed packages"""
    return self.installer.list_installed()

def get_package_path(self, name: str, version: str = "latest") -> Optional[str]:
    """Gets path to installed package"""
    return self.installer.get_package_path(name, version)
```

def main():
“”“Example usage”””
pm = JanusPackageManager()

```
print("=== Janus Package Manager ===\n")

# List available packages
print("Available packages:")
for pkg in pm.list_available():
    print(f"  - {pkg['name']}@{pkg['version']}: {pkg['description']}")

# Search for math packages
print("\nSearching for 'math':")
results = pm.search("math")
for result in results:
    print(f"  - {result['name']}@{result['version']}")

# Install a package
print("\nInstalling janus-math-utils...")
success, message = pm.install("janus-math-utils")
print(f"  {message}")

# List installed packages
print("\nInstalled packages:")
for pkg in pm.list_installed():
    print(f"  - {pkg['name']}@{pkg['version']} ({pkg['size_mb']:.2f} MB)")

# Create and publish a new package
print("\nCreating new package...")
new_package = pm.builder.create_package_from_code(
    name="my-custom-utils",
    version="1.0.0",
    description="My custom utility functions",
    author="Janus User",
    language="python",
    source_files={
        "utils.py": """
```

def greet(name):
return f”Hello, {name}!”

def add(a, b):
return a + b
“””
}
)

```
pm.publish(new_package)
print("  Published my-custom-utils@1.0.0")

# Install the new package
print("\nInstalling my-custom-utils...")
success, message = pm.install("my-custom-utils")
print(f"  {message}")

# Get package path
path = pm.get_package_path("my-custom-utils")
print(f"\nPackage installed at: {path}")
```

if **name** == ‘**main**’:
main()