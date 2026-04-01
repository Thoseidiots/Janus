from .python_adapter import PythonAdapter
from .javascript_typescript_adapter import JavaScriptTypeScriptAdapter
from .java_kotlin_adapter import JavaKotlinAdapter
from .c_cpp_adapter import CCppAdapter
from .rust_adapter import RustAdapter
from .go_adapter import GoAdapter
from .csharp_adapter import CSharpAdapter
from .ruby_adapter import RubyAdapter
from .php_adapter import PhpAdapter
from .lua_adapter import LuaAdapter
from .swift_adapter import SwiftAdapter
from .zig_adapter import ZigAdapter
from .functional_dsl_adapter import FunctionalDslAdapter

ALL_ADAPTERS = [
    PythonAdapter(),
    JavaScriptTypeScriptAdapter(),
    JavaKotlinAdapter(),
    CCppAdapter(),
    RustAdapter(),
    GoAdapter(),
    CSharpAdapter(),
    RubyAdapter(),
    PhpAdapter(),
    LuaAdapter(),
    SwiftAdapter(),
    ZigAdapter(),
    FunctionalDslAdapter(),
]

__all__ = [
    "PythonAdapter",
    "JavaScriptTypeScriptAdapter",
    "JavaKotlinAdapter",
    "CCppAdapter",
    "RustAdapter",
    "GoAdapter",
    "CSharpAdapter",
    "RubyAdapter",
    "PhpAdapter",
    "LuaAdapter",
    "SwiftAdapter",
    "ZigAdapter",
    "FunctionalDslAdapter",
    "ALL_ADAPTERS",
]
