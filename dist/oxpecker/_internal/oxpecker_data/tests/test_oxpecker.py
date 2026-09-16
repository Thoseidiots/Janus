"""
Comprehensive test suite for Universal Oxpecker.
Run with:  cd /home/user/universal_oxpecker && python -m pytest tests/ -v
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
from core.engine import DebugEngine
from core.events import Issue
from adapters import ALL_ADAPTERS
from adapters.python_adapter import PythonAdapter
from adapters.javascript_typescript_adapter import JavaScriptTypeScriptAdapter
from adapters.java_kotlin_adapter import JavaKotlinAdapter
from adapters.c_cpp_adapter import CCppAdapter
from adapters.rust_adapter import RustAdapter
from adapters.go_adapter import GoAdapter
from adapters.csharp_adapter import CSharpAdapter
from adapters.ruby_adapter import RubyAdapter
from adapters.php_adapter import PhpAdapter
from adapters.lua_adapter import LuaAdapter
from adapters.swift_adapter import SwiftAdapter
from adapters.zig_adapter import ZigAdapter
from adapters.functional_dsl_adapter import FunctionalDslAdapter
from analysis.fix_suggester import FixSuggester
from analysis.repair_workflow import AutomatedProgramRepair
from core.scanner import ProjectScanner


# ── Helper ────────────────────────────────────────────────────────────────────
def _engine() -> DebugEngine:
    e = DebugEngine()
    for a in ALL_ADAPTERS:
        e.register_adapter(a)
    return e


# ── Engine tests ──────────────────────────────────────────────────────────────
class TestEngine:
    def test_all_adapters_registered(self):
        e = _engine()
        langs = e.registered_languages()
        assert "python"             in langs
        assert "javascript/typescript" in langs
        assert "java/kotlin"        in langs
        assert "c/c++"              in langs
        assert "rust"               in langs
        assert "go"                 in langs
        assert "csharp"             in langs
        assert "ruby"               in langs
        assert "php"                in langs
        assert "lua"                in langs
        assert "swift"              in langs
        assert "zig"                in langs
        assert "functional/DSL"     in langs

    def test_detect_python(self):
        e = _engine()
        adapter = e.detect_adapter("script.py")
        assert adapter.language == "python"

    def test_detect_typescript(self):
        e = _engine()
        adapter = e.detect_adapter("app.ts")
        assert adapter.language == "javascript/typescript"

    def test_detect_rust(self):
        e = _engine()
        adapter = e.detect_adapter("main.rs")
        assert adapter.language == "rust"

    def test_detect_go(self):
        e = _engine()
        adapter = e.detect_adapter("server.go")
        assert adapter.language == "go"

    def test_detect_unknown_raises(self):
        e = _engine()
        with pytest.raises(ValueError):
            e.detect_adapter("unknownfile.xyz123")

    def test_start_session(self):
        e = _engine()
        session = e.start("hello.py")
        assert session["language"] == "python"
        assert "adapter" in session


# ── Python adapter ────────────────────────────────────────────────────────────
class TestPythonAdapter:
    adapter = PythonAdapter()

    def test_detect_py_extension(self):
        assert self.adapter.detect("foo.py")

    def test_detect_python_tag(self):
        assert self.adapter.detect("python")

    def test_no_detect_js(self):
        assert not self.adapter.detect("foo.js")

    def test_syntax_error(self):
        code = "def foo(:\n    pass\n"
        issues = self.adapter.analyze_code(code, "test.py")
        assert any(i.severity == "error" for i in issues)

    def test_mutable_default_arg(self):
        code = "def f(x=[]):\n    return x\n"
        issues = self.adapter.analyze_code(code, "test.py")
        assert any("mutable" in (i.message or "").lower() for i in issues)

    def test_bare_except(self):
        code = "try:\n    pass\nexcept:\n    pass\n"
        issues = self.adapter.analyze_code(code, "test.py")
        assert any("bare" in (i.message or "").lower() or "except" in (i.message or "").lower()
                   for i in issues)

    def test_clean_code_no_issues(self):
        code = "def greet(name: str) -> str:\n    return f'Hello {name}'\n"
        issues = self.adapter.analyze_code(code)
        # no errors
        assert not any(i.severity == "error" for i in issues)

    def test_normalize_error(self):
        try:
            raise ValueError("test error")
        except ValueError as e:
            issue = self.adapter.normalize_error(e)
        assert issue.severity == "error"
        assert "test error" in issue.message
        assert issue.language == "python"

    def test_analyze_fixture_file(self):
        fixture = os.path.join(os.path.dirname(__file__), "fixtures", "bad.py")
        issues = self.adapter.analyze_file(fixture)
        assert len(issues) > 0


# ── JavaScript / TypeScript adapter ──────────────────────────────────────────
class TestJSAdapter:
    adapter = JavaScriptTypeScriptAdapter()

    def test_detect_js(self):
        assert self.adapter.detect("app.js")

    def test_detect_ts(self):
        assert self.adapter.detect("main.ts")

    def test_detect_tsx(self):
        assert self.adapter.detect("Component.tsx")

    def test_no_detect_py(self):
        assert not self.adapter.detect("script.py")

    def test_var_warning(self):
        code = "var x = 1;\n"
        issues = self.adapter.analyze_code(code, "test.js")
        assert any("var" in (i.message or "").lower() for i in issues)

    def test_loose_equality_warning(self):
        code = 'if (x == "hello") {}\n'
        issues = self.adapter.analyze_code(code, "test.js")
        assert any("===" in (i.message or "") or "strict" in (i.message or "").lower()
                   for i in issues)

    def test_strict_equality_has_no_false_positive(self):
        code = 'if (x === "hello") {}\n'
        issues = self.adapter.analyze_code(code, "test.js")
        assert not any("strict equality" in (i.message or "").lower() for i in issues)

    def test_eval_warning(self):
        code = "eval('console.log(1)');\n"
        issues = self.adapter.analyze_code(code, "test.js")
        assert any("eval" in (i.message or "").lower() for i in issues)

    def test_normalize_error(self):
        err = Exception("TypeError: Cannot read properties of undefined (reading 'foo')\n    at Object.<anonymous> (app.js:10:5)")
        issue = self.adapter.normalize_error(err)
        assert issue.severity == "error"


# ── Java/Kotlin adapter ───────────────────────────────────────────────────────
class TestJavaAdapter:
    adapter = JavaKotlinAdapter()

    def test_detect_java(self):
        assert self.adapter.detect("Main.java")

    def test_detect_kotlin(self):
        assert self.adapter.detect("App.kt")

    def test_string_equals_null_warning(self):
        code = 'if (obj.equals(null)) {}\n'
        issues = self.adapter.analyze_code(code, "Main.java")
        assert any(i.severity in ("warning", "error") for i in issues)

    def test_string_ref_equality(self):
        code = 'if (str == "hello") {}\n'
        issues = self.adapter.analyze_code(code, "Main.java")
        assert any(i.severity in ("warning", "error") for i in issues)


# ── C/C++ adapter ─────────────────────────────────────────────────────────────
class TestCCppAdapter:
    adapter = CCppAdapter()

    def test_detect_c(self):
        assert self.adapter.detect("main.c")

    def test_detect_cpp(self):
        assert self.adapter.detect("engine.cpp")

    def test_gets_error(self):
        code = "gets(buf);\n"
        issues = self.adapter.analyze_code(code, "test.c")
        assert any("gets" in (i.message or "").lower() for i in issues)
        assert any(i.severity == "error" for i in issues)

    def test_sprintf_warning(self):
        code = "sprintf(buf, fmt, val);\n"
        issues = self.adapter.analyze_code(code, "test.c")
        assert any("sprintf" in (i.message or "").lower() for i in issues)


# ── Rust adapter ──────────────────────────────────────────────────────────────
class TestRustAdapter:
    adapter = RustAdapter()

    def test_detect_rs(self):
        assert self.adapter.detect("main.rs")

    def test_unwrap_warning(self):
        code = "let x = some_func().unwrap();\n"
        issues = self.adapter.analyze_code(code, "main.rs")
        assert any("unwrap" in (i.message or "").lower() for i in issues)

    def test_unsafe_warning(self):
        code = "unsafe {\n    *ptr = 42;\n}\n"
        issues = self.adapter.analyze_code(code, "main.rs")
        assert any("unsafe" in (i.message or "").lower() for i in issues)


# ── Go adapter ────────────────────────────────────────────────────────────────
class TestGoAdapter:
    adapter = GoAdapter()

    def test_detect_go(self):
        assert self.adapter.detect("server.go")

    def test_panic_warning(self):
        code = 'panic("something broke")\n'
        issues = self.adapter.analyze_code(code, "main.go")
        assert any("panic" in (i.message or "").lower() for i in issues)

    def test_println_info(self):
        code = 'fmt.Println("debug")\n'
        issues = self.adapter.analyze_code(code, "main.go")
        assert any(i.severity == "info" for i in issues)


# ── C# adapter ────────────────────────────────────────────────────────────────
class TestCSharpAdapter:
    adapter = CSharpAdapter()

    def test_detect_cs(self):
        assert self.adapter.detect("Program.cs")

    def test_task_result_warning(self):
        code = "var result = task.Result;\n"
        issues = self.adapter.analyze_code(code, "Program.cs")
        assert any(".result" in (i.message or "").lower() or "deadlock" in (i.message or "").lower()
                   for i in issues)

    def test_thread_sleep_warning(self):
        code = "Thread.Sleep(1000);\n"
        issues = self.adapter.analyze_code(code, "Program.cs")
        assert any("thread" in (i.message or "").lower() for i in issues)


# ── Ruby adapter ──────────────────────────────────────────────────────────────
class TestRubyAdapter:
    adapter = RubyAdapter()

    def test_detect_rb(self):
        assert self.adapter.detect("app.rb")

    def test_detect_gemfile(self):
        assert self.adapter.detect("Gemfile")

    def test_eval_warning(self):
        code = "eval(user_input)\n"
        issues = self.adapter.analyze_code(code, "app.rb")
        assert any("eval" in (i.message or "").lower() for i in issues)

    def test_bare_rescue(self):
        code = "begin\n  foo\nrescue\n  nil\nend\n"
        issues = self.adapter.analyze_code(code, "app.rb")
        assert any("rescue" in (i.message or "").lower() for i in issues)


# ── PHP adapter ───────────────────────────────────────────────────────────────
class TestPhpAdapter:
    adapter = PhpAdapter()

    def test_detect_php(self):
        assert self.adapter.detect("index.php")

    def test_eval_error(self):
        code = "eval($user_input);\n"
        issues = self.adapter.analyze_code(code, "index.php")
        assert any(i.severity == "error" for i in issues)

    def test_mysql_deprecated(self):
        code = "mysql_query($sql);\n"
        issues = self.adapter.analyze_code(code, "index.php")
        assert any("mysql" in (i.message or "").lower() for i in issues)


# ── Lua adapter ───────────────────────────────────────────────────────────────
class TestLuaAdapter:
    adapter = LuaAdapter()

    def test_detect_lua(self):
        assert self.adapter.detect("script.lua")

    def test_loadstring_warning(self):
        code = "loadstring('print(1)')()\n"
        issues = self.adapter.analyze_code(code, "script.lua")
        assert any("load" in (i.message or "").lower() for i in issues)


# ── Swift adapter ─────────────────────────────────────────────────────────────
class TestSwiftAdapter:
    adapter = SwiftAdapter()

    def test_detect_swift(self):
        assert self.adapter.detect("ViewController.swift")

    def test_force_try_warning(self):
        code = "let data = try! Data(contentsOf: url)\n"
        issues = self.adapter.analyze_code(code, "ViewController.swift")
        assert any("try!" in (i.message or "") or "crash" in (i.message or "").lower()
                   for i in issues)


# ── Zig adapter ───────────────────────────────────────────────────────────────
class TestZigAdapter:
    adapter = ZigAdapter()

    def test_detect_zig(self):
        assert self.adapter.detect("main.zig")

    def test_unreachable_warning(self):
        code = "unreachable;\n"
        issues = self.adapter.analyze_code(code, "main.zig")
        assert any("unreachable" in (i.message or "").lower() for i in issues)


# ── Functional/DSL adapter ────────────────────────────────────────────────────
class TestFunctionalDSLAdapter:
    adapter = FunctionalDslAdapter()

    def test_detect_haskell(self):
        assert self.adapter.detect("Main.hs")

    def test_detect_elixir(self):
        assert self.adapter.detect("app.ex")

    def test_detect_sql(self):
        assert self.adapter.detect("query.sql")

    def test_detect_yaml(self):
        assert self.adapter.detect("config.yml")

    def test_sql_select_star(self):
        code = "SELECT * FROM users;\n"
        issues = self.adapter.analyze_code(code, "query.sql")
        assert any("select *" in (i.message or "").lower() for i in issues)

    def test_yaml_tab_error(self):
        code = "key:\n\tvalue\n"
        issues = self.adapter.analyze_code(code, "config.yaml")
        assert any("tab" in (i.message or "").lower() for i in issues)


# ── FixSuggester ──────────────────────────────────────────────────────────────
class TestFixSuggester:
    suggester = FixSuggester()

    def _issue(self, msg, lang="python"):
        return Issue(severity="error", language=lang, message=msg)

    def test_name_error_suggestion(self):
        issue = self._issue("NameError: name 'foo' is not defined")
        suggestion = self.suggester.suggest(issue)
        assert suggestion is not None
        assert "foo" in suggestion

    def test_null_pointer_js(self):
        issue = self._issue("TypeError: Cannot read properties of null (reading 'bar')",
                            lang="javascript/typescript")
        suggestion = self.suggester.suggest(issue)
        assert suggestion is not None
        assert "null" in suggestion.lower() or "optional" in suggestion.lower()

    def test_zero_division_python(self):
        issue = self._issue("ZeroDivisionError: division by zero")
        suggestion = self.suggester.suggest(issue)
        assert suggestion is not None
        assert "0" in suggestion or "zero" in suggestion.lower()

    def test_unknown_error_returns_none(self):
        issue = self._issue("Some totally unknown error nobody has ever seen")
        suggestion = self.suggester.suggest(issue)
        # May be None OR a generic suggestion – just check it doesn't crash
        assert suggestion is None or isinstance(suggestion, str)

    def test_enrich_list(self):
        issues = [
            self._issue("NameError: name 'x' is not defined"),
            self._issue("ImportError: No module named 'numpy'"),
        ]
        enriched = self.suggester.enrich(issues)
        assert all(i.fix_suggestion is not None for i in enriched)
        assert all(i.complexity_tier is not None for i in enriched)

    def test_classify_complexity_tier_1(self):
        issue = Issue(
            severity="warning",
            language="javascript/typescript",
            message="Use strict equality '===' instead of '=='.",
        )
        assert self.suggester.classify_complexity(issue) == "Tier 1"

    def test_classify_complexity_tier_3(self):
        issue = Issue(
            severity="error",
            language="c/c++",
            message="segmentation fault",
        )
        assert self.suggester.classify_complexity(issue) == "Tier 3"


class TestProjectScanner:
    def test_scan_directory_finds_supported_files(self, tmp_path):
        project = tmp_path / "project"
        project.mkdir()
        (project / "a.py").write_text("try:\n    pass\nexcept:\n    pass\n", encoding="utf-8")
        (project / "b.js").write_text("var x = 1;\n", encoding="utf-8")
        (project / "notes.txt").write_text("ignore me", encoding="utf-8")

        scanner = ProjectScanner(_engine())
        results = scanner.scan(str(project), workers=2)

        assert len(results) == 2
        assert any(result.source_path.endswith("a.py") for result in results)
        assert any(result.source_path.endswith("b.js") for result in results)
        assert all(result.issues for result in results)


class TestAutomatedProgramRepair:
    def test_repair_file_works_on_copy_and_keeps_original(self, tmp_path):
        source = tmp_path / "bad.py"
        source.write_text("try:\n    pass\nexcept:\n    pass\n", encoding="utf-8")

        repairer = AutomatedProgramRepair(_engine())
        result = repairer.repair_file(str(source))

        assert result.original_path == str(source)
        assert result.working_path != str(source)
        assert source.read_text(encoding="utf-8") == "try:\n    pass\nexcept:\n    pass\n"
        assert "except Exception:" in open(result.working_path, encoding="utf-8").read()
        assert result.fixed_issue_count >= 1
        assert result.applied_patches
        assert all(patch.approval == "approved" for patch in result.applied_patches)

    def test_repair_file_javascript_strict_equality(self, tmp_path):
        source = tmp_path / "bad.js"
        source.write_text('if (x == "hello") {\n  var y = 1;\n}\n', encoding="utf-8")

        repairer = AutomatedProgramRepair(_engine())
        result = repairer.repair_file(str(source))
        repaired = open(result.working_path, encoding="utf-8").read()

        assert "===" in repaired or "let y" in repaired or "const y" in repaired
        assert result.fixed_issue_count >= 1

    def test_repair_file_records_ast_candidates_and_history(self, tmp_path):
        source = tmp_path / "bad.py"
        source.write_text("try:\n    pass\nexcept:\n    pass\n", encoding="utf-8")

        repairer = AutomatedProgramRepair(_engine())
        result = repairer.repair_file(str(source))

        assert any(candidate.patch_kind == "ast" for candidate in result.candidate_history)
        assert result.rollback_history
        assert result.applied_patches[0].plugin_name in {"python-ast", "python-text"}

    def test_repair_file_runs_pytest_validation(self, tmp_path):
        project = tmp_path / "project"
        project.mkdir()
        source = project / "bad_module.py"
        source.write_text(
            "def wrap(value):\n"
            "    try:\n"
            "        return value\n"
            "    except:\n"
            "        return None\n",
            encoding="utf-8",
        )
        tests_dir = project / "tests"
        tests_dir.mkdir()
        (tests_dir / "test_bad_module.py").write_text(
            "from bad_module import wrap\n\n"
            "def test_wrap():\n"
            "    assert wrap(3) == 3\n",
            encoding="utf-8",
        )

        repairer = AutomatedProgramRepair(_engine())
        result = repairer.repair_file(str(source))

        assert any(run.kind == "baseline" and run.success for run in result.test_runs)
        assert any(run.kind == "candidate" and run.success for run in result.test_runs)
        assert any(patch.test_status == "passed" for patch in result.applied_patches)

    def test_rollback_restores_previous_snapshot(self, tmp_path):
        source = tmp_path / "bad.py"
        original = "try:\n    pass\nexcept:\n    pass\n"
        source.write_text(original, encoding="utf-8")

        repairer = AutomatedProgramRepair(_engine())
        result = repairer.repair_file(str(source))
        assert open(result.working_path, encoding="utf-8").read() != original

        repairer.rollback_last(result.working_path)
        restored = open(result.working_path, encoding="utf-8").read()
        assert restored == original
