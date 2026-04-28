"""
Command-line interface for the Janus Dependency Analyzer.

This module provides a comprehensive CLI for running full and incremental scans,
analyzing application capabilities, generating reports, and managing configuration.
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

from .scanners.system_scanner import SystemScannerImpl
from .analyzers.capability_analyzer import CapabilityAnalyzerImpl
from .config.configuration import ConfigurationManager
from .core.models import Configuration


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Rich console for pretty output
console = Console()

# Supported output formats
SUPPORTED_FORMATS = ['json', 'csv', 'html', 'table']


# ---------------------------------------------------------------------------
# CLI group
# ---------------------------------------------------------------------------

@click.group()
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging.')
@click.option(
    '--config', '-c',
    type=click.Path(exists=True),
    help='Path to a JSON or YAML configuration file.',
)
@click.pass_context
def cli(ctx: click.Context, verbose: bool, config: Optional[str]) -> None:
    """Janus Dependency Analyzer — discover and analyze application dependencies."""

    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Load configuration
    configuration = Configuration()
    if config:
        try:
            manager = ConfigurationManager()
            configuration = manager.parse_file(Path(config))
            console.print(f"[green]Configuration loaded from {config}[/green]")
        except Exception as exc:
            console.print(f"[red]Failed to load configuration from {config}: {exc}[/red]")
            sys.exit(1)

    ctx.ensure_object(dict)
    ctx.obj['config'] = configuration
    ctx.obj['verbose'] = verbose


# ---------------------------------------------------------------------------
# scan command
# ---------------------------------------------------------------------------

@cli.command()
@click.option(
    '--incremental', '-i',
    is_flag=True,
    default=False,
    help='Perform an incremental scan (only detect changes since last scan).',
)
@click.option(
    '--since',
    type=str,
    default=None,
    help=(
        'ISO-8601 datetime for incremental scan baseline '
        '(e.g. "2024-01-01T00:00:00"). Defaults to 24 hours ago.'
    ),
)
@click.option(
    '--output', '-o',
    type=click.Path(),
    default=None,
    help='Output file path for the scan results.',
)
@click.option(
    '--format', '-f',
    'output_format',
    type=click.Choice(['json', 'table'], case_sensitive=False),
    default='table',
    show_default=True,
    help='Output format for scan results.',
)
@click.pass_context
def scan(
    ctx: click.Context,
    incremental: bool,
    since: Optional[str],
    output: Optional[str],
    output_format: str,
) -> None:
    """Perform a full or incremental system scan for installed applications.

    By default a full scan is performed. Use --incremental to scan only for
    changes since a previous scan.

    Examples:

    \b
        janus-analyzer scan
        janus-analyzer scan --incremental
        janus-analyzer scan --incremental --since 2024-06-01T00:00:00
        janus-analyzer scan --format json --output results.json
    """
    scanner = SystemScannerImpl()

    if incremental:
        # Determine baseline time
        if since:
            try:
                last_scan_time = datetime.fromisoformat(since)
            except ValueError:
                console.print(
                    f"[red]Invalid --since value '{since}'. "
                    "Expected ISO-8601 format, e.g. '2024-01-01T00:00:00'.[/red]"
                )
                sys.exit(1)
        else:
            from datetime import timedelta
            last_scan_time = datetime.now() - timedelta(hours=24)

        console.print(
            f"[bold blue]Starting incremental scan "
            f"(since {last_scan_time.isoformat()})...[/bold blue]"
        )
        scan_label = "Scanning for changes..."
        scan_fn = lambda: scanner.scan_incremental(last_scan_time)  # noqa: E731
    else:
        console.print("[bold blue]Starting full system scan...[/bold blue]")
        scan_label = "Scanning system for applications..."
        scan_fn = scanner.scan_full  # noqa: E731

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task(scan_label, total=None)
            scan_result = scan_fn()
            progress.update(task, description="Scan completed!")

        # Display results
        if output_format == 'table':
            _display_scan_results_table(scan_result)
        elif output_format == 'json':
            _display_scan_results_json(scan_result, output)

        # Summary
        scan_type_label = "Incremental scan" if incremental else "Full scan"
        console.print(f"\n[green]{scan_type_label} completed successfully![/green]")
        console.print(
            f"Found {scan_result.total_applications} applications "
            f"({scan_result.accessible_applications} accessible)"
        )

        if scan_result.errors:
            console.print(f"[red]Encountered {len(scan_result.errors)} error(s)[/red]")
        if scan_result.warnings:
            console.print(f"[yellow]Encountered {len(scan_result.warnings)} warning(s)[/yellow]")

    except Exception as exc:
        console.print(f"[red]Scan failed: {exc}[/red]")
        logger.exception("Scan failed")
        sys.exit(1)


# ---------------------------------------------------------------------------
# analyze command
# ---------------------------------------------------------------------------

@cli.command()
@click.argument('app_name')
@click.option(
    '--output', '-o',
    type=click.Path(),
    default=None,
    help='Output file path for the analysis results.',
)
@click.option(
    '--format', '-f',
    'output_format',
    type=click.Choice(['json', 'table'], case_sensitive=False),
    default='table',
    show_default=True,
    help='Output format for analysis results.',
)
@click.pass_context
def analyze(
    ctx: click.Context,
    app_name: str,
    output: Optional[str],
    output_format: str,
) -> None:
    """Analyze capabilities of a specific application.

    APP_NAME is matched case-insensitively against discovered application names.

    Examples:

    \b
        janus-analyzer analyze git
        janus-analyzer analyze ffmpeg --format json --output ffmpeg_caps.json
    """
    console.print(f"[bold blue]Analyzing application: {app_name}[/bold blue]")

    try:
        scanner = SystemScannerImpl()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Scanning for application...", total=None)
            scan_result = scanner.scan_full()
            progress.update(task, description="Scan complete, searching for application...")

        # Find the application
        target_app = None
        for app in scan_result.applications:
            if app_name.lower() in app.name.lower():
                target_app = app
                break

        if not target_app:
            console.print(f"[red]Application '{app_name}' not found on this system.[/red]")
            sys.exit(1)

        console.print(
            f"Found: [green]{target_app.name}[/green] "
            f"v{target_app.version or 'unknown'}"
        )

        # Analyze capabilities
        analyzer = CapabilityAnalyzerImpl()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Analyzing capabilities...", total=None)
            capabilities = analyzer.analyze_application(target_app)
            progress.update(task, description="Analysis completed!")

        if output_format == 'table':
            _display_capabilities_table(capabilities)
        elif output_format == 'json':
            cap_data = [
                {
                    "id": cap.id,
                    "name": cap.name,
                    "category": cap.category.value,
                    "interface_type": cap.interface_type.value,
                    "description": cap.description,
                    "confidence_score": cap.confidence_score,
                    "detection_method": cap.detection_method,
                }
                for cap in capabilities
            ]
            json_output = json.dumps(
                {"application": target_app.name, "capabilities": cap_data},
                indent=2,
            )
            if output:
                Path(output).write_text(json_output, encoding='utf-8')
                console.print(f"[green]Results saved to {output}[/green]")
            else:
                console.print(json_output)

        console.print(f"\n[green]Analysis completed![/green]")
        console.print(f"Found {len(capabilities)} capability/capabilities")

    except SystemExit:
        raise
    except Exception as exc:
        console.print(f"[red]Analysis failed: {exc}[/red]")
        logger.exception("Analysis failed")
        sys.exit(1)


# ---------------------------------------------------------------------------
# report command
# ---------------------------------------------------------------------------

@cli.command()
@click.option(
    '--type', '-t',
    'report_type',
    type=click.Choice(
        ['summary', 'capabilities', 'dependencies', 'priority', 'full'],
        case_sensitive=False,
    ),
    default='summary',
    show_default=True,
    help='Type of report to generate.',
)
@click.option(
    '--output', '-o',
    type=click.Path(),
    default=None,
    help='Output file path (or directory when --format is used).',
)
@click.option(
    '--format', '-f',
    'output_format',
    type=click.Choice(['json', 'csv', 'html', 'table'], case_sensitive=False),
    default='table',
    show_default=True,
    help='Output format for the report.',
)
@click.option(
    '--all-formats',
    is_flag=True,
    default=False,
    help='Export the report in all supported formats (json, csv, html) to --output directory.',
)
@click.option(
    '--codebase', '-b',
    type=click.Path(exists=True),
    default='.',
    show_default=True,
    help='Path to the Janus codebase to analyze for dependencies.',
)
@click.pass_context
def report(
    ctx: click.Context,
    report_type: str,
    output: Optional[str],
    output_format: str,
    all_formats: bool,
    codebase: str,
) -> None:
    """Generate analysis reports.

    Runs a system scan and produces the requested report type.

    Examples:

    \b
        janus-analyzer report
        janus-analyzer report --type full --format html --output report.html
        janus-analyzer report --type dependencies --codebase /path/to/janus
        janus-analyzer report --type full --all-formats --output ./reports/
    """
    from .reports.generator import ReportGenerator
    from .dependency.mapper import DependencyMapper
    from .priority.engine import PriorityEngine, AnalysisContext

    config: Configuration = ctx.obj['config']

    console.print(
        f"[bold blue]Generating {report_type} report "
        f"(format: {output_format})...[/bold blue]"
    )

    try:
        # Step 1: Scan
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Scanning system...", total=None)
            scanner = SystemScannerImpl()
            scan_result = scanner.scan_full()
            progress.update(task, description="Scan complete.")

        # Step 2: Analyze capabilities (for capability/full reports)
        capabilities = []
        if report_type in ('capabilities', 'priority', 'full'):
            from concurrent.futures import ThreadPoolExecutor, as_completed
            import multiprocessing
            from janus_dependency_analyzer.filters.app_filter import ApplicationFilter, FilterConfig
            from janus_dependency_analyzer.filters.deduplicator import ApplicationDeduplicator
            
            analyzer = CapabilityAnalyzerImpl()
            accessible_apps = [a for a in scan_result.applications if a.is_accessible]
            
            # Apply deduplication to remove duplicate versions
            deduplicator = ApplicationDeduplicator()
            deduplicated_apps = deduplicator.deduplicate(accessible_apps)
            
            console.print(
                f"[dim]Deduplicated {len(accessible_apps)} apps → {len(deduplicated_apps)} "
                f"({len(accessible_apps) - len(deduplicated_apps)} duplicates removed)[/dim]"
            )
            
            # Apply smart filtering to focus on development tools
            app_filter = ApplicationFilter(FilterConfig(enabled=True))
            filtered_apps = app_filter.get_apps_to_analyze(deduplicated_apps)
            
            console.print(
                f"[dim]Filtered {len(deduplicated_apps)} apps → {len(filtered_apps)} "
                f"to analyze ({len(deduplicated_apps) - len(filtered_apps)} skipped)[/dim]"
            )
            
            # Use parallel processing with worker count based on CPU cores
            max_workers = min(multiprocessing.cpu_count() * 2, len(filtered_apps))
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task(
                    "Analyzing capabilities...", total=len(filtered_apps)
                )
                
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    # Submit all analysis tasks
                    future_to_app = {
                        executor.submit(analyzer.analyze_application, app): app
                        for app in filtered_apps
                    }
                    
                    # Collect results as they complete
                    for future in as_completed(future_to_app):
                        app = future_to_app[future]
                        try:
                            caps = future.result(timeout=60)
                            capabilities.extend(caps)
                        except Exception as exc:
                            logger.error(f"Failed to analyze {app.name}: {exc}")
                        finally:
                            progress.advance(task)

        # Step 3: Map dependencies (for dependency/full reports)
        dependency_mappings = []
        if report_type in ('dependencies', 'full'):
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("Mapping dependencies...", total=None)
                mapper = DependencyMapper(Path(codebase))
                dep_map = mapper.scan_codebase()
                dependency_mappings = list(dep_map.values())
                progress.update(task, description="Dependency mapping complete.")

        # Step 4: Rank capabilities (for priority/full reports)
        ranked_capabilities = []
        if report_type in ('priority', 'full') and capabilities:
            engine = PriorityEngine()
            contexts = {
                cap.id: AnalysisContext(usage_frequency=0, max_frequency=1)
                for cap in capabilities
            }
            ranked_capabilities = engine.rank_capabilities(capabilities, contexts)

        # Step 5: Build report data
        generator = ReportGenerator()

        if report_type == 'summary':
            report_data = generator.generate_summary_report([scan_result])
        elif report_type == 'capabilities':
            report_data = generator.generate_capability_inventory(capabilities)
        elif report_type == 'dependencies':
            # Convert DependencyMapping from mapper to core model format
            from .core.models import DependencyMapping as CoreDependencyMapping, UsagePattern
            core_deps = []
            for dm in dependency_mappings:
                core_deps.append(CoreDependencyMapping(
                    janus_component=dm.first_seen,
                    external_application=dm.application_name,
                    usage_pattern=UsagePattern(
                        invocation_method=dm.invocations[0].invocation_method
                        if dm.invocations else "unknown",
                    ),
                    frequency=dm.invocation_count,
                    last_used=datetime.now(),
                    context=dm.first_seen,
                ))
            report_data = generator.generate_dependency_report(core_deps)
        elif report_type == 'priority':
            report_data = generator.generate_priority_report(ranked_capabilities)
        elif report_type == 'full':
            from .core.models import DependencyMapping as CoreDependencyMapping, UsagePattern
            core_deps = []
            for dm in dependency_mappings:
                core_deps.append(CoreDependencyMapping(
                    janus_component=dm.first_seen,
                    external_application=dm.application_name,
                    usage_pattern=UsagePattern(
                        invocation_method=dm.invocations[0].invocation_method
                        if dm.invocations else "unknown",
                    ),
                    frequency=dm.invocation_count,
                    last_used=datetime.now(),
                    context=dm.first_seen,
                ))
            report_data = generator.generate_full_report(
                scan_results=[scan_result],
                capabilities=capabilities,
                dependencies=core_deps,
                ranked_capabilities=ranked_capabilities,
            )
        else:
            console.print(f"[red]Unknown report type: {report_type}[/red]")
            sys.exit(1)

        # Sanitize paths if configured
        if config.sanitize_paths:
            report_data = generator.sanitize_report_paths(report_data)

        # Step 6: Output
        if all_formats:
            out_dir = Path(output) if output else Path('.')
            paths = generator.export_all_formats(
                report_data, out_dir, base_filename=f"{report_type}_report"
            )
            console.print(f"[green]Reports exported:[/green]")
            for fmt, path in paths.items():
                console.print(f"  {fmt}: {path}")
        elif output_format == 'table':
            _display_report_table(report_data, report_type)
        else:
            out_path = Path(output) if output else None
            if out_path:
                generator.export_report(report_data, output_format, out_path)
                console.print(f"[green]Report saved to {out_path}[/green]")
            else:
                # Print to stdout
                if output_format == 'json':
                    console.print(json.dumps(report_data, indent=2, default=str))
                else:
                    # For csv/html without output path, save to a temp file and show path
                    import tempfile
                    suffix = f".{output_format}"
                    with tempfile.NamedTemporaryFile(
                        mode='w', suffix=suffix, delete=False, encoding='utf-8'
                    ) as tmp:
                        tmp_path = Path(tmp.name)
                    generator.export_report(report_data, output_format, tmp_path)
                    console.print(
                        f"[yellow]Report written to temporary file: {tmp_path}[/yellow]"
                    )

        console.print(f"\n[green]{report_type.capitalize()} report generated successfully![/green]")
        
        # Save cache if analyzer was used
        if report_type in ('capabilities', 'priority', 'full') and 'analyzer' in locals():
            analyzer.save_cache()
            cache_stats = analyzer.get_cache_stats()
            if cache_stats.get('total_entries', 0) > 0:
                console.print(
                    f"[dim]Cache: {cache_stats['total_entries']} entries, "
                    f"{cache_stats['cache_size_mb']}MB[/dim]"
                )

    except SystemExit:
        raise
    except Exception as exc:
        console.print(f"[red]Report generation failed: {exc}[/red]")
        logger.exception("Report generation failed")
        sys.exit(1)


# ---------------------------------------------------------------------------
# configure command
# ---------------------------------------------------------------------------

@cli.command()
@click.option(
    '--show',
    is_flag=True,
    default=False,
    help='Display the current (or default) configuration.',
)
@click.option(
    '--generate',
    type=click.Path(),
    default=None,
    help='Generate a default configuration file at the given path.',
)
@click.option(
    '--format', '-f',
    'output_format',
    type=click.Choice(['json', 'yaml'], case_sensitive=False),
    default='json',
    show_default=True,
    help='Format for the generated configuration file.',
)
@click.option(
    '--validate',
    type=click.Path(exists=True),
    default=None,
    help='Validate an existing configuration file.',
)
@click.pass_context
def configure(
    ctx: click.Context,
    show: bool,
    generate: Optional[str],
    output_format: str,
    validate: Optional[str],
) -> None:
    """Manage analyzer configuration.

    Use --show to display the active configuration, --generate to create a
    default config file, or --validate to check an existing config file.

    Examples:

    \b
        janus-analyzer configure --show
        janus-analyzer configure --generate config.json
        janus-analyzer configure --generate config.yaml --format yaml
        janus-analyzer configure --validate my_config.json
    """
    manager = ConfigurationManager()
    config: Configuration = ctx.obj['config']

    if validate:
        try:
            loaded = manager.parse_file(Path(validate))
            result = manager.validate(loaded)
            if result.is_valid:
                console.print(f"[green]Configuration file '{validate}' is valid.[/green]")
            else:
                console.print(f"[red]Configuration file '{validate}' is invalid:[/red]")
                for err in result.errors:
                    console.print(f"  [red]• {err}[/red]")
            if result.warnings:
                for warn in result.warnings:
                    console.print(f"  [yellow]⚠ {warn}[/yellow]")
        except Exception as exc:
            console.print(f"[red]Failed to validate '{validate}': {exc}[/red]")
            sys.exit(1)
        return

    if generate:
        try:
            default_config = Configuration()
            formatted = manager.format(default_config, format=output_format)
            out_path = Path(generate)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(formatted, encoding='utf-8')
            console.print(
                f"[green]Default configuration written to '{generate}' "
                f"({output_format} format).[/green]"
            )
        except Exception as exc:
            console.print(f"[red]Failed to generate configuration: {exc}[/red]")
            sys.exit(1)
        return

    if show or not (validate or generate):
        # Display current configuration
        _display_configuration(config)


# ---------------------------------------------------------------------------
# info command
# ---------------------------------------------------------------------------

@cli.command()
@click.pass_context
def info(ctx: click.Context) -> None:
    """Display system information and active configuration."""
    config: Configuration = ctx.obj['config']

    console.print("[bold blue]Janus Dependency Analyzer — System Information[/bold blue]\n")

    scanner = SystemScannerImpl()
    platform = scanner.detect_platform()

    info_table = Table(title="System Information")
    info_table.add_column("Property", style="cyan")
    info_table.add_column("Value", style="green")

    info_table.add_row("Platform", platform.value.capitalize())
    info_table.add_row("Scanner", scanner.get_platform_scanner().__class__.__name__)

    import sys as _sys
    info_table.add_row("Python Version", _sys.version.split()[0])

    console.print(info_table)
    _display_configuration(config)


# ---------------------------------------------------------------------------
# Private display helpers
# ---------------------------------------------------------------------------

def _display_configuration(config: Configuration) -> None:
    """Render a Configuration object as a Rich table."""
    config_table = Table(title="Active Configuration")
    config_table.add_column("Setting", style="cyan")
    config_table.add_column("Value", style="green")

    config_table.add_row("Scan Timeout", f"{config.scan_timeout_seconds}s")
    config_table.add_row("Max Applications", str(config.max_applications_per_scan))
    config_table.add_row("Min Confidence", str(config.min_confidence_threshold))
    config_table.add_row("Analysis Timeout", f"{config.analysis_timeout_seconds}s")
    config_table.add_row("Respect Access Controls", str(config.respect_access_controls))
    config_table.add_row("Encrypt Stored Data", str(config.encrypt_stored_data))
    config_table.add_row("Audit Logging", str(config.audit_logging_enabled))
    config_table.add_row("Report Formats", ", ".join(config.report_formats))
    config_table.add_row("Include Charts", str(config.include_charts))
    config_table.add_row("Sanitize Paths", str(config.sanitize_paths))
    config_table.add_row(
        "Priority Weights",
        (
            f"usage={config.priority_weights.usage}, "
            f"complexity={config.priority_weights.complexity}, "
            f"security={config.priority_weights.security}, "
            f"performance={config.priority_weights.performance}"
        ),
    )
    if config.scan_exclusion_patterns:
        config_table.add_row(
            "Exclusion Patterns", ", ".join(config.scan_exclusion_patterns)
        )

    console.print(config_table)


def _display_scan_results_table(scan_result) -> None:
    """Display scan results in a Rich table (first 20 apps)."""
    table = Table(
        title=f"Discovered Applications ({scan_result.platform.value.capitalize()})"
    )
    table.add_column("Name", style="cyan")
    table.add_column("Version", style="green")
    table.add_column("Path", style="blue")
    table.add_column("Accessible", style="yellow")

    for app in scan_result.applications[:20]:
        accessible = "✓" if app.is_accessible else "✗"
        table.add_row(
            app.name,
            app.version or "Unknown",
            str(app.executable_path),
            accessible,
        )

    console.print(table)

    if len(scan_result.applications) > 20:
        console.print(
            f"[dim]... and {len(scan_result.applications) - 20} more applications[/dim]"
        )


def _display_scan_results_json(scan_result, output_path: Optional[str]) -> None:
    """Display or save scan results as JSON."""
    data = {
        "scan_info": {
            "platform": scan_result.platform.value,
            "scan_type": scan_result.scan_type,
            "start_time": scan_result.scan_start_time.isoformat(),
            "end_time": (
                scan_result.scan_end_time.isoformat()
                if scan_result.scan_end_time
                else None
            ),
            "total_applications": scan_result.total_applications,
            "accessible_applications": scan_result.accessible_applications,
        },
        "applications": [
            {
                "id": app.id,
                "name": app.name,
                "version": app.version,
                "executable_path": str(app.executable_path),
                "installation_path": str(app.installation_path),
                "is_accessible": app.is_accessible,
                "access_error": app.access_error,
                "discovered_at": app.discovered_at.isoformat(),
            }
            for app in scan_result.applications
        ],
        "errors": scan_result.errors,
        "warnings": scan_result.warnings,
    }

    json_output = json.dumps(data, indent=2)

    if output_path:
        Path(output_path).write_text(json_output, encoding='utf-8')
        console.print(f"[green]Results saved to {output_path}[/green]")
    else:
        console.print(json_output)


def _display_capabilities_table(capabilities) -> None:
    """Display capabilities in a Rich table."""
    if not capabilities:
        console.print("[yellow]No capabilities found.[/yellow]")
        return

    table = Table(title="Application Capabilities")
    table.add_column("Name", style="cyan")
    table.add_column("Category", style="green")
    table.add_column("Interface", style="blue")
    table.add_column("Confidence", style="yellow")
    table.add_column("Description", style="white")

    for cap in capabilities:
        confidence_str = f"{cap.confidence_score:.2f}"
        description = (
            cap.description[:50] + "..."
            if len(cap.description) > 50
            else cap.description
        )
        table.add_row(
            cap.name,
            cap.category.value.replace('_', ' ').title(),
            cap.interface_type.value.replace('_', ' ').title(),
            confidence_str,
            description,
        )

    console.print(table)


def _display_report_table(report_data: dict, report_type: str) -> None:
    """Render report data as a Rich table (best-effort for table format)."""
    console.print(f"\n[bold]Report: {report_type.capitalize()}[/bold]\n")

    for key, value in report_data.items():
        if key == 'generated_at':
            continue
        if isinstance(value, (str, int, float, bool)):
            console.print(f"  [cyan]{key}:[/cyan] {value}")
        elif isinstance(value, list) and value:
            if isinstance(value[0], dict):
                # Render as a table
                tbl = Table(title=key.replace('_', ' ').title())
                headers = list(value[0].keys())
                for h in headers:
                    tbl.add_column(h.replace('_', ' ').title(), style="cyan")
                for row in value[:20]:
                    tbl.add_row(*[str(row.get(h, '')) for h in headers])
                console.print(tbl)
                if len(value) > 20:
                    console.print(f"[dim]... and {len(value) - 20} more rows[/dim]")
            else:
                console.print(f"  [cyan]{key}:[/cyan] {', '.join(str(v) for v in value)}")
        elif isinstance(value, dict):
            tbl = Table(title=key.replace('_', ' ').title())
            tbl.add_column("Key", style="cyan")
            tbl.add_column("Value", style="green")
            for k, v in value.items():
                tbl.add_row(str(k), str(v))
            console.print(tbl)

    if 'generated_at' in report_data:
        console.print(f"\n[dim]Generated at: {report_data['generated_at']}[/dim]")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Main entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()
