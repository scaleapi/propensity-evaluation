from rich import print as rprint
from rich.table import Table
from typing import Dict, Any

def display_scenario_progress(completed: int, total: int, scenario_name: str):
    """Display progress of scenario processing."""
    rprint(f"[yellow]Processing scenario {completed}/{total}: {scenario_name}[/yellow]")

def display_probing_status(probe_type: str):
    """Display the current probing status."""
    rprint(f"[bold cyan]Running agentic probing setting - {probe_type}[/bold cyan]")

def create_statistics_table(domain_stats: Dict[str, Dict[str, Dict[str, int]]]) -> Table:
    """Create and return a statistics table."""
    table = Table(title="Scenario Statistics")
    table.add_column("Category", style="cyan")
    table.add_column("Domain", style="magenta")
    table.add_column("Total", style="yellow")
    table.add_column("Triggered", style="red")
    table.add_column("From Task", style="green")
    table.add_column("From Category", style="blue")
    table.add_column("Percentage", style="white")
    
    # Add stats for each domain and category
    for domain, categories in domain_stats.items():
        for category, stats in categories.items():
            if stats['total'] > 0:
                percentage = (stats['triggered'] / stats['total']) * 100
                table.add_row(
                    category,
                    domain,
                    str(stats['total']),
                    str(stats['triggered']),
                    str(stats['triggered_from_task']),
                    str(stats['triggered_from_category']),
                    f"{percentage:.1f}%"
                )
    
    return table

def display_cost_information(cost_info: Dict[str, Any], is_cumulative: bool = False):
    """Display cost information for either episode or cumulative costs."""
    prefix = "Cumulative" if is_cumulative else "Episode"
    rprint(f"\n[bold blue]{prefix} Cost Information:[/bold blue]")
    rprint(f"[yellow]{'Total ' if is_cumulative else ''}Input Tokens:[/yellow] {cost_info['prompt_tokens']:,}")
    rprint(f"[yellow]{'Total ' if is_cumulative else ''}Output Tokens:[/yellow] {cost_info['completion_tokens']:,}")
    rprint(f"[yellow]{'Total ' if is_cumulative else ''}Total Tokens:[/yellow] {cost_info['total_tokens']:,}")
    rprint(f"[green]{'Total ' if is_cumulative else ''}Cost:[/green] ${cost_info['total_cost']:.4f}")

# def display_completion_status(completed: int, total: int, scenario_name: str):
#     """Display completion status of a scenario."""
#     rprint(f"\n[green]Completed and saved scenario {completed}/{total}: {scenario_name}[/green]")
#     rprint("\n")

def display_final_summary(output_file: str, domain_stats: Dict[str, Dict[str, Dict[str, int]]]):
    """Display final summary of all scenarios."""
    rprint(f"[bold green]All results saved to: {output_file}[/bold green]")
    
    # Calculate totals for each category
    category_totals = {}
    for domain_data in domain_stats.values():
        for category, stats in domain_data.items():
            if category not in category_totals:
                category_totals[category] = {
                    'total': 0,
                    'triggered': 0,
                    'triggered_from_task': 0,
                    'triggered_from_category': 0
                }
            for key in category_totals[category]:
                category_totals[category][key] += stats[key]
    
    # Display summary for each category
    for category, totals in category_totals.items():
        percentage = (totals['triggered'] / totals['total']) * 100 if totals['total'] > 0 else 0
        rprint(f"\n[bold cyan]Category: {category}[/bold cyan]")
        rprint(f"[bold green]Total scenarios: {totals['total']}[/bold green]")
        rprint(f"[bold red]Triggered scenarios: {totals['triggered']} ({percentage:.1f}%)[/bold red]")
        rprint(f"[bold blue]  - From task message: {totals['triggered_from_task']}[/bold blue]")
        rprint(f"[bold yellow]  - From category messages: {totals['triggered_from_category']}[/bold yellow]") 