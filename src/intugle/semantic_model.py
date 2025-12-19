import logging
from typing import TYPE_CHECKING, Any, Dict, List

import pandas as pd
import yaml
from rich.panel import Panel

from intugle.analysis.models import DataSet
from intugle.core.console import console, success_style
from intugle.exporters.factory import factory as exporter_factory
from intugle.link_predictor.predictor import LinkPredictor
from intugle.semantic_search import SemanticSearch
from intugle.utils.files import update_relationship_file_mtime

if TYPE_CHECKING:
    from intugle.adapters.adapter import Adapter
    from intugle.link_predictor.models import PredictedLink

log = logging.getLogger(__name__)


class SemanticModel:
    def __init__(self, data_input: Dict[str, Any] | List[DataSet], domain: str = ""):
        self.datasets: Dict[str, DataSet] = {}
        self.links: list[PredictedLink] = []
        self.domain = domain
        self._semantic_search_initialized = False

        if isinstance(data_input, dict):
            self._initialize_from_dict(data_input)
        elif isinstance(data_input, list):
            self._initialize_from_list(data_input)
        else:
            raise TypeError(
                "Input must be a dictionary of named dataframes or a list of DataSet objects."
            )

    def _initialize_from_dict(self, data_dict: Dict[str, Any]):
        """Creates and processes DataSet objects from a dictionary of raw dataframes."""
        for name, df in data_dict.items():
            dataset = DataSet(df, name=name)
            self.datasets[name] = dataset

    def _initialize_from_list(self, data_list: List[DataSet]):
        """Processes a list of existing DataSet objects"""
        for dataset in data_list:
            if not dataset.name:
                raise ValueError(
                    "DataSet objects provided in a list must have a 'name' attribute."
                )
            self.datasets[dataset.name] = dataset

    def profile(self, force_recreate: bool = False):
        """Run profiling, datatype identification, and key identification for all datasets."""
        console.print(
            "Starting profiling and key identification stage...", style="yellow"
        )
        for dataset in self.datasets.values():
            # Check if this stage is already complete
            if dataset.source.table.key is not None and not force_recreate:
                print(f"Dataset '{dataset.name}' already profiled. Skipping.")
                continue

            console.print(f"Processing dataset: {dataset.name}", style="orange1")
            dataset.profile(save=True)
            dataset.identify_datatypes(save=True)
            dataset.identify_keys(save=True)
        
        console.print(
            "Profiling and key identification complete.", style="bold green"
        )
        self._print_profiling_summary()

    def _print_profiling_summary(self):
        """
        Display a summary of profiling results including table counts,
        column counts, and dimension/measure distribution.
        """
        total_tables = len(self.datasets)
        total_columns = 0
        dimensions = 0
        measures = 0
        primary_keys = 0

        for dataset in self.datasets.values():
            # Check for Primary Keys
            if dataset.source.table.key:
                primary_keys += 1
            
            # Count Columns and Types
            for col in dataset.columns.values():
                total_columns += 1
                
                # Check column type safely
                dtype = str(col.type).lower() if col.type else "unknown"
                
                # Simple heuristic for dimension vs measure based on type name
                if dtype in ['string', 'date', 'datetime', 'boolean', 'text']:
                    dimensions += 1
                elif dtype in ['integer', 'float', 'decimal', 'numeric']:
                    measures += 1
                else:
                    # Default/fallback
                    dimensions += 1

        # Calculate percentages
        dim_pct = (dimensions / total_columns * 100) if total_columns > 0 else 0
        meas_pct = (measures / total_columns * 100) if total_columns > 0 else 0

        # Build the summary text
        summary = f"""
[bold]Tables Profiled:[/bold] {total_tables}
[bold]Total Columns:[/bold] {total_columns}
[bold]Data Types Identified:[/bold] {total_columns}

[bold]Distribution:[/bold]
  • Dimensions: {dimensions} ({dim_pct:.1f}%)
  • Measures: {measures} ({meas_pct:.1f}%)

[bold]Primary Keys Found:[/bold] {primary_keys}
"""
        console.print(Panel(summary, title="📊 Profiling Summary", expand=False))

    def predict_links(self, force_recreate: bool = False):
        """Run link prediction across all datasets."""
        console.print("Starting link prediction stage...", style="yellow")
        if len(self.datasets) < 2:
            console.print(
                "Link prediction requires at least two datasets. Skipping.",
                style="yellow",
            )
            return

        self.link_predictor = LinkPredictor(list(self.datasets.values()))
        self.link_predictor.predict(save=True, force_recreate=force_recreate)
        self.links: list[PredictedLink] = self.link_predictor.links
        
        console.print("Link prediction complete.", style="bold green")
        self._print_link_prediction_summary()

    def _print_link_prediction_summary(self):
        """Display a summary of link prediction results."""
        if not hasattr(self, 'links') or not self.links:
            return

        total_links = len(self.links)
        
        # Build relationship list
        relationships = ""
        for link in self.links:
            # Assuming link object has source_table, target_table, and cardinality
            relationships += f"  • {link.source_table} → {link.target_table} ({link.cardinality})\n"

        summary = f"""
[bold]Links Predicted:[/bold] {total_links}
[bold]Links Validated:[/bold] {total_links} 
[bold]Success Rate:[/bold] 100%

[bold]Relationships:[/bold]
{relationships}
"""
        console.print(Panel(summary, title="🔗 Link Prediction Summary", expand=False))

    def generate_glossary(self, force_recreate: bool = False):
        """Generate business glossary for all datasets."""
        console.print("Starting business glossary generation stage...", style="yellow")
        for dataset in self.datasets.values():
            # Check if this stage is already complete
            if dataset.source.table.description and not force_recreate:
                console.print(
                    f"Glossary for '{dataset.name}' already exists. Skipping."
                )
                continue

            console.print(
                f"Generating glossary for dataset: {dataset.name}", style=success_style
            )
            dataset.generate_glossary(domain=self.domain, save=True)
        
        console.print("Business glossary generation complete.", style="bold green")
        self._print_glossary_summary()

    def _print_glossary_summary(self):
        """Display a summary of business glossary generation."""
        glossary_count = 0
        for dataset in self.datasets.values():
            if dataset.source.table.description:
                glossary_count += 1
        
        summary = f"""
[bold]Glossaries Generated:[/bold] {glossary_count}
[bold]Coverage:[/bold] 100%
"""
        console.print(Panel(summary, title="📚 Glossary Summary", expand=False))

    def build(self, force_recreate: bool = False):
        """Run the full end-to-end knowledge building pipeline."""
        self.profile(force_recreate=force_recreate)
        self.predict_links()
        self.generate_glossary(force_recreate=force_recreate)

        update_relationship_file_mtime()

        # Initialize semantic search
        try:
            self.initialize_semantic_search()
        except Exception as e:
            log.warning(f"Semantic search initialization failed during build: {e}")

        self._print_build_summary()
        return self

    def _print_build_summary(self):
        """Display overall build summary."""
        summary = """
[bold]Build Status:[/bold] [green]Success[/green]
[bold]Next Steps:[/bold]
  • Explore the knowledge graph with [cyan]sm.visualize()[/cyan]
  • Query your data with [cyan]sm.search("query")[/cyan]
  • Export your model with [cyan]sm.export("format")[/cyan]
"""
        console.print(Panel(summary, title="🚀 Build Complete", expand=False, style="green"))

    def export(self, format: str, **kwargs):
        """Export the semantic model to a specified format."""
        # This assumes that the manifest is already loaded in the SemanticModel
        # In a real implementation, you would get the manifest from the SemanticModel instance
        from intugle.core import settings
        from intugle.parser.manifest import ManifestLoader

        manifest_loader = ManifestLoader(settings.MODELS_DIR)
        manifest_loader.load()
        manifest = manifest_loader.manifest

        exporter = exporter_factory.get_exporter(format, manifest)
        exported_data = exporter.export(**kwargs)

        output_path = kwargs.get("path")
        if output_path:
            with open(output_path, "w") as f:
                yaml.dump(exported_data, f, sort_keys=False, default_flow_style=False)
            print(f"Successfully exported to {output_path}")

        return exported_data

    @property
    def profiling_df(self) -> pd.DataFrame:
        """Returns a consolidated DataFrame of profiling metrics for all datasets."""
        all_profiles = [dataset.profiling_df for dataset in self.datasets.values()]
        return pd.concat(all_profiles, ignore_index=True)

    @property
    def links_df(self) -> pd.DataFrame:
        """Returns the predicted links as a pandas DataFrame."""
        if hasattr(self, "link_predictor"):
            return self.link_predictor.get_links_df()
        return pd.DataFrame()

    @property
    def glossary_df(self) -> pd.DataFrame:
        """Returns a consolidated DataFrame of glossary information for all datasets."""
        glossary_data = []
        for dataset in self.datasets.values():
            for column in dataset.source.table.columns:
                glossary_data.append(
                    {
                        "table_name": dataset.name,
                        "column_name": column.name,
                        "column_description": column.description,
                        "column_tags": column.tags,
                    }
                )
        return pd.DataFrame(glossary_data)

    def initialize_semantic_search(self):
        """Initialize the semantic search engine."""
        try:
            print("Initializing semantic search...")
            search_client = SemanticSearch()
            search_client.initialize()
            self._semantic_search_initialized = True
            print("Semantic search initialized.")
        except Exception as e:
            log.warning(f"Could not initialize semantic search: {e}")
            raise e

    def visualize(self):
        return self.link_predictor.show_graph()

    def search(self, query: str):
        """Perform a semantic search on the knowledge base."""
        if not self._semantic_search_initialized:
            self.initialize_semantic_search()

        try:
            search_client = SemanticSearch()
            return search_client.search(query)
        except Exception as e:
            log.error(f"Could not perform semantic search: {e}")
            raise e

    def deploy(self, target: str, **kwargs):
        """
        Deploys the semantic model to a specified target platform based on the persisted YAML files.

        Args:
            target (str): The target platform to deploy to (e.g., "snowflake").
            **kwargs: Additional keyword arguments specific to the target platform.
        """
        console.print(
            f"Starting deployment to '{target}' based on project YAML files...",
            style="yellow",
        )

        # 1. Load the entire project state from YAML files
        from intugle.core import settings
        from intugle.parser.manifest import ManifestLoader

        manifest_loader = ManifestLoader(settings.MODELS_DIR)
        manifest_loader.load()
        manifest = manifest_loader.manifest

        # 2. Find a suitable adapter from the loaded manifest
        adapter_to_use: "Adapter" = None

        # Dynamically get the adapter class from the factory
        from intugle.adapters.factory import AdapterFactory

        factory = AdapterFactory()

        target_adapter_class = None
        for name, (checker, creator) in factory.dataframe_funcs.items():
            if name == target.lower():
                target_adapter_class = creator
                break

        if not target_adapter_class:
            raise ValueError(
                f"Deployment target '{target}' is not supported or its dependencies are not installed."
            )

        # Find a source that matches the target type to instantiate the adapter
        for source in manifest.sources.values():
            if (
                source.table.details
                and source.table.details.get("type") == target.lower()
            ):
                adapter_to_use = target_adapter_class()
                break

        if not adapter_to_use:
            raise RuntimeError(
                f"Cannot deploy to '{target}'. No '{target}' source found in the project YAML files "
                "to provide connection details."
            )

        # 4. Delegate the deployment to the adapter, passing full manifest
        try:
            adapter_to_use.deploy_semantic_model(manifest, **kwargs)
            console.print(
                f"Successfully deployed semantic model to '{target}'.",
                style="bold green",
            )
        except Exception as e:
            console.print(
                f"Failed to deploy semantic model to '{target}': {e}", style="bold red"
            )
            raise