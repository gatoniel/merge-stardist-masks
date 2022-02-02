"""Command-line interface."""
import click


@click.command()
@click.version_option()
def main() -> None:
    """Merge Stardist Masks."""


if __name__ == "__main__":
    main(prog_name="merge-stardist-masks")  # pragma: no cover
