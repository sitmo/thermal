"""Console script for thermal."""

import click


@click.command()
def main():
    """Main entrypoint."""
    click.echo("thermal")
    click.echo("=" * len("thermal"))
    click.echo("Surrogate timeseries generation")


if __name__ == "__main__":
    main()  # pragma: no cover
