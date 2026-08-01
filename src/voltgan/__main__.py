import sys

_VALID = {"ingest", "dataset", "inspect", "stats", "train", "finetune", "evaluation"}

_USAGE = """\
Usage: python -m voltgan <command>

Commands:
  ingest                      Run data ingestion pipeline (Wuppertal + Oxford)
  dataset                     Generate dataset statistics tables and figures
  inspect <hdf-rel-path>      Inspect structure and contents of an HDF file
  stats                       Calculate training statistics
  train                       Train a model
  finetune                    Finetune the model on the Oxford dataset
  evaluation                  Run estimator evaluation + PFI + Oxford tables/charts
"""


def _main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in _VALID:
        print(_USAGE)
        sys.exit(1)

    command = sys.argv[1]
    sys.argv = [sys.argv[0], *sys.argv[2:]]

    match command:
        case "ingest":
            from voltgan.cli.ingest import main as _main_fn
        case "dataset":
            from voltgan.cli.dataset import main as _main_fn
        case "inspect":
            from voltgan.cli.inspect import main as _main_fn
        case "stats":
            from voltgan.cli.stats import main as _main_fn
        case "train":
            from voltgan.cli.train import main as _main_fn
        case "finetune":
            from voltgan.cli.finetune import main as _main_fn
        case "evaluation":
            from voltgan.cli.evaluation import main as _main_fn
        case _:
            print(f"Unknown command: {command!r}")
            sys.exit(1)

    _main_fn()


if __name__ == "__main__":
    _main()
