import sys


def _usage():
    print("Usage: python -m voltgan <command>")
    print("Commands:")
    print("  pre     Run preprocessing on dataset")
    print("  train   Run model training")
    print("  infer   Run inference on an HDF file")
    sys.exit(1)


if len(sys.argv) < 2:
    _usage()

command = sys.argv.pop(1)

match command:
    case "train":
        from voltgan.train import main

        main()
    case "infer":
        from voltgan.infer import main

        main()
    case "pre":
        from voltgan.pre import main

        main()
    case _:
        print(f"Unknown command: {command!r}")
        _usage()
