import sys


def _usage():
    print("Usage: python -m voltgan <command>")
    print("Commands:")
    print("  pre               Run preprocessing on dataset")
    print("  train-generator   Run generator model training")
    print("  train-estimator   Run SoH estimator model training")
    print("  infer-generator   Run generator inference on an HDF file")
    print("  infer-estimator   Run SoH estimator inference on an HDF file")
    sys.exit(1)


if len(sys.argv) < 2:
    _usage()

command = sys.argv.pop(1)

match command:
    case "train-generator":
        from voltgan.train_generator import main

        main()
    case "train-estimator":
        from voltgan.train_estimator import main

        main()
    case "infer-generator":
        from voltgan.infer_generator import main

        main()
    case "infer-estimator":
        from voltgan.infer_estimator import main

        main()
    case "pre":
        from voltgan.pre import main

        main()
    case _:
        print(f"Unknown command: {command!r}")
        _usage()
