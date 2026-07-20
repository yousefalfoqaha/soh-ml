import sys


def _usage():
    print("Usage: python -m voltgan <command>")
    print("Commands:")
    print("  pre               Run preprocessing on dataset")
    print("  train-generator   Run generator model training")
    print("  train-generator-mse  Run generator model training with MSE loss")
    print("  train-estimator   Run SoH estimator model training")
    print("  infer-generator   Run generator inference on an HDF file")
    print("  infer-estimator   Run SoH estimator inference on an HDF file")
    print("  evaluate-estimator  Evaluate SoH estimator across MCU files")
    print("  evaluate-generator  Evaluate sequence generator across MCU files")
    print("  inspect           Inspect structure and contents of an HDF file")
    print("  plot-soh          Plot SoH degradation curve per MCU")
    print("  plot-soh-trajectories  Plot SoH trajectories across all MCUs")
    sys.exit(1)


if len(sys.argv) < 2:
    _usage()

command = sys.argv.pop(1)

match command:
    case "train-generator":
        from voltgan.train_generator import main

        main()
    case "train-generator-mse":
        from voltgan.train_generator_mse import main

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
    case "evaluate-estimator":
        from voltgan.evaluate_estimator import main

        main()
    case "evaluate-generator":
        from voltgan.evaluate_generator import main

        main()
    case "pre":
        from voltgan.pre import main

        main()
    case "inspect":
        from voltgan.inspect import main

        main()
    case "plot-soh":
        from voltgan.plot_soh import main

        main()
    case "plot-soh-trajectories":
        from voltgan.plot_soh_trajectories import main

        main()
    case _:
        print(f"Unknown command: {command!r}")
        _usage()
