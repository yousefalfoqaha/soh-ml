import sys


def _usage():
    print("Usage: python -m voltgan <command>")
    print("Commands:")
    print("  train   Run model training")
    print("  infer   Run inference on an HDF file")
    sys.exit(1)


if len(sys.argv) < 2:
    _usage()

command = sys.argv.pop(1)

if command == "train":
    from voltgan.train import main

    main()
elif command == "infer":
    from voltgan.infer import main

    main()
else:
    print(f"Unknown command: {command!r}")
    _usage()
