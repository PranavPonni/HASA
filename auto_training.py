import argparse
import importlib

def main():
 
    parser = argparse.ArgumentParser(description='Run a Python script.')
    parser.add_argument('--param', type=str, help='model.file_name')

    args = parser.parse_args()

    module_name = args.param

    module = importlib.import_module("auto."+module_name)

    if hasattr(module, 'main'):
        module.main()
    else:
        print(f"No main function in {args.filename}")

if __name__ == "__main__":
    main()