import argparse

from utils.image import ImageResizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resize and save images")
    parser.add_argument("-i", "--input_dir", nargs="?", type=str, required=True, help="Input directory")
    parser.add_argument("-o", "--output_dir", nargs="?", type=str, required=True, help="Output directory")
    parser.add_argument("-s", "--size", nargs="?", type=int, required=True, help="Image size")
    args = parser.parse_args()

    resizer = ImageResizer(args.input_dir, args.output_dir, size=args.size, fit_inside=False, rename=True)
    resizer.run()
