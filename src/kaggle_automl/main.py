#!/usr/bin/env python
import warnings
import argparse
from kaggle_automl.zero_shot_crew import ZeroShot

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")


def parse_args():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description='KaggleAutoML CLI')
    parser.add_argument('command', choices=['run'], help='Command to execute')
    parser.add_argument('--topic', type=str, required=False, help='Topic for analysis')
    return parser.parse_args()


def run(topic):
    """
    Run the crew.
    """
    inputs = {'topic': topic}
    ZeroShot().crew().kickoff(inputs=inputs)


def main():
    args = parse_args()

    if args.command == 'run':
        run(args.topic)
    else:
        raise Exception("Invalid command")


if __name__ == '__main__':
    main()
