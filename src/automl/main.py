#!/usr/bin/env python
import warnings
import argparse
from automl.few_shot_crew import FewShot
from automl.zero_shot_crew import ZeroShot

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")


def parse_args():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description='KaggleAutoML CLI')
    parser.add_argument('command', choices=['run'], help='Command to execute')
    parser.add_argument('--action', type=str, required=True, help='Action to be performed')
    parser.add_argument('--max-iterations', type=int, default=5, required=True, help='Maximum number of iterations')
    return parser.parse_args()


# TODO: Add smart stop condition
def run(action, max_iterations=5):
    """
    Run the crew.
    """
    inputs = {'topic': action}
    ZeroShot().crew().kickoff(inputs=inputs)

    iteration_count = 0
    few_shot_crew = FewShot().crew()

    while iteration_count < max_iterations:
        few_shot_crew.kickoff()
        iteration_count += 1

    if iteration_count == max_iterations:
        print("Maximum iterations reached. Exiting.")


def main():
    args = parse_args()

    if args.command == 'run':
        run(args.action, args.max_iterations)
    else:
        raise Exception("Invalid command")


if __name__ == '__main__':
    main()
