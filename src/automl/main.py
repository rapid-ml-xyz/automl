#!/usr/bin/env python
import warnings
import argparse
from .st_eda_interface import StreamlitInterface
from .few_shot_crew import FewShot
from .zero_shot_crew import ZeroShot

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")


def parse_args():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description='KaggleAutoML CLI')
    parser.add_argument('command', choices=['explore', 'run'], help='Command to execute')
    parser.add_argument('--action', type=str, required=True, help='Action to be performed')
    parser.add_argument('--max-iterations', type=int, default=5, required=False, help='Maximum number of iterations')
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


def explore(action):
    """Run the exploration interface."""
    interface = StreamlitInterface()
    interface.explore(action)


def main():
    args = parse_args()

    if args.command == 'run':
        run(args.action, args.max_iterations)
    elif args.command == 'explore':
        explore(args.action)
    else:
        raise Exception("Invalid command")


if __name__ == '__main__':
    main()
