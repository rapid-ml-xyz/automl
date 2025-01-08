import sys
from src.automl.streamlit_interface import StreamlitInterface


def main():
    action = sys.argv[1] if len(sys.argv) > 1 else "default_action"
    interface = StreamlitInterface()
    interface.explore(action)


if __name__ == "__main__":
    main()
