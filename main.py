import argparse
import signal
import sys
import logging
from core.controller import AgentController

def main():
    parser = argparse.ArgumentParser(description="UI Agent Controller")
    parser.add_argument("--freq", type=float, default=1.0, help="Control loop frequency in Hz")
    parser.add_argument("--max-ticks", type=int, default=None, help="Max ticks to run before exit")
    args = parser.parse_args()

    controller = AgentController(frequency=args.freq, max_ticks=args.max_ticks)

    def signal_handler(sig, frame):
        print("\nCtrl+C pressed. Initiating shutdown...")
        controller.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    print(f"Starting Agent at {args.freq}Hz...")
    controller.spin()

if __name__ == "__main__":
    main()
