import argparse
import signal
import sys
import logging

logging.basicConfig(level=logging.INFO)


def run_controller(freq: float, max_ticks: int | None):
    """Run the original PPE controller loop."""
    from src.core.controller import AgentController

    controller = AgentController(frequency=freq, max_ticks=max_ticks)

    def signal_handler(sig, frame):
        print("\nCtrl+C pressed. Initiating shutdown...")
        controller.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    print(f"Starting Agent at {freq}Hz...")
    controller.spin()


def run_api_server(host: str, port: int):
    """Run the FastAPI server for browser automation."""
    import uvicorn
    from src.api.server import app

    print(f"Starting UI-Agent API server on {host}:{port}")
    uvicorn.run(app, host=host, port=port)


def main():
    parser = argparse.ArgumentParser(description="UI Agent - Controller or API Server")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Controller mode (original behavior)
    controller_parser = subparsers.add_parser("controller", help="Run PPE controller loop")
    controller_parser.add_argument("--freq", type=float, default=1.0, help="Control loop frequency in Hz")
    controller_parser.add_argument("--max-ticks", type=int, default=None, help="Max ticks to run before exit")

    # API server mode (new)
    api_parser = subparsers.add_parser("api", help="Run browser automation API server")
    api_parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind to")
    api_parser.add_argument("--port", type=int, default=8000, help="Port to listen on")

    args = parser.parse_args()

    if args.command == "api":
        run_api_server(args.host, args.port)
    elif args.command == "controller":
        run_controller(args.freq, args.max_ticks)
    else:
        # Default: show help or run controller for backwards compatibility
        if len(sys.argv) == 1:
            parser.print_help()
        else:
            # Try to parse as old-style arguments for backwards compatibility
            old_parser = argparse.ArgumentParser(description="UI Agent Controller")
            old_parser.add_argument("--freq", type=float, default=1.0, help="Control loop frequency in Hz")
            old_parser.add_argument("--max-ticks", type=int, default=None, help="Max ticks to run before exit")
            old_args = old_parser.parse_args()
            run_controller(old_args.freq, old_args.max_ticks)


if __name__ == "__main__":
    main()
