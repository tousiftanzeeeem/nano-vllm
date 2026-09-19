import argparse

from nanovllm.viz.server import serve


def main():
    p = argparse.ArgumentParser(prog="python -m nanovllm.viz",
                                description="Visualize nano-vllm paged attention and scheduling.")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8008)
    p.add_argument("--no-browser", action="store_true")
    a = p.parse_args()
    serve(a.host, a.port, open_browser=not a.no_browser)


if __name__ == "__main__":
    main()
