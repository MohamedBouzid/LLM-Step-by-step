class MyApp:
    
    def __init__(self, config):
        self.config = config

    def run(self):
        print("App is running with", self.config)


def parse_args():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="dev")
    return p.parse_args()


def main(argv=None):
    """Entry point usable by CLI, tests, or import."""
    args = parse_args()
    app = MyApp(config=args.config)
    app.run()


if __name__ == "__main__":
    main()