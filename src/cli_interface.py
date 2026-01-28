import cmd
import json

from orchestrator_bot import BotOrchestrator, demo


class BotCLI(cmd.Cmd):
    intro = "Bot Orchestration System CLI. Type 'help' or '?' for commands."
    prompt = "(bot-orchestrator) "

    def __init__(self) -> None:
        super().__init__()
        self.orchestrator = BotOrchestrator(max_workers=3)

    def do_create(self, arg: str) -> None:
        """Create a new task: create <task_type> <json_parameters>
        Example: create data_processor '{"data": [1,2,3], "operation": "sum"}'"""
        try:
            args = arg.split(" ", 1)
            if len(args) < 2:
                print("Usage: create <task_type> <json_parameters>")
                return

            task_type, params_json = args
            parameters = json.loads(params_json)

            task_id = self.orchestrator.create_task(task_type, parameters)
            print(f"Created task: {task_id}")

        except json.JSONDecodeError:
            print("Invalid JSON parameters")
        except Exception as exc:
            print(f"Error: {exc}")

    def do_status(self, arg: str) -> None:
        """Get status of a task: status <task_id>"""
        if not arg:
            print("Please provide a task ID")
            return

        status = self.orchestrator.get_task_status(arg)
        print(json.dumps(status, indent=2))

    def do_list(self, arg: str) -> None:
        """List all tasks."""
        tasks = self.orchestrator.list_tasks()
        print(json.dumps(tasks, indent=2))

    def do_stop(self, arg: str) -> None:
        """Stop a task: stop <task_id>"""
        if not arg:
            print("Please provide a task ID")
            return

        if self.orchestrator.stop_task(arg):
            print(f"Task {arg} stopped")
        else:
            print(f"Failed to stop task {arg}")

    def do_demo(self, arg: str) -> None:
        """Run a demonstration of the system."""
        demo()

    def do_exit(self, arg: str) -> bool:
        """Exit the CLI."""
        print("Shutting down...")
        self.orchestrator.stop_all()
        return True

    def do_quit(self, arg: str) -> bool:
        """Exit the CLI."""
        return self.do_exit(arg)


if __name__ == "__main__":
    BotCLI().cmdloop()
