
import logging
import json

class ABCProgram:
    """
    Class for an ABC FHE program
    """

    def __init__(self, log_level=logging.INFO):
        self.log_level = log_level
        logging.basicConfig(level=self.log_level)

    def add_main_fn(self, body : dict, args : dict) -> None:
        """
        Add a main function and its arguments to the program.

        :param body: ABC AST as a JSON-equivalent dictionary
        :param args: Argument dictionary of the format described in ABCVisitor._args_to_dict
        """

        self.main_fn    = body
        self.main_args  = args

    def run(self, **kwargs):
        """
        Executes the main function with the given variable assignments.
        Raises an error when not all required variables are defined.

        :param kwargs: arguments for the main function. Default values are used for undefined optional variables.
        :return: whatever the defined FHE main function returns.
        """

        # TODO
        logging.debug(
            f"Running main function:\n\targs: {self.main_args}"
            f"\n\tmain AST:\n{json.dumps(self.main_fn, indent=2)}"
        )
