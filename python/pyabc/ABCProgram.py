
import logging
import json
from ._abc_wrapper import *
from .ABCJsonAstBuilder import ABCJsonAstBuilder

class ABCProgram:
    """
    Class for an ABC FHE program
    """

    def __init__(self, log_level=logging.INFO):
        self.log_level = log_level
        logging.basicConfig(level=self.log_level)

        self.builder = ABCJsonAstBuilder(log_level=log_level)

    #
    # Internal helper functions
    #
    def _create_fhe_args_block(self, fhe_args):
        stmts = []
        for fhe_arg_name, fhe_arg_val in fhe_args.items():
            abc_var = self.builder.make_variable(fhe_arg_name)
            abc_val = self.builder.make_literal(fhe_arg_val)
            abc_var_decl = self.builder.make_variable_declaration(abc_var, abc_val)
            stmts.append(abc_var_decl)

        return self.builder.make_block(stmts)


    #
    # External functions
    #
    def add_main_fn(self, body : dict, args : dict, ret_vars : dict = {}, ret_constants : list = []) -> None:
        """
        Add a main function and its arguments to the program.

        :param body: ABC AST as a JSON-equivalent dictionary
        :param args: Argument dictionary of the format described in ABCVisitor._args_to_dict
        :param ret_vars: A dictionary with variable names as keys and the position in the return vector as value
        :param ret_constants: A list with constant return values. This must be at least as long to cover all gaps in ret_vars.
                              Additional values will be appended to the return vector.
        """

        self.main_fn         = body
        self.main_args       = args
        self.ret_vars        = ret_vars
        self.ret_constants   = ret_constants

        self.compile()

    def compile(self):
        """
        Compile the main function to an executable ABC AST, i.e., parse the JSON representation to an ABC AST
        but not yet specify input/output values.

        :return: void, the compiled main function is stored internally in self.cpp_program.
        """
        if self.main_fn:
            self.cpp_program = IntProgram(json.dumps(self.main_fn))
        else:
            logging.error("Set main function first, before trying to compile!")
            exit(1)

    def execute(self, *args, **kwargs):
        """
        Executes the main function with the given variable assignments.
        Raises an error when not all required variables are defined.

        :param kwargs: arguments for the main function. Default values are used for undefined optional variables.
        :return: whatever the defined FHE main function returns.
        """

        logging.debug(f"Running main function with args: {self.main_args}")

        if not self.cpp_program:
            self.compile()

        # Ensure all arguments are present
        if len(args) > len(self.main_args):
            logging.error(f"Too many arguments supplied for main (max. {len(self.main_args)} are allowed).")
            exit(1)

        fhe_args = dict()
        main_args_keys = list(self.main_args.keys())
        ## Add positional arguments with names in the order of those specified in self.main_args.
        for i, arg in enumerate(args):
            fhe_args[main_args_keys[i]] = arg

        ## Add keyword arguments (if they are actually arguments of main)
        for arg_name, arg_val in kwargs.items():
            if arg_name not in main_args_keys:
                logging.error(f"Unknown argument '{arg_name}' for main.")
            fhe_args[arg_name] = arg_val

        ## Add remaining arguments of main that have default values
        for arg_name in list(main_args_keys)[len(args):]:
            if arg_name not in kwargs:
                arg_meta = self.main_args[arg_name]
                if not arg_meta["opt"]:
                    logging.error(f"Mandatory argument '{arg_name}' is missing!")
                    exit(1)
                fhe_args[arg_name] = arg_meta["value"]

        ## Create FHE arguments: make block of assignment statements
        fhe_args_block = self._create_fhe_args_block(fhe_args)

        ret_var_names = list(self.ret_vars.keys())
        result = self.cpp_program.execute(json.dumps(fhe_args_block), ret_var_names)

        res_vec = []
        res_val_idx = 0
        for ret_vars_idx, ret_var in enumerate (ret_var_names):
            # Add constants to fill gaps between variables
            res_vec += self.ret_constants[res_val_idx : self.ret_vars[ret_var]]
            res_val_idx += self.ret_vars[ret_var] - res_val_idx

            # Add result for variable
            res_vec.append(result[ret_vars_idx])

        # Append remaining constants
        res_vec += self.ret_constants[res_val_idx:]

        return res_vec[0] if len(res_vec) == 1 else tuple(res_vec)
