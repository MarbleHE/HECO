
import logging
import json
import os

from inspect import getsource, getmodule
from ast import parse

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
        self.abc_ast_json = []

    #
    # Internal helper functions
    #
    def _create_fhe_args_block(self, fhe_args : dict) -> dict:
        """
        Translate a dictionary of variable-value assignments to a block of ABC variable declaration statements.

        :param fhe_args: Dictionary of the form { var_name : (value, is_secret), ... }
        :return: JSON dictionary equivalent to an AST subtree for a block of variable declaration statements.
        """
        stmts = []
        for fhe_arg_name, fhe_arg_val in fhe_args.items():
            abc_var = self.builder.make_variable(fhe_arg_name)
            abc_val = self.builder.make_value(fhe_arg_val[0])
            abc_var_decl = self.builder.make_variable_declaration(abc_var, abc_val, fhe_arg_val[1])
            stmts.append(abc_var_decl)

        return self.builder.make_block(stmts)


    #
    # External functions
    #

    def add_fn(self, fn_json):
        """
        Add the json representation of an ABC AST function to self.abc_ast_json. This function will be added to a
        module in MLIR.

        :param fn_json: json representation of an ABC AST function
        """

        self.abc_ast_json.append(fn_json)

    def compile(self):
        """
        Compile the function to MLIR (but not yet specify input/output values).

        :return: void, the compiled main function is stored internally in self.cpp_program.
        """

        if len(self.abc_ast_json) > 0:
            self.cpp_program = ABCProgramWrapper(json.dumps(self.abc_ast_json[0]))
            for fn in self.abc_ast_json[1:]:
                self.cpp_program.add_fn(json.dumps(fn))

            # TODO: Printing MLIR for the moment, remove when we actually execute it.
            self.cpp_program.dump()
        else:
            logging.error("A Python program must first be parsed to a json representation of the ABC AST before it can "
                          "be compiled")
            exit(1)

    def execute(self, *args, **kwargs):
        """
        Executes the main function with the given variable assignments.
        Raises an error when not all required variables are defined.

        :param kwargs: arguments for the main function. Default values are used for undefined optional variables.
        :return: whatever the defined FHE main function returns.
        """

        # TODO: remove
        logging.warning("Execution is temporarily deactivated until we switched to MLIR!")
        return

        # logging.debug(f"Running main function with args: {self.main_args}")
        #
        # if not self.cpp_program:
        #     self.compile()
        #
        # # Ensure all arguments are present
        # if len(args) > len(self.main_args):
        #     logging.error(f"Too many arguments supplied for main (max. {len(self.main_args)} are allowed).")
        #     exit(1)
        #
        # fhe_args = dict()
        # main_args_keys = list(self.main_args.keys())
        # ## Add positional arguments with names in the order of those specified in self.main_args.
        # for i, arg in enumerate(args):
        #     fhe_args[main_args_keys[i]] = (arg, self.main_args[main_args_keys[i]]["secret"])
        #
        # ## Add keyword arguments (if they are actually arguments of main)
        # for arg_name, arg_val in kwargs.items():
        #     if arg_name not in main_args_keys:
        #         logging.error(f"Unknown argument '{arg_name}' for main.")
        #     fhe_args[arg_name] = (arg_val, self.main_args[arg_name]["secret"])
        #
        # ## Add remaining arguments of main that have default values
        # for arg_name in list(main_args_keys)[len(args):]:
        #     if arg_name not in kwargs:
        #         arg_meta = self.main_args[arg_name]
        #         if not arg_meta["opt"]:
        #             logging.error(f"Mandatory argument '{arg_name}' is missing!")
        #             exit(1)
        #         fhe_args[arg_name] = (arg_meta["value"], arg_meta["secret"])
        #
        # ## Create FHE arguments: make block of assignment statements
        # fhe_args_block = self._create_fhe_args_block(fhe_args)
        #
        # ret_var_names = list(self.ret_vars.keys())
        # result = self.cpp_program.execute(json.dumps(fhe_args_block), ret_var_names)
        #
        # res_vec = []
        # res_val_idx = 0
        # for ret_vars_idx, ret_var in enumerate (ret_var_names):
        #     # Add constants to fill gaps between variables
        #     res_vec += self.ret_constants[res_val_idx : self.ret_vars[ret_var]]
        #     res_val_idx = self.ret_vars[ret_var]
        #
        #     # Add result for variable
        #
        #     # TODO: This is a hack necessary due to python's dynamic (return) types.
        #     #   For now, lists that only have a single value are always converted to that value only.
        #     #   In the future, we maybe could leverage type hints to know what return value python actually expects.
        #     val = result[ret_vars_idx]
        #     if isinstance(val, list) and len(val) == 1:
        #         val = val[0]
        #     res_vec.append(val)
        #
        # # Append remaining constants
        # res_vec += self.ret_constants[res_val_idx:]
        #
        # return res_vec[0] if len(res_vec) == 1 else tuple(res_vec)

    def set_src(self, parent_frame):
        """
        Stores the source information. The context source stores the code of the functions on the call stack,
        the src code stores the source code of the entire file from which the ABCContext was called from.
        """

        self.src_module = getmodule(parent_frame)
        self.src_context = getsource(parent_frame)
        self.src_call_stack_ast = parse(self.src_context)

        self.src_file_path = os.path.realpath(self.src_module.__file__)
        with open(self.src_file_path, "r") as src_fp:
            self.src_code = src_fp.read()
        self.src_code_ast = parse(self.src_code)

    def set_curr_blocks(self, block):
        """
        Stores the source code blocks of the current ABCContext call.
        """
        self.src_curr_blocks = block
        self.src_curr_blocks_asts = list(map(parse, self.src_curr_blocks))

