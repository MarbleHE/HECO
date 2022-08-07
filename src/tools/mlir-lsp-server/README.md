# Extended MLIR: Language Server Protocol Server

This is an extended version of the
mlir-lsp-server(https://mlir.llvm.org/docs/Tools/MLIRLSP/#supporting-custom-dialects-and-passes),
extending the MLIR language server protocol to support the dialects defined in this project.

In order to use this with the [MLIR extension for vscode](https://marketplace.visualstudio.com/items?itemName=llvm-vs-code-extensions.vscode-mlir),
you will need to add `"mlir.server_path"`  to your workspace's `.vscode/settings.json` and set it to the path of the built executable,
e.g. `"mlir.server_path" : "build/bin/mlir-lsp-server-custom"`