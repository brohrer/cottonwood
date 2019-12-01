def indent(unindented, n_spaces=2):
    """
    Indent a multi-line string using spaces.
    """
    indent = " " * n_spaces
    newline = "\n" + indent
    return indent + newline.join(unindented.split("\n"))
