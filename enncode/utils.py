from itertools import product

def _node_to_string(node):
    """
    Generates a string representation of a node.

    Args:
        node (dict): A dictionary describing the ONNX node. Expected to have the following keys:
        "name", "type", "input", "output", "attributes", "initializers", and "constants".

    Returns:
        str: A string detailing the node's name, type, inputs, outputs, and attributes.
    """
    name = node.get("name")
    type = node.get("type")
    inputs = ", ".join(f"name: {inp['name']}, shape: {inp.get('shape')}" for inp in node.get("input"))
    outputs = ", ".join(f"name: {out['name']}, shape: {out.get('shape')}" for out in node.get("output"))
    attributes_str = ", ".join(f"name: {attr['name']}, value: {attr['value']}"for attr in node.get("attributes"))

    return (
        f"Node(Name: {name}, Type: {type}, "
        f"Inputs: [{inputs}], Outputs: [{outputs}], "
        f"Attributes: {{{attributes_str}}})"
    )

def _extract_shape(tensor):
    """
    Extracts the shape from an ONNX tensor, excluding the batch size dimension.

    Args:
        tensor (onnx.TensorProto): An ONNX tensor protocol buffer object from which the shape is extracted.

    Returns:
        list: A list representing the shape of the tensor, excluding the first
        dimension representing the batch size. If the resulting shape is a single
        dimension, it is returned as a one-element list.
    """
    shape = [dim.dim_value for dim in tensor.type.tensor_type.shape.dim[1:]]  # Exclude batch size
    return shape if len(shape) > 1 else [shape[0]]

def _get_data_type(scalar_data_type):
    """
    Maps an integer data type code to a corresponding struct format character
    used in Python's `struct` module.

    Args:
        scalar_data_type (int): An integer representing the ONNX data type code.

    Returns:
        str: The character that `struct.unpack` would use for this data type.

    Raises:
        ValueError: If the provided data type code is unsupported by this library.
    """
    data_types = {
        1: 'f',   # FLOAT
        2: 'B',   # UINT8
        3: 'b',   # INT8
        4: 'H',   # UINT16
        5: 'h',   # INT16
        6: 'i',   # INT32
        7: 'q',   # INT64
        10: 'e',  # FLOAT16
        11: 'd',  # DOUBLE
        12: 'I',  # UINT32
        13: 'Q',  # UINT64
    }
    if scalar_data_type in data_types:
        return data_types[scalar_data_type]
    else:
        raise ValueError(f"Unsupported data type: {scalar_data_type}")
    
def _unsqueeze_shape(input_shape, axes):
    """
    Used in the Unsqueeze node parser.
    Inserts singleton dimensions into the input shape at specified axes.

    Negative axes are adjusted based on the current length of the shape list.
    This method returns a new shape reflecting the additional dimensions.

    Args:
        input_shape (list): The original shape of the tensor as a list of ints.
        axes (list): A list of integer axes indicating where to insert new dimensions.
            Negative values are interpreted relative to the end of the shape.

    Returns:
        list: A new list representing the shape after inserting singleton
        dimensions at the specified axes.
    """
    output_shape = input_shape.copy()
    for axis in sorted(axes):
        if axis < 0:
            axis += len(output_shape) + 1
        output_shape.insert(axis, 1)
    return output_shape


def _generate_indices(shape):
    """
    Generates index tuples for multi-dimensional Gurobi variables.

    Args:
        shape (list or int): The shape of the tensor as a list of dimensions,
            or a single integer if it is 1-dimensional.

    Returns:
        An iterator that yields each index tuple (or single index if the tensor
        is one-dimensional).

    Raises:
        ValueError: If the shape is neither an integer nor a list/tuple of integers.
    """
    if isinstance(shape, int):
        return range(shape)
    elif isinstance(shape, (list, tuple)):
        return product(*[range(dim) for dim in shape])
    else:
        raise ValueError("Shape must be an integer or a list/tuple of integers.")