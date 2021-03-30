import numpy as np
from ..optimizer import OptimizerPass

class InsertCopyLayer(OptimizerPass):
    def match(self, node, lastnodes):
        # Match on the layer that comes after the input node
        return node.get_input_node().__class__.__name__ == 'Input' and node.get_attr('class_name') == 'Conv2D'

    def transform(self, model, node, lastnodes):
        # Add a check if we should do this or not by detecting if we are using
        # Phil's implementation. In the final version, this should be
        # done with model.config object 

        # This will be the input
        input_var = node.get_input_variable()

        # Make some attributes for the copy layer required by its config
        # e.g., the size of the stream, maybe something else
        attrs = {
            'class_name' : 'Copy',
            'data_format' : node.get_attr('data_format'),
            'n_chan' : node.get_attr('n_chan'),
            'n_elem' : node.get_attr('in_width') * node.get_attr('in_height')
        }

        # Insert new Copy node above our layer
        copy_layer = model.make_node('Copy', 'cpy_' + node.name, attrs, node.inputs.copy())
        # Ensure precision remains the same
        copy_layer.get_output_variable().type.precision = node.get_input_variable().type.precision
        model.insert_node(copy_layer)

        # Return True because we changed the graph
        return True