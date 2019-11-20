import os
import inspect

from .execution_plan import get_primitive_path


def get_method_arguments(method):
    return set(inspect.signature(method).parameters.keys())


def convert_pipeline_to_script(steps):
    with open(os.path.join(os.path.dirname(__file__), 'pipeline_script.template')) as f:
        pipeline_script_template = f.read()

    primitives_code = ''
    for step_index, step in enumerate(steps):
        primitive_code = ''

        # import code
        primitive_path = get_primitive_path(step['primitive']['name'])
        module_name, class_name = '.'.join(primitive_path.split('.')[:-1]), primitive_path.split('.')[-1]
        primitive_code += 'from {} import {}\n'.format(module_name, class_name)

        # primitive constructor code
        parameters = {key: value for key, value in step['primitive'].get('humanReadableParameters', {}).items()}
        if primitive_path.startswith('blinded'):
            primitive_code += 'parameters = {}\n'
            for key, value in parameters.items():
                primitive_code += "parameters[{}] = {}\n".format(repr(key), repr(value))
            primitive_code += 'primitive = {class_name}(**parameters)\n'.format(class_name=class_name)
        else:
            primitive_code += 'parameters = {}\n'
            for key, value in parameters.items():
                primitive_code += "parameters[{}] = {}\n".format(repr(key), repr(value))
            primitive_code += 'primitive = PrimitiveWrapper({class_name}(**parameters))\n'.format(class_name=class_name)
        # primitive arguments
        step_inputs = {}
        for key, value in step['inputs'].items():
            if value.startswith('inputs.'):
                step_inputs[key] = 'dataset'
            else:
                step_inputs[key] = 'step_{}_output'.format(int(value[6:]))

        training_arguments = {}
        produce_arguments = {}
        if primitive_path.startswith('blinded'):
            training_arguments = {}
            for argument, value in step_inputs.items():
                produce_arguments[argument] = value
        else:
            for argument, value in step_inputs.items():
                if argument == 'inputs':
                    training_arguments[argument] = value
                    produce_arguments[argument] = value
                else:
                    training_arguments[argument] = value

        # primitive train code
        if len(training_arguments) > 0:
            primitive_code += 'primitive.set_training_data({})\n'.format(
                ', '.join('{}={}'.format(key, value) for key, value in training_arguments.items()))
            primitive_code += 'primitive.fit()\n'

        # primitive produce code
        if len(produce_arguments) > 0:
            primitive_code += 'step_{}_output = primitive.produce({}).value\n'.format(
                step_index, ', '.join('{}={}'.format(key, value) for key, value in produce_arguments.items()))

        primitives_code += primitive_code + '\n'

    return pipeline_script_template.format(primitives_code=primitives_code)
