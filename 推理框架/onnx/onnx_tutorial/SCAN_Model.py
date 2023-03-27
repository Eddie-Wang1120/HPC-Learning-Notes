import numpy
from onnx import numpy_helper, TensorProto
from onnx.helper import (
    make_model, make_node, set_model_props, make_tensor, make_graph,
    make_tensor_value_info)
from onnx.checker import check_model

# subgraph
initializers = []
nodes = []
inputs = []
outputs = []

value = make_tensor_value_info('next_in', 1, [None, 4])
inputs.append(value)
value = make_tensor_value_info('next', 1, [None])
inputs.append(value)

value = make_tensor_value_info('next_out', 1, [None, None])
outputs.append(value)
value = make_tensor_value_info('scan_out', 1, [None])
outputs.append(value)

node = make_node(
    'Identity', ['next_in'], ['next_out'],
    name='cdistd_17_Identity', domain='')
nodes.append(node)

node = make_node(
    'Sub', ['next_in', 'next'], ['cdistdf_17_C0'],
    name='cdistdf_17_Sub', domain='')
nodes.append(node)

node = make_node(
    'ReduceSumSquare', ['cdistdf_17_C0'], ['cdistdf_17_reduced0'],
    name='cdistdf_17_ReduceSumSquare', axes=[1], keepdims=0, domain='')
nodes.append(node)

node = make_node(
    'Identity', ['cdistdf_17_reduced0'],
    ['scan_out'], name='cdistdf_17_Identity', domain='')
nodes.append(node)

graph = make_graph(nodes, 'OnnxIdentity',
                   inputs, outputs, initializers)

# main graph

initializers = []
nodes = []
inputs = []
outputs = []

opsets = {'': 15, 'ai.onnx.ml': 15}
target_opset = 15  # subgraphs

# initializers
list_value = [23.29599822460675, -120.86516699239603, -144.70495899914215, -260.08772982740413,
              154.65272105889147, -122.23295157108991, 247.45232560871727, -182.83789715805776,
              -132.92727431421793, 147.48710175784703, 88.27761768038069, -14.87785569894749,
              111.71487894705504, 301.0518319089629, -29.64235742280055, -113.78493504731911,
              -204.41218591022718, 112.26561056133608, 66.04032954135549,
              -229.5428380626701, -33.549262642481615, -140.95737409864623, -87.8145187836131,
              -90.61397011283958, 57.185488100413366, 56.864151796743855, 77.09054590340892,
              -187.72501631246712, -42.779503579806025, -21.642642730674076, -44.58517761667535,
              78.56025104939847, -23.92423223842056, 234.9166231927213, -73.73512816431007,
              -10.150864499514297, -70.37105466673813, 65.5755688281476, 108.68676290979731, -78.36748960443065]
value = numpy.array(list_value, dtype=numpy.float64).reshape((2, 20))
tensor = numpy_helper.from_array(
    value, name='knny_ArrayFeatureExtractorcst')
initializers.append(tensor)

list_value = [1.1394007205963135, -0.6848101019859314, -1.234825849533081, 0.4023416340351105,
              0.17742614448070526, 0.46278226375579834, -0.4017809331417084, -1.630198359489441,
              -0.5096521973609924, 0.7774903774261475, -0.4380742907524109, -1.2527953386306763,
              -1.0485529899597168, 1.950775384902954, -1.420017957687378, -1.7062702178955078,
              1.8675580024719238, -0.15135720372200012, -0.9772778749465942, 0.9500884413719177,
              -2.5529897212982178, -0.7421650290489197, 0.653618574142456, 0.8644362092018127,
              1.5327792167663574, 0.37816253304481506, 1.4693588018417358, 0.154947429895401,
              -0.6724604368209839, -1.7262825965881348, -0.35955315828323364, -0.8131462931632996,
              -0.8707971572875977, 0.056165341287851334, -0.5788496732711792, -0.3115525245666504,
              1.2302906513214111, -0.302302747964859, 1.202379822731018, -0.38732680678367615,
              2.269754648208618, -0.18718385696411133, -1.4543657302856445, 0.04575851559638977,
              -0.9072983860969543, 0.12898291647434235, 0.05194539576768875, 0.7290905714035034,
              1.4940791130065918, -0.8540957570075989, -0.2051582634449005, 0.3130677044391632,
              1.764052391052246, 2.2408931255340576, 0.40015721321105957, 0.978738009929657,
              0.06651721894741058, -0.3627411723136902, 0.30247190594673157, -0.6343221068382263,
              -0.5108051300048828, 0.4283318817615509, -1.18063223361969, -0.02818222902715206,
              -1.6138978004455566, 0.38690251111984253, -0.21274028718471527, -0.8954665660858154,
              0.7610377073287964, 0.3336743414402008, 0.12167501449584961, 0.44386324286460876,
              -0.10321885347366333, 1.4542734622955322, 0.4105985164642334, 0.14404356479644775,
              -0.8877857327461243, 0.15634897351264954, -1.980796456336975, -0.34791216254234314]
value = numpy.array(list_value, dtype=numpy.float32).reshape((20, 4))
tensor = numpy_helper.from_array(value, name='Sc_Scancst')
initializers.append(tensor)

value = numpy.array([2], dtype=numpy.int64)
tensor = numpy_helper.from_array(value, name='To_TopKcst')
initializers.append(tensor)

value = numpy.array([2, -1, 2], dtype=numpy.int64)
tensor = numpy_helper.from_array(value, name='knny_Reshapecst')
initializers.append(tensor)

# inputs
value = make_tensor_value_info('input', 1, [None, 4])
inputs.append(value)

# outputs
value = make_tensor_value_info('variable', 1, [None, 2])
outputs.append(value)

# nodes

node = make_node(
    'Scan', ['input', 'Sc_Scancst'], ['UU032UU', 'UU033UU'],
    name='Sc_Scan', body=graph, num_scan_inputs=1, domain='')
nodes.append(node)

node = make_node(
    'Transpose', ['UU033UU'], ['Tr_transposed0'],
    name='Tr_Transpose', perm=[1, 0], domain='')
nodes.append(node)

node = make_node(
    'Sqrt', ['Tr_transposed0'], ['Sq_Y0'],
    name='Sq_Sqrt', domain='')
nodes.append(node)

node = make_node(
    'TopK', ['Sq_Y0', 'To_TopKcst'], ['To_Values0', 'To_Indices1'],
    name='To_TopK', largest=0, sorted=1, domain='')
nodes.append(node)

node = make_node(
    'Flatten', ['To_Indices1'], ['knny_output0'],
    name='knny_Flatten', domain='')
nodes.append(node)

node = make_node(
    'ArrayFeatureExtractor',
    ['knny_ArrayFeatureExtractorcst', 'knny_output0'], ['knny_Z0'],
    name='knny_ArrayFeatureExtractor', domain='ai.onnx.ml')
nodes.append(node)

node = make_node(
    'Reshape', ['knny_Z0', 'knny_Reshapecst'], ['knny_reshaped0'],
    name='knny_Reshape', allowzero=0, domain='')
nodes.append(node)

node = make_node(
    'Transpose', ['knny_reshaped0'], ['knny_transposed0'],
    name='knny_Transpose', perm=[1, 0, 2], domain='')
nodes.append(node)

node = make_node(
    'Cast', ['knny_transposed0'], ['Ca_output0'],
    name='Ca_Cast', to=TensorProto.FLOAT, domain='')
nodes.append(node)

node = make_node(
    'ReduceMean', ['Ca_output0'], ['variable'],
    name='Re_ReduceMean', axes=[2], keepdims=0, domain='')
nodes.append(node)

# graph
graph = make_graph(nodes, 'KNN regressor', inputs, outputs, initializers)

# model
onnx_model = make_model(graph)
onnx_model.ir_version = 8
onnx_model.producer_name = 'skl2onnx'
onnx_model.producer_version = ''
onnx_model.domain = 'ai.onnx'
onnx_model.model_version = 0
onnx_model.doc_string = ''
set_model_props(onnx_model, {})

# opsets
del onnx_model.opset_import[:]  # pylint: disable=E1101
for dom, value in opsets.items():
    op_set = onnx_model.opset_import.add()
    op_set.domain = dom
    op_set.version = value

check_model(onnx_model)
with open("knnr.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())

print(onnx_model)