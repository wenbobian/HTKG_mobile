import mxnet as mx

from collections import namedtuple
Batch = namedtuple('Batch', ['data'])

ctx = mx.cpu()
sym, arg_params, aux_params = mx.model.load_checkpoint('mobilenet_v2', 0)
all_layers = sym.get_internals()
# print all_layers.list_outputs()[-10:]

fe_sym = all_layers['pool6_output']
fe_mod = mx.mod.Module(symbol=fe_sym, context=ctx, label_names=None)
fe_mod.bind(for_training=False, data_shapes=[('data', (1,3,224,224))])
fe_mod.set_params(arg_params, aux_params)

img = mx.image.imread('./cat.jpg')
print type(img)
img = mx.image.imresize(img, 224, 224) # resize
img = img.transpose((2, 0, 1)) # Channel first
img = img.expand_dims(axis=0) # batchify
fe_mod.forward(Batch([img]))
features = fe_mod.get_outputs()[0]
pedestrian_feature = []
for i in xrange(len(features.asnumpy()[0])):
    pedestrian_feature.append(features.asnumpy()[0][i][0][0])
print pedestrian_feature

