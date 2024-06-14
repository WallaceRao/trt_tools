import numpy as np
import tensorrt as trt
import os
import pycuda.driver as cuda
import argparse
 
 
def GiB(val):
    return val * 1 << 30
 
def ONNX_build_engine(onnx_file_path, write_engine=True):
    # :return: engine
 
    G_LOGGER = trt.Logger(trt.Logger.WARNING)
    explicit_batch = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    batch_size = 64
    with trt.Builder(G_LOGGER) as builder, builder.create_network(explicit_batch) as network, trt.OnnxParser(network,
                                                                                                             G_LOGGER) as parser:
        #builder.max_batch_size = batch_size
        config = builder.create_builder_config()
        #config.max_workspace_size = GiB(2)
        config.set_flag(trt.BuilderFlag.FP16)
        print('Loading ONNX file from path {}...'.format(onnx_file_path))
        with open(onnx_file_path, 'rb') as model:
            print('Beginning ONNX file parsing')
            parser.parse(model.read())
        print('Completed parsing of ONNX file')
        print('Building an engine from file {}; this may take a while...'.format(onnx_file_path))
        profile = builder.create_optimization_profile()
        profile.set_shape("mel", (1, 937, 100), (16, 937, 100), (64, 937, 100))
        profile.set_shape("conditioner", (1, 937, 384), (16, 937, 384), (64, 937, 384))
        profile.set_shape("timestep", (1, ), (16, ), (64,))
        config.add_optimization_profile(profile)
 
        engine = builder.build_engine(network, config)
        print("Completed creating Engine")
        if write_engine:
            engine_file_path = 'efficientnet_b1.trt'
            with open(engine_file_path, "wb") as f:
                f.write(engine.serialize())
        return engine
 
 
onnx_file_path = r'skipnet_0712.onnx'
onnx_file_path = r'model2.onnx'
onnx_file_path = r'noise_model.onnx'
# onnx_file_path = r'mobileone_0713.onnx'
write_engine = True
engine = ONNX_build_engine(onnx_file_path, write_engine)
with open('noise_model.engine', mode='wb') as f:
    f.write(bytearray(engine.serialize()))
    print("generating file done!")
