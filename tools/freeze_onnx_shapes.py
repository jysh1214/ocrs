#!/usr/bin/env python3
"""Rewrite an ONNX model so that OpenCV's DNN module (4.x) can import it and run it at any input size.

OpenCV's ONNX importer resolves shapes once, at import time:

* it cannot evaluate `Shape` on an input of unknown size, which is how the PP-OCR text recognisers compute
  their `Reshape` targets;
* it turns `Squeeze` / `Unsqueeze` into a reshape to whatever shape it saw at import.

This script folds every shape computation into a constant, then removes the dependence on the batch size and
the image width again: `Reshape` targets get the placeholders 0 (copy this dimension from the input) and -1
(infer it), `Squeeze` / `Unsqueeze` become a `Transpose` plus a `Reshape` that only uses placeholders. The
result declares a fixed input shape (N=1, W=320) and still accepts any batch size and width.

Needs only `onnx` and `numpy`:

    freeze_onnx_shapes.py in.onnx out.onnx [--height 48]
"""
import argparse

import numpy as np
import onnx
from onnx import helper, numpy_helper, shape_inference
from onnx.reference import ReferenceEvaluator


def input_of(model):
    initializers = {i.name for i in model.graph.initializer}
    (inp,) = [i for i in model.graph.input if i.name not in initializers]
    return inp


def known_shapes(model):
    """Tensor name -> shape, for every tensor ONNX shape inference fully resolves."""
    graph = shape_inference.infer_shapes(model, data_prop=True).graph
    shapes = {}
    for info in [*graph.value_info, *graph.input, *graph.output]:
        tensor = info.type.tensor_type
        if tensor.HasField("shape") and all(d.HasField("dim_value") for d in tensor.shape.dim):
            shapes[info.name] = [d.dim_value for d in tensor.shape.dim]
    return shapes


def evaluate(node, inputs, model):
    """Outputs of a single node whose inputs are all constants."""
    graph = helper.make_graph(
        [node],
        "fold",
        [],
        [helper.make_empty_tensor_value_info(o) for o in node.output if o],
        [numpy_helper.from_array(value, name) for name, value in inputs.items()],
    )
    single = helper.make_model(graph, opset_imports=list(model.opset_import), ir_version=model.ir_version)
    return ReferenceEvaluator(single).run(None, {})


def fold_pass(model):
    """Replace `Shape` of statically shaped tensors and nodes with constant inputs. True if anything changed."""
    graph = model.graph
    shapes = known_shapes(model)
    consts = {i.name: numpy_helper.to_array(i) for i in graph.initializer}
    kept, folded = [], {}
    for node in graph.node:
        if node.op_type == "Shape" and not node.attribute and node.input[0] in shapes:
            values = [np.array(shapes[node.input[0]], dtype=np.int64)]
        elif all(i == "" or i in consts for i in node.input):
            values = evaluate(node, {i: consts[i] for i in node.input if i}, model)
        else:
            kept.append(node)
            continue
        for name, value in zip([o for o in node.output if o], values):
            consts[name] = folded[name] = np.asarray(value)
    if not folded:
        return False
    del graph.node[:]
    graph.node.extend(kept)
    graph.initializer.extend(numpy_helper.from_array(value, name) for name, value in folded.items())
    return True


def drop_unused_initializers(graph):
    used = {i for node in graph.node for i in node.input} | {o.name for o in graph.output}
    live = [i for i in graph.initializer if i.name in used]
    del graph.initializer[:]
    graph.initializer.extend(live)


def fold(model, shape):
    """Copy of `model` specialised to one concrete input shape, with all shape arithmetic folded away."""
    model = onnx.ModelProto.FromString(model.SerializeToString())
    dims = input_of(model).type.tensor_type.shape.dim
    assert len(dims) == len(shape), f"the model input has {len(dims)} dimensions"
    for dim, value in zip(dims, shape):
        dim.Clear()
        dim.dim_value = value
    for out in model.graph.output:  # stale symbolic output shapes confuse shape inference
        out.type.tensor_type.ClearField("shape")
    del model.graph.value_info[:]
    while fold_pass(model):
        pass
    drop_unused_initializers(model.graph)
    leftover = sorted({n.op_type for n in model.graph.node if n.op_type in ("Shape", "Size")})
    assert not leftover, f"could not fold {leftover}"
    return model


def generalise_reshape_targets(variants):
    """Make the first of several folds of one model (at different input shapes) independent of that shape.

    `variants` are (folded model, tensor shapes) pairs. They may only differ in `Reshape` targets; an entry
    that differs becomes 0 where it always equals the input's dimension at that position, else -1.
    """
    base, _ = variants[0]
    constants = [{i.name: numpy_helper.to_array(i) for i in model.graph.initializer} for model, _ in variants]
    assert all(
        [n.op_type for n in model.graph.node] == [n.op_type for n in base.graph.node] for model, _ in variants
    ), "the folded graphs differ in structure"
    patched = {}
    for index, node in enumerate(base.graph.node):
        for slot, name in enumerate(node.input):
            if name not in constants[0]:
                continue
            values = [c[model.graph.node[index].input[slot]] for c, (model, _) in zip(constants, variants)]
            if all(v.shape == values[0].shape and np.array_equal(v, values[0]) for v in values):
                continue
            assert node.op_type == "Reshape" and slot == 1, f"{node.op_type} input {slot} depends on the input shape"
            sources = [shapes[model.graph.node[index].input[0]] for model, shapes in variants]
            target = values[0].copy()
            for axis in range(len(target)):
                column = [int(v[axis]) for v in values]
                if len(set(column)) > 1:
                    copies = all(axis < len(src) and src[axis] == value for src, value in zip(sources, column))
                    target[axis] = 0 if copies else -1
            assert np.count_nonzero(target == -1) <= 1, f"{node.name}: two inferred dimensions in {target}"
            patched[name] = target
            print(f"  {node.name}: Reshape {values[0].tolist()} -> {target.tolist()}")
    for init in base.graph.initializer:
        if init.name in patched:
            init.CopyFrom(numpy_helper.from_array(patched[init.name], init.name))
    return base


def replace_squeezes(model, shapes):
    """Express `Squeeze` / `Unsqueeze` without naming any dimension that could change with the input.

    Squeeze: move the axes of size 1 to the end, then reshape to [0, ..., 0], which copies the leading
    dimensions and drops the rest. Unsqueeze: append the new axes with a reshape to [0, ..., 0, 1, ...],
    then move them into place.
    """
    graph = model.graph
    constants = {i.name: numpy_helper.to_array(i) for i in graph.initializer}

    def shape_constant(name, values):
        graph.initializer.append(numpy_helper.from_array(np.array(values, dtype=np.int64), name))
        return name

    nodes = []
    for node in graph.node:
        if node.op_type not in ("Squeeze", "Unsqueeze"):
            nodes.append(node)
            continue
        src, dst = node.input[0], node.output[0]
        rank = len(shapes[src])
        axes = [a.ints for a in node.attribute if a.name == "axes"]
        axes = list(axes[0]) if axes else constants[node.input[1]].tolist()
        if node.op_type == "Squeeze":
            axes = sorted(a % rank for a in axes)
            kept = [a for a in range(rank) if a not in axes]
            perm = kept + axes
            moved = src
            if perm != list(range(rank)):
                moved = f"{dst}__moved"
                nodes.append(helper.make_node("Transpose", [src], [moved], perm=perm, name=f"{node.name}__move"))
            target = shape_constant(f"{dst}__shape", [0] * len(kept))
            nodes.append(helper.make_node("Reshape", [moved, target], [dst], name=node.name))
        else:
            out_rank = rank + len(axes)
            axes = sorted(a % out_rank for a in axes)
            # the reshape leaves the existing axes in front and the new ones behind them
            old, new = iter(range(rank)), iter(range(rank, out_rank))
            perm = [next(new) if a in axes else next(old) for a in range(out_rank)]
            target = shape_constant(f"{dst}__shape", [0] * rank + [1] * len(axes))
            if perm == list(range(out_rank)):
                nodes.append(helper.make_node("Reshape", [src, target], [dst], name=node.name))
            else:
                grown = f"{dst}__grown"
                nodes.append(helper.make_node("Reshape", [src, target], [grown], name=node.name))
                nodes.append(helper.make_node("Transpose", [grown], [dst], perm=perm, name=f"{node.name}__move"))
        print(f"  {node.name}: {node.op_type} axes {axes} of a rank {rank} tensor -> Transpose/Reshape")
    del graph.node[:]
    graph.node.extend(nodes)
    drop_unused_initializers(graph)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("src")
    parser.add_argument("dst")
    parser.add_argument("--height", type=int, default=48, help="input height the model was trained for")
    args = parser.parse_args()

    original = onnx.load(args.src)
    h = args.height
    print(f"{args.src}:")
    # The reference shape, one with another width and one with another batch size. The reference shape stays
    # declared on the input: OpenCV gives up on symbolic dimensions at import time.
    variants = [fold(original, shape) for shape in ([1, 3, h, 320], [1, 3, h, 544], [3, 3, h, 320])]
    variants = [(model, known_shapes(model)) for model in variants]
    model = generalise_reshape_targets(variants)
    replace_squeezes(model, variants[0][1])
    del model.graph.output[:]
    model.graph.output.extend(original.graph.output)  # same tensors, with their symbolic shapes
    # Annotate the intermediate tensors: OpenCV only recognises the layer normalisations (and then runs them
    # as one layer its CUDA backend supports) when it knows the rank of their input.
    del model.graph.value_info[:]
    model.graph.value_info.extend(shape_inference.infer_shapes(model).graph.value_info)
    onnx.checker.check_model(model)

    # the rewritten model has to agree with the original on a shape that was not used above
    probe = np.random.default_rng(0).uniform(-1, 1, (2, 3, h, 416)).astype(np.float32)
    name = input_of(original).name
    want = ReferenceEvaluator(original).run(None, {name: probe})[0]
    got = ReferenceEvaluator(model).run(None, {name: probe})[0]
    assert want.shape == got.shape, (want.shape, got.shape)
    error = float(np.abs(want - got).max())
    assert error < 1e-4, f"outputs differ by {error}"
    print(f"  output {list(got.shape)} matches the original (max abs difference {error:.2e})")

    onnx.save(model, args.dst)
    print(f"wrote {args.dst}: {len(model.graph.node)} nodes")


if __name__ == "__main__":
    main()
