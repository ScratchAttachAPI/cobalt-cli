"""Export a half-width, full-depth TinyGPT as a native Scratch project."""

import argparse
import hashlib
import json
import math
import zipfile
from pathlib import Path


EMBED_DIM = 256
NUM_HEADS = 8
FFN_DIM = 1024
NUM_LAYERS = 8
CONTEXT_LENGTH = 128
RUNTIME_CONTEXT_LENGTH = CONTEXT_LENGTH
VOCAB_SIZE = 14_371
OUTPUT_VOCAB_SIZE = VOCAB_SIZE
GENERATION_TOKENS = 20
TEMPERATURE = 0.8


def stable_id(text):
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:20]


def costume(svg, name, center_x, center_y):
    asset_id = hashlib.md5(svg).hexdigest()
    return {
        "name": name,
        "bitmapResolution": 1,
        "dataFormat": "svg",
        "assetId": asset_id,
        "md5ext": f"{asset_id}.svg",
        "rotationCenterX": center_x,
        "rotationCenterY": center_y,
    }


class Graph:
    def __init__(self, tensor_info):
        self.tensor_info = tensor_info
        self.nodes = []

    def emit(self, opcode, *args):
        index = len(self.nodes)
        self.nodes.append((opcode, args))
        return ("ref", index)

    def ref(self, index):
        return ("ref", index)

    def variable(self, name):
        return self.emit("variable", name)

    def weight(self, name, flat_index):
        info = self.tensor_info[name]
        if isinstance(flat_index, int):
            weight_index = info["offset"] + flat_index + 1
        else:
            weight_index = self.emit("add", flat_index, info["offset"])
        raw = self.emit("weight_item", weight_index)
        return self.emit("multiply", raw, info["scale"])


def add(graph, left, right):
    return graph.emit("add", left, right)


def multiply(graph, left, right):
    return graph.emit("multiply", left, right)


def divide(graph, left, right):
    return graph.emit("divide", left, right)


def matvec(graph, weight_name, bias_name, inputs, output_count, input_count):
    output = []
    for row in range(output_count):
        total = graph.weight(bias_name, row)
        for column in range(input_count):
            product = multiply(graph, graph.weight(weight_name, row * input_count + column), inputs[column])
            total = add(graph, total, product)
        output.append(total)
    return output


def layer_norm(graph, values, prefix):
    count = len(values)
    mean = divide(graph, sum_graph(graph, values), count)
    squared_deltas = []
    for value in values:
        delta = graph.emit("subtract", value, mean)
        squared_deltas.append(multiply(graph, delta, delta))
    variance = divide(graph, sum_graph(graph, squared_deltas), count)
    variance_epsilon = add(graph, variance, 0.00001)
    stddev = graph.emit("mathop", "sqrt", variance_epsilon)
    normalized = []
    for index, value in enumerate(values):
        centered = graph.emit("subtract", value, mean)
        scaled = divide(graph, centered, stddev)
        gamma = graph.weight(prefix + ".weight", index)
        beta = graph.weight(prefix + ".bias", index)
        normalized.append(add(graph, multiply(graph, scaled, gamma), beta))
    return normalized


def sum_graph(graph, values):
    total = values[0]
    for value in values[1:]:
        total = add(graph, total, value)
    return total


def gelu(graph, value):
    square = multiply(graph, value, value)
    cube = multiply(graph, square, value)
    cubic_term = multiply(graph, cube, 0.044715)
    inner = multiply(graph, add(graph, value, cubic_term), math.sqrt(2 / math.pi))
    exp_input = multiply(graph, inner, 2)
    exp_value = graph.emit("mathop", "e ^", exp_input)
    tanh_value = divide(
        graph,
        graph.emit("subtract", exp_value, 1),
        graph.emit("add", exp_value, 1),
    )
    return multiply(graph, multiply(graph, value, 0.5), add(graph, tanh_value, 1))


def build_forward(graph, token_variables, vocab_size):
    dimension = EMBED_DIM
    head_size = dimension // NUM_HEADS
    states = []

    for position, token_variable in enumerate(token_variables):
        token = graph.variable(token_variable)
        token_base = multiply(graph, token, dimension)
        state = []
        for column in range(dimension):
            token_index = add(graph, token_base, column + 1)
            token_value = graph.weight("token_embedding.weight", token_index)
            position_value = graph.weight("position_embedding.weight", position * dimension + column)
            state.append(add(graph, token_value, position_value))
        states.append(state)

    prefix = "blocks.0."
    normalized = [layer_norm(graph, state, prefix + "ln1") for state in states]
    qkv = [
        matvec(
            graph,
            prefix + "attention.qkv.weight",
            prefix + "attention.qkv.bias",
            state,
            3 * dimension,
            dimension,
        )
        for state in normalized
    ]

    attended = [[None] * dimension for _ in range(CONTEXT_LENGTH)]
    for query_position in range(CONTEXT_LENGTH):
        for head in range(NUM_HEADS):
            head_start = head * head_size
            scores = []
            for key_position in range(query_position + 1):
                terms = [
                    multiply(
                        graph,
                        qkv[query_position][index],
                        qkv[key_position][dimension + index],
                    )
                    for index in range(head_start, head_start + head_size)
                ]
                scores.append(divide(graph, sum_graph(graph, terms), math.sqrt(head_size)))

            if len(scores) == 1:
                probabilities = [1]
            else:
                absolute_delta = graph.emit("mathop", "abs", graph.emit("subtract", scores[0], scores[1]))
                maximum = divide(graph, add(graph, add(graph, scores[0], scores[1]), absolute_delta), 2)
                exponentials = [
                    graph.emit("mathop", "e ^", graph.emit("subtract", score, maximum))
                    for score in scores
                ]
                denominator = sum_graph(graph, exponentials)
                probabilities = [divide(graph, value, denominator) for value in exponentials]

            for offset in range(head_size):
                value_terms = [
                    multiply(graph, probabilities[key_position], qkv[key_position][2 * dimension + head_start + offset])
                    for key_position in range(query_position + 1)
                ]
                attended[query_position][head_start + offset] = sum_graph(graph, value_terms)

    for position in range(CONTEXT_LENGTH):
        projected = matvec(
            graph,
            prefix + "attention.proj.weight",
            prefix + "attention.proj.bias",
            attended[position],
            dimension,
            dimension,
        )
        residual = [add(graph, states[position][index], projected[index]) for index in range(dimension)]
        normalized_state = layer_norm(graph, residual, prefix + "ln2")
        hidden = matvec(
            graph,
            prefix + "ffn.net.0.weight",
            prefix + "ffn.net.0.bias",
            normalized_state,
            FFN_DIM,
            dimension,
        )
        activated = [gelu(graph, value) for value in hidden]
        feed_forward = matvec(
            graph,
            prefix + "ffn.net.2.weight",
            prefix + "ffn.net.2.bias",
            activated,
            dimension,
            FFN_DIM,
        )
        states[position] = [add(graph, residual[index], feed_forward[index]) for index in range(dimension)]

    final_state = layer_norm(graph, states[-1], "ln_final")
    logits = []
    for token_id in range(vocab_size):
        products = [
            multiply(graph, final_state[column], graph.weight("token_embedding.weight", token_id * dimension + column))
            for column in range(dimension)
        ]
        logits.append(sum_graph(graph, products))
    return logits


class ProjectBuilder:
    def __init__(self):
        self.blocks = {}
        self.counter = 0
        self.variable_ids = {name: stable_id("var:" + name) for name in (
            "tokenA", "tokenB", "bestToken", "bestScore", "scanIndex",
            "promptText", "currentWord", "currentChar", "promptToken", "storyText",
            "progressText", "progressBar", "progressCell", "generationStep", "generationProgress",
            "pos", "row", "col", "acc", "mean", "variance", "delta", "stddev",
            "head", "query", "key", "feature", "channel", "tokenValue", "tempValue",
            "sumExp", "maxScore", "weightValue", "inputValue", "expValue", "candidate",
            "sampleThreshold", "cumulativeProbability", "selected",
            "scaleIndex", "cacheLen", "prefillIndex",
        )}
        self.list_ids = {name: stable_id("list:" + name) for name in (
            "weights", "vocab", "work", "logits", "tokens", "x", "norm", "qkv",
            "attention", "hidden", "scores", "probabilities", "scaled",
            "keyCache", "valueCache", "weightScales", "bitPowers",
        )}

    def new_id(self):
        self.counter += 1
        return stable_id(f"block:{self.counter}")

    def add_block(self, opcode, parent=None, inputs=None, fields=None, shadow=False, top_level=False, x=0, y=0):
        block_id = self.new_id()
        self.blocks[block_id] = {
            "opcode": opcode,
            "next": None,
            "parent": parent,
            "inputs": inputs or {},
            "fields": fields or {},
            "shadow": shadow,
            "topLevel": top_level,
        }
        if top_level:
            self.blocks[block_id]["x"] = x
            self.blocks[block_id]["y"] = y
        return block_id

    def literal(self, value):
        return [1, [4, str(value)]]

    def shadow(self, value):
        return [4, str(value)]

    def expression(self, value, parent, read_result=True):
        if isinstance(value, (int, float)):
            return self.literal(value)
        kind, payload = value if isinstance(value, tuple) and len(value) == 2 else (None, None)
        if kind == "ref" and read_result:
            return self.reporter(
                "data_itemoflist", parent, {"INDEX": self.literal(payload + 1)},
                {"LIST": ["work", self.list_ids["work"]]},
            )
        if kind != "ref":
            raise TypeError(f"Unsupported expression operand: {value!r}")
        opcode, args = self.graph.nodes[payload]
        if opcode == "variable":
            return self.reporter("data_variable", parent, {}, {"VARIABLE": [args[0], self.variable_ids[args[0]]]})
        if opcode == "weight_item":
            block_id = self.add_block(
                "data_itemoflist", parent,
                fields={"LIST": ["weights", self.list_ids["weights"]]},
            )
            self.blocks[block_id]["inputs"]["INDEX"] = self.input_for(args[0], block_id, 0)
            return [3, block_id, [10, ""]]
        if opcode == "mathop":
            block_id = self.new_id()
            self.blocks[block_id] = {
                "opcode": "operator_mathop", "next": None, "parent": parent,
                "inputs": {"NUM": self.input_for(args[1], block_id, 0)},
                "fields": {"OPERATOR": [args[0], None]}, "shadow": False, "topLevel": False,
            }
            return [3, block_id, [10, "0"]]
        if opcode in ("add", "subtract", "multiply", "divide"):
            scratch_opcode = {
                "add": "operator_add", "subtract": "operator_subtract",
                "multiply": "operator_multiply", "divide": "operator_divide",
            }[opcode]
            block_id = self.new_id()
            self.blocks[block_id] = {
                "opcode": scratch_opcode, "next": None, "parent": parent,
                "inputs": {}, "fields": {}, "shadow": False, "topLevel": False,
            }
            self.blocks[block_id]["inputs"]["NUM1"] = self.input_for(args[0], block_id, 0)
            self.blocks[block_id]["inputs"]["NUM2"] = self.input_for(args[1], block_id, 0)
            return [3, block_id, [10, "0"]]
        raise ValueError(f"Unknown graph operation: {opcode}")

    def reporter(self, opcode, parent, inputs, fields):
        block_id = self.add_block(opcode, parent, inputs, fields)
        return [3, block_id, [10, ""]]

    def input_for(self, value, parent, default):
        if isinstance(value, tuple) and value[0] == "ref":
            return [3, self.expression(value, parent)[1], self.shadow(default)]
        if isinstance(value, (int, float)):
            return self.literal(value)
        return [3, self.expression(value, parent)[1], self.literal(default)]

    def add_action(self, chain, opcode, inputs=None, fields=None):
        parent = chain[-1] if chain else None
        block_id = self.add_block(opcode, parent, inputs or {}, fields or {})
        if chain:
            self.blocks[chain[-1]]["next"] = block_id
        chain.append(block_id)
        return block_id

    def set_variable(self, chain, name, value):
        block_id = self.add_action(chain, "data_setvariableto", fields={"VARIABLE": [name, self.variable_ids[name]]})
        self.blocks[block_id]["inputs"]["VALUE"] = self.input_for(value, block_id, 0)
        return block_id

    def variable_reporter(self, name, parent):
        return self.reporter(
            "data_variable", parent, {},
            {"VARIABLE": [name, self.variable_ids[name]]},
        )

    def connect_reporter(self, block_id, input_name, reporter, fallback="", numeric=False):
        shadow = self.shadow(fallback) if numeric else [10, str(fallback)]
        self.blocks[block_id]["inputs"][input_name] = [3, reporter[1], shadow]
        self.blocks[reporter[1]]["parent"] = block_id

    def set_reporter(self, chain, name, reporter, fallback="", numeric=False):
        block_id = self.add_action(
            chain, "data_setvariableto",
            fields={"VARIABLE": [name, self.variable_ids[name]]},
        )
        self.connect_reporter(block_id, "VALUE", reporter, fallback, numeric)
        return block_id

    def emit_node(self, opcode, *args):
        return self.graph.emit(opcode, *args)

    def variable_node(self, name):
        return self.graph.variable(name)

    def compile_graph(self, chain, parent=None):
        for index, (opcode, args) in enumerate(self.graph.nodes):
            command = self.add_block(
                "data_replaceitemoflist",
                chain[-1] if chain else parent,
                fields={"LIST": ["work", self.list_ids["work"]]},
            )
            if chain:
                self.blocks[chain[-1]]["next"] = command
            chain.append(command)
            self.blocks[command]["inputs"]["INDEX"] = self.literal(index + 1)
            self.blocks[command]["inputs"]["ITEM"] = self.instruction_expression(index, command)

    def instruction_expression(self, index, parent):
        return self.expression(("ref", index), parent, read_result=False)


class ScratchCode:
    """Build native Scratch commands and reporter trees for loop-based inference."""

    STRING_VARIABLES = {"promptText", "currentWord", "currentChar", "storyText", "progressText", "progressBar"}
    STRING_LISTS = {"vocab"}
    BOOLEAN_OPS = {"operator_equals", "operator_gt", "operator_lt", "operator_and", "operator_or", "operator_not"}
    BINARY_INPUTS = {
        "operator_add": ("NUM1", "NUM2"),
        "operator_subtract": ("NUM1", "NUM2"),
        "operator_multiply": ("NUM1", "NUM2"),
        "operator_divide": ("NUM1", "NUM2"),
        "operator_mod": ("NUM1", "NUM2"),
        "operator_random": ("FROM", "TO"),
        "operator_equals": ("OPERAND1", "OPERAND2"),
        "operator_gt": ("OPERAND1", "OPERAND2"),
        "operator_lt": ("OPERAND1", "OPERAND2"),
        "operator_and": ("OPERAND1", "OPERAND2"),
        "operator_or": ("OPERAND1", "OPERAND2"),
        "operator_join": ("STRING1", "STRING2"),
    }

    def __init__(self, builder):
        self.builder = builder

    @staticmethod
    def var(name):
        return ("var", name)

    @staticmethod
    def item(name, index):
        return ("item", name, index)

    @staticmethod
    def length(name):
        return ("length", name)

    @staticmethod
    def string_length(name):
        return ("string_length", name)

    @staticmethod
    def item_number(name, value):
        return ("item_number", name, value)

    @staticmethod
    def letter(index, text):
        return ("letter", index, text)

    @staticmethod
    def op(name, left, right):
        return ("op", name, left, right)

    @staticmethod
    def unary(name, value):
        return ("unary", name, value)

    @staticmethod
    def math(name, value):
        return ("math", name, value)

    @staticmethod
    def text(value):
        return ("text", value)

    def connect(self, block_id, input_name, value, fallback=0):
        if isinstance(value, bool):
            self.builder.blocks[block_id]["inputs"][input_name] = [1, [10, "true" if value else "false"]]
        elif isinstance(value, (int, float)):
            self.builder.blocks[block_id]["inputs"][input_name] = self.builder.literal(value)
        elif isinstance(value, tuple) and value[0] == "text":
            self.builder.blocks[block_id]["inputs"][input_name] = [1, [10, str(value[1])]]
        else:
            reporter_id, kind = self.expression(value, block_id)
            if kind == "boolean":
                self.builder.blocks[block_id]["inputs"][input_name] = [2, reporter_id]
            else:
                shadow = [10, ""] if kind == "string" else self.builder.shadow(fallback)
                self.builder.blocks[block_id]["inputs"][input_name] = [3, reporter_id, shadow]

    def expression(self, expression, parent):
        tag = expression[0]
        if tag == "var":
            name = expression[1]
            block_id = self.builder.add_block(
                "data_variable", parent,
                fields={"VARIABLE": [name, self.builder.variable_ids[name]]},
            )
            kind = "string" if name in self.STRING_VARIABLES else "number"
        elif tag == "item":
            name, index = expression[1:]
            block_id = self.builder.add_block(
                "data_itemoflist", parent,
                fields={"LIST": [name, self.builder.list_ids[name]]},
            )
            self.connect(block_id, "INDEX", index)
            kind = "string" if name in self.STRING_LISTS else "number"
        elif tag == "length":
            name = expression[1]
            block_id = self.builder.add_block(
                "data_lengthoflist", parent,
                fields={"LIST": [name, self.builder.list_ids[name]]},
            )
            kind = "number"
        elif tag == "string_length":
            name = expression[1]
            block_id = self.builder.add_block("operator_length", parent)
            self.connect(block_id, "STRING", self.var(name))
            kind = "number"
        elif tag == "item_number":
            name, value = expression[1:]
            block_id = self.builder.add_block(
                "data_itemnumoflist", parent,
                fields={"LIST": [name, self.builder.list_ids[name]]},
            )
            self.connect(block_id, "ITEM", value)
            kind = "number"
        elif tag == "letter":
            index, text = expression[1:]
            block_id = self.builder.add_block("operator_letter_of", parent)
            self.connect(block_id, "LETTER", index)
            self.connect(block_id, "STRING", text)
            kind = "string"
        elif tag == "op":
            _, opcode, left, right = expression
            block_id = self.builder.add_block(opcode, parent)
            input_names = self.BINARY_INPUTS[opcode]
            self.connect(block_id, input_names[0], left)
            self.connect(block_id, input_names[1], right)
            kind = "boolean" if opcode in self.BOOLEAN_OPS else ("string" if opcode == "operator_join" else "number")
        elif tag == "unary":
            _, opcode, value = expression
            block_id = self.builder.add_block(opcode, parent)
            input_name = "OPERAND" if opcode == "operator_not" else "NUM"
            self.connect(block_id, input_name, value)
            kind = "boolean" if opcode == "operator_not" else "number"
        elif tag == "math":
            _, operator, value = expression
            block_id = self.builder.add_block(
                "operator_mathop", parent,
                fields={"OPERATOR": [operator, None]},
            )
            self.connect(block_id, "NUM", value)
            kind = "number"
        else:
            raise ValueError(f"Unknown Scratch expression: {expression!r}")
        return block_id, kind

    def command(self, chain, opcode, fields=None, parent=None):
        block_parent = chain[-1] if chain else parent
        block_id = self.builder.add_block(opcode, block_parent, fields=fields)
        if chain:
            self.builder.blocks[chain[-1]]["next"] = block_id
        chain.append(block_id)
        return block_id

    def set_var(self, chain, name, value):
        block_id = self.command(chain, "data_setvariableto", fields={
            "VARIABLE": [name, self.builder.variable_ids[name]],
        })
        self.connect(block_id, "VALUE", value)
        return block_id

    def replace_item(self, chain, list_name, index, value):
        block_id = self.command(chain, "data_replaceitemoflist", fields={
            "LIST": [list_name, self.builder.list_ids[list_name]],
        })
        self.connect(block_id, "INDEX", index)
        self.connect(block_id, "ITEM", value)
        return block_id

    def add_item(self, chain, list_name, value):
        block_id = self.command(chain, "data_addtolist", fields={
            "LIST": [list_name, self.builder.list_ids[list_name]],
        })
        self.connect(block_id, "ITEM", value)
        return block_id

    def delete_item(self, chain, list_name, index):
        block_id = self.command(chain, "data_deleteoflist", fields={
            "LIST": [list_name, self.builder.list_ids[list_name]],
        })
        self.connect(block_id, "INDEX", index)
        return block_id

    def repeat(self, chain, times, build_body):
        loop_id = self.command(chain, "control_repeat")
        self.connect(loop_id, "TIMES", times)
        body = []
        build_body(body, loop_id)
        if body:
            self.builder.blocks[loop_id]["inputs"]["SUBSTACK"] = [2, body[0]]
            self.builder.blocks[body[0]]["parent"] = loop_id
        return loop_id

    def if_block(self, chain, condition, build_body, opcode="control_if"):
        block_id = self.command(chain, opcode)
        self.connect(block_id, "CONDITION", condition)
        body = []
        build_body(body, block_id)
        if body:
            self.builder.blocks[block_id]["inputs"]["SUBSTACK"] = [2, body[0]]
            self.builder.blocks[body[0]]["parent"] = block_id
        return block_id


def build_state(torch, state):
    old_dimension = state["token_embedding.weight"].shape[1]
    from config import NUM_HEADS as SOURCE_NUM_HEADS

    if old_dimension % SOURCE_NUM_HEADS:
        raise ValueError("Source embedding size must be divisible by the configured source head count")
    if EMBED_DIM % NUM_HEADS or NUM_HEADS > SOURCE_NUM_HEADS:
        raise ValueError("Target embedding/head configuration cannot be mapped from this checkpoint")
    source_head_size = old_dimension // SOURCE_NUM_HEADS
    target_head_size = EMBED_DIM // NUM_HEADS
    if target_head_size > source_head_size:
        raise ValueError("Target head width cannot exceed the checkpoint head width")
    channel_ids = [
        head * source_head_size + offset
        for head in range(NUM_HEADS)
        for offset in range(target_head_size)
    ]
    qkv_rows = channel_ids + [old_dimension + item for item in channel_ids] + [2 * old_dimension + item for item in channel_ids]
    tensors = {
        "token_embedding.weight": state["token_embedding.weight"][:VOCAB_SIZE, :EMBED_DIM].clone(),
        "position_embedding.weight": state["position_embedding.weight"][:CONTEXT_LENGTH, :EMBED_DIM].clone(),
        "ln_final.weight": state["ln_final.weight"][:EMBED_DIM].clone(),
        "ln_final.bias": state["ln_final.bias"][:EMBED_DIM].clone(),
    }
    channel_index = torch.tensor(channel_ids)
    qkv_index = torch.tensor(qkv_rows)
    for layer in range(NUM_LAYERS):
        prefix = f"blocks.{layer}."
        for name in ("ln1.weight", "ln1.bias", "ln2.weight", "ln2.bias"):
            tensors[prefix + name] = state[prefix + name][:EMBED_DIM].clone()
        tensors[prefix + "attention.qkv.weight"] = state[prefix + "attention.qkv.weight"].index_select(
            0, qkv_index
        ).index_select(1, channel_index).clone()
        tensors[prefix + "attention.qkv.bias"] = state[prefix + "attention.qkv.bias"].index_select(
            0, qkv_index
        ).clone()
        tensors[prefix + "attention.proj.weight"] = state[prefix + "attention.proj.weight"].index_select(
            0, channel_index
        ).index_select(1, channel_index).clone()
        tensors[prefix + "attention.proj.bias"] = state[prefix + "attention.proj.bias"].index_select(
            0, channel_index
        ).clone()
        tensors[prefix + "ffn.net.0.weight"] = state[prefix + "ffn.net.0.weight"][:FFN_DIM, :].index_select(
            1, channel_index
        ).clone()
        tensors[prefix + "ffn.net.0.bias"] = state[prefix + "ffn.net.0.bias"][:FFN_DIM].clone()
        tensors[prefix + "ffn.net.2.weight"] = state[prefix + "ffn.net.2.weight"].index_select(
            0, channel_index
        ).clone()
        tensors[prefix + "ffn.net.2.bias"] = state[prefix + "ffn.net.2.bias"].index_select(
            0, channel_index
        ).clone()
    return tensors


def make_native_project(tensors, vocab):
    builder = ProjectBuilder()

    def numeric_input(block_id, name, value):
        builder.blocks[block_id]["inputs"][name] = builder.literal(value)

    def text_input(block_id, name, value):
        builder.blocks[block_id]["inputs"][name] = [1, [10, str(value)]]

    def variable_input(block_id, input_name, variable_name, numeric=False):
        reporter = builder.variable_reporter(variable_name, block_id)
        builder.connect_reporter(block_id, input_name, reporter, 0 if numeric else "", numeric)

    def length_reporter(variable_name, parent):
        length = builder.add_block("operator_length", parent)
        variable_input(length, "STRING", variable_name)
        return [3, length, [10, ""]]

    def comparison(opcode, left_reporter, right_value, numeric=True):
        condition = builder.add_block(opcode)
        builder.connect_reporter(condition, "OPERAND1", left_reporter, 0 if numeric else "", numeric)
        if numeric:
            numeric_input(condition, "OPERAND2", right_value)
        else:
            text_input(condition, "OPERAND2", right_value)
        return condition

    def start_substack(owner, input_name, opcode="data_setvariableto", fields=None):
        root = builder.add_block(opcode, owner, fields=fields or {})
        builder.blocks[owner]["inputs"][input_name] = [2, root]
        return [root]

    def append_word_token(chain, owner=None, input_name="SUBSTACK"):
        nonempty = comparison("operator_gt", length_reporter("currentWord", None), 0)
        if owner is None:
            process_if = builder.add_action(chain, "control_if")
        else:
            process_if = builder.add_block("control_if", owner)
            builder.blocks[owner]["inputs"][input_name] = [2, process_if]
            chain = [process_if]
        builder.blocks[process_if]["inputs"]["CONDITION"] = [2, nonempty]
        builder.blocks[nonempty]["parent"] = process_if
        word_chain = start_substack(
            process_if, "SUBSTACK", fields={"VARIABLE": ["promptToken", builder.variable_ids["promptToken"]]}
        )

        subtract = builder.add_block("operator_subtract", word_chain[-1])
        item_number = builder.add_block(
            "data_itemnumoflist", subtract,
            fields={"LIST": ["vocab", builder.list_ids["vocab"]]},
        )
        variable_input(item_number, "ITEM", "currentWord")
        builder.connect_reporter(subtract, "NUM1", [3, item_number, [10, ""]], 0, True)
        numeric_input(subtract, "NUM2", 1)
        builder.blocks[word_chain[-1]]["inputs"]["VALUE"] = [3, subtract, builder.shadow(0)]

        unknown = comparison(
            "operator_lt", builder.variable_reporter("promptToken", None), 1
        )
        unknown_if = builder.add_action(word_chain, "control_if")
        builder.blocks[unknown_if]["inputs"]["CONDITION"] = [2, unknown]
        builder.blocks[unknown]["parent"] = unknown_if
        unknown_chain = start_substack(
            unknown_if, "SUBSTACK", fields={"VARIABLE": ["promptToken", builder.variable_ids["promptToken"]]}
        )
        numeric_input(unknown_chain[-1], "VALUE", 1)

        builder.set_reporter(word_chain, "tokenA", builder.variable_reporter("tokenB", None), numeric=True)
        builder.set_reporter(word_chain, "tokenB", builder.variable_reporter("promptToken", None), numeric=True)
        return chain

    def set_text_literal(chain, name, value):
        block = builder.add_action(chain, "data_setvariableto", fields={"VARIABLE": [name, builder.variable_ids[name]]})
        text_input(block, "VALUE", value)
        return block

    offsets = {}
    weights = []
    tensor_info = {}
    for name, tensor in tensors.items():
        flat = tensor.detach().cpu().float().reshape(-1)
        maximum = float(flat.abs().max().item()) if flat.numel() else 0.0
        scale = maximum / 32767 if maximum else 1.0
        quantized = (flat / scale).round().clamp(-32767, 32767).tolist()
        offsets[name] = len(weights)
        weights.extend(quantized)
        tensor_info[name] = {"offset": offsets[name], "scale": scale}

    builder.graph = Graph(tensor_info)
    logits = build_forward(builder.graph, ["tokenA", "tokenB"], OUTPUT_VOCAB_SIZE)
    flag = builder.add_block("event_whenflagclicked", top_level=True, x=30, y=30)
    chain = [flag]
    ask = builder.add_action(
        chain, "sensing_askandwait",
        {"QUESTION": [1, [10, "Type a lowercase story prompt; separate words with spaces:"]]},
    )
    answer = builder.reporter("sensing_answer", ask, {}, {})
    builder.set_reporter(chain, "promptText", answer)
    for name, value in (("tokenA", 0), ("tokenB", 1), ("scanIndex", 1)):
        builder.set_variable(chain, name, value)
    set_text_literal(chain, "currentWord", "")
    builder.set_reporter(chain, "storyText", builder.variable_reporter("promptText", None))

    prompt_length = length_reporter("promptText", None)
    scan_repeat = builder.add_action(chain, "control_repeat")
    builder.connect_reporter(scan_repeat, "TIMES", prompt_length, 0, numeric=True)
    scan_body = []
    current_char_set = builder.add_block(
        "data_setvariableto", scan_repeat,
        fields={"VARIABLE": ["currentChar", builder.variable_ids["currentChar"]]},
    )
    scan_body.append(current_char_set)
    builder.blocks[scan_repeat]["inputs"]["SUBSTACK"] = [2, current_char_set]
    letter = builder.add_block("operator_letter_of", current_char_set)
    variable_input(letter, "LETTER", "scanIndex", numeric=True)
    variable_input(letter, "STRING", "promptText")
    builder.connect_reporter(current_char_set, "VALUE", [3, letter, [10, ""]])

    equals_space = comparison("operator_equals", builder.variable_reporter("currentChar", None), " ", numeric=False)
    separator_if = builder.add_action(scan_body, "control_if_else")
    builder.blocks[separator_if]["inputs"]["CONDITION"] = [2, equals_space]
    builder.blocks[equals_space]["parent"] = separator_if
    separator_body = append_word_token([], separator_if)
    set_text_literal(separator_body, "currentWord", "")

    next_word_body = start_substack(separator_if, "SUBSTACK2", fields={"VARIABLE": ["currentWord", builder.variable_ids["currentWord"]]})
    join_word = builder.add_block("operator_join", next_word_body[-1])
    variable_input(join_word, "STRING1", "currentWord")
    variable_input(join_word, "STRING2", "currentChar")
    builder.blocks[next_word_body[-1]]["inputs"]["VALUE"] = [3, join_word, [10, ""]]

    increment = builder.add_block(
        "operator_add", None,
    )
    scan_index_set = builder.add_action(scan_body, "data_setvariableto", fields={"VARIABLE": ["scanIndex", builder.variable_ids["scanIndex"]]})
    variable_input(increment, "NUM1", "scanIndex", numeric=True)
    numeric_input(increment, "NUM2", 1)
    builder.blocks[increment]["parent"] = scan_index_set
    builder.blocks[scan_index_set]["inputs"]["VALUE"] = [3, increment, builder.shadow(0)]

    append_word_token(chain)

    generate_repeat = builder.add_action(chain, "control_repeat", {"TIMES": builder.literal(GENERATION_TOKENS)})
    generation_body = []
    builder.compile_graph(generation_body, parent=generate_repeat)
    for index, logit in enumerate(logits):
        command = builder.add_action(generation_body, "data_replaceitemoflist", fields={"LIST": ["logits", builder.list_ids["logits"]]})
        numeric_input(command, "INDEX", index + 1)
        builder.blocks[command]["inputs"]["ITEM"] = builder.input_for(logit, command, 0)

    builder.set_variable(generation_body, "bestToken", 2)
    builder.set_variable(generation_body, "bestScore", -1_000_000_000)
    for token_id in range(2, OUTPUT_VOCAB_SIZE):
        score_condition = builder.add_block("operator_gt")
        score = builder.reporter(
            "data_itemoflist", score_condition, {"INDEX": builder.literal(token_id + 1)},
            {"LIST": ["logits", builder.list_ids["logits"]]},
        )
        current_best = builder.variable_reporter("bestScore", score_condition)
        builder.blocks[score_condition]["inputs"] = {"OPERAND1": score, "OPERAND2": current_best}

        not_recent = []
        for context_name in ("tokenA", "tokenB"):
            equals_context = builder.add_block("operator_equals")
            variable_input(equals_context, "OPERAND1", context_name, numeric=True)
            numeric_input(equals_context, "OPERAND2", token_id)
            not_equal = builder.add_block("operator_not")
            builder.blocks[not_equal]["inputs"]["OPERAND"] = [2, equals_context]
            builder.blocks[equals_context]["parent"] = not_equal
            not_recent.append(not_equal)
        distinct_context = builder.add_block("operator_and")
        builder.blocks[distinct_context]["inputs"] = {
            "OPERAND1": [2, not_recent[0]],
            "OPERAND2": [2, not_recent[1]],
        }
        for reporter_id in not_recent:
            builder.blocks[reporter_id]["parent"] = distinct_context
        condition = builder.add_block("operator_and")
        builder.blocks[condition]["inputs"] = {
            "OPERAND1": [2, score_condition],
            "OPERAND2": [2, distinct_context],
        }
        builder.blocks[score_condition]["parent"] = condition
        builder.blocks[distinct_context]["parent"] = condition

        substack = builder.add_block("data_setvariableto", None, fields={"VARIABLE": ["bestScore", builder.variable_ids["bestScore"]]})
        builder.blocks[substack]["inputs"]["VALUE"] = builder.reporter("data_itemoflist", substack, {"INDEX": builder.literal(token_id + 1)}, {"LIST": ["logits", builder.list_ids["logits"]]})
        set_token = builder.add_block("data_setvariableto", substack, fields={"VARIABLE": ["bestToken", builder.variable_ids["bestToken"]]})
        numeric_input(set_token, "VALUE", token_id)
        builder.blocks[substack]["next"] = set_token
        if_id = builder.add_action(generation_body, "control_if", {"CONDITION": [2, condition], "SUBSTACK": [2, substack]})
        builder.blocks[condition]["parent"] = if_id
        builder.blocks[substack]["parent"] = if_id

    story_set = builder.add_action(generation_body, "data_setvariableto", fields={"VARIABLE": ["storyText", builder.variable_ids["storyText"]]})
    story_join = builder.add_block("operator_join", story_set)
    builder.blocks[story_set]["inputs"]["VALUE"] = [3, story_join, [10, ""]]
    story_var = builder.variable_reporter("storyText", story_join)
    builder.connect_reporter(story_join, "STRING1", story_var)
    next_join = builder.add_block("operator_join", story_join)
    builder.blocks[story_join]["inputs"]["STRING2"] = [3, next_join, [10, ""]]
    builder.blocks[next_join]["inputs"]["STRING1"] = [1, [10, " "]]
    word_reporter = builder.add_block(
        "data_itemoflist", next_join,
        fields={"LIST": ["vocab", builder.list_ids["vocab"]]},
    )
    token_index = builder.add_block("operator_add", word_reporter)
    best_token = builder.variable_reporter("bestToken", token_index)
    builder.connect_reporter(token_index, "NUM1", best_token, 0, numeric=True)
    numeric_input(token_index, "NUM2", 1)
    builder.blocks[word_reporter]["inputs"]["INDEX"] = [3, token_index, builder.shadow(1)]
    builder.blocks[next_join]["inputs"]["STRING2"] = [3, word_reporter, [10, ""]]

    builder.set_reporter(generation_body, "tokenA", builder.variable_reporter("tokenB", None), numeric=True)
    builder.set_reporter(generation_body, "tokenB", builder.variable_reporter("bestToken", None), numeric=True)
    builder.blocks[generate_repeat]["inputs"]["SUBSTACK"] = [2, generation_body[0]]

    say_block = builder.add_action(chain, "looks_say")
    story_reporter = builder.variable_reporter("storyText", say_block)
    builder.connect_reporter(say_block, "MESSAGE", story_reporter)

    stage_svg = b'<svg xmlns="http://www.w3.org/2000/svg" width="480" height="360"><rect width="480" height="360" fill="#fff"/></svg>'
    sprite_svg = b'<svg xmlns="http://www.w3.org/2000/svg" width="1" height="1"><rect width="1" height="1" fill="none"/></svg>'
    stage = {
        "isStage": True, "name": "Stage", "variables": {}, "lists": {}, "broadcasts": {},
        "blocks": {}, "comments": {}, "currentCostume": 0,
        "costumes": [costume(stage_svg, "backdrop1", 240, 180)], "sounds": [], "volume": 100,
        "layerOrder": 0, "tempo": 60, "videoTransparency": 50, "videoState": "on",
        "textToSpeechLanguage": None,
    }
    sprite_lists = {
        "weights": ["weights", weights],
        "vocab": ["vocab", vocab],
        "work": ["work", [0] * len(builder.graph.nodes)],
        "logits": ["logits", [0] * OUTPUT_VOCAB_SIZE],
    }
    sprite = {
        "isStage": False, "name": "TinyGPT Scratch", "variables": {
            builder.variable_ids["tokenA"]: ["tokenA", 1],
            builder.variable_ids["tokenB"]: ["tokenB", 1],
            builder.variable_ids["bestToken"]: ["bestToken", 0],
            builder.variable_ids["bestScore"]: ["bestScore", 0],
            builder.variable_ids["scanIndex"]: ["scanIndex", 0],
        }, "lists": {
            builder.list_ids[name]: values for name, values in sprite_lists.items()
        }, "broadcasts": {}, "blocks": builder.blocks, "comments": {}, "currentCostume": 0,
        "costumes": [costume(sprite_svg, "blank", 0.5, 0.5)], "sounds": [], "volume": 100,
        "layerOrder": 1, "visible": True, "x": 0, "y": 0, "size": 100,
        "direction": 90, "draggable": False, "rotationStyle": "all around",
    }
    project = {
        "targets": [stage, sprite], "monitors": [], "extensions": [],
        "meta": {"semver": "3.0.0", "vm": "0.2.0", "agent": "native-tinygpt-exporter"},
    }
    return project, stage_svg, sprite_svg, len(builder.graph.nodes)


def make_full_native_project(tensors, vocab):
    torch = __import__("torch")
    builder = ProjectBuilder()
    code = ScratchCode(builder)
    V = code.var
    I = code.item
    O = code.op
    M = code.math
    L = code.length
    tensor_info = {}
    weights = []
    weight_scales = []
    row_scales = {}
    for name, tensor in tensors.items():
        if tensor.ndim != 2:
            continue
        rows = tensor.detach().cpu().float().reshape(tensor.shape[0], -1)
        peaks = rows.abs().amax(dim=1)
        bias_name = name[:-7] + ".bias" if name.endswith(".weight") else ""
        if bias_name in tensors and tensors[bias_name].numel() == len(peaks):
            peaks = torch.maximum(peaks, tensors[bias_name].detach().cpu().float().abs())
        row_scales[name] = torch.where(peaks > 0, peaks / 3, torch.ones_like(peaks))

    logical_offset = 0
    pending_bits = 0
    pending_count = 0
    for name, tensor in tensors.items():
        flat = tensor.detach().cpu().float().reshape(-1)
        bias_weight = name[:-5] + ".weight" if name.endswith(".bias") else ""
        if name in row_scales:
            scales = row_scales[name]
            rows = flat.reshape(tensor.shape[0], -1)
            quantized = (rows / scales[:, None]).round().clamp(-3, 3).to(dtype=torch.int8).reshape(-1).tolist()
            scale_offset = len(weight_scales)
            weight_scales.extend(scales.tolist())
            tensor_info[name] = {
                "offset": logical_offset,
                "scale_offset": scale_offset,
                "row_width": rows.shape[1],
                "size": flat.numel(),
            }
        elif bias_weight in row_scales:
            scales = row_scales[bias_weight]
            quantized = (flat / scales).round().clamp(-3, 3).to(dtype=torch.int8).tolist()
            tensor_info[name] = {
                "offset": logical_offset,
                "scale_offset": tensor_info[bias_weight]["scale_offset"],
                "size": flat.numel(),
            }
        else:
            maximum = float(flat.abs().max().item()) if flat.numel() else 0.0
            scale = maximum / 3 if maximum else 1.0
            quantized = (flat / scale).round().clamp(-3, 3).to(dtype=torch.int8).tolist()
            scale_offset = len(weight_scales)
            weight_scales.append(scale)
            tensor_info[name] = {"offset": logical_offset, "scale_offset": scale_offset, "size": flat.numel()}
        logical_offset += flat.numel()
        for value in quantized:
            pending_bits |= (value + 3) << (3 * pending_count)
            pending_count += 1
            if pending_count == 8:
                weights.append(pending_bits)
                pending_bits = 0
                pending_count = 0
    if pending_count:
        weights.append(pending_bits)

    def read_weight(name, index):
        info = tensor_info[name]
        absolute_index = O("operator_add", info["offset"], index) if not isinstance(index, int) else info["offset"] + index
        packed_index = O("operator_add", M("floor", O("operator_divide", absolute_index, 8)), 1)
        packed_value = I("weights", packed_index)
        power_index = O("operator_add", O("operator_mod", absolute_index, 8), 1)
        power = I("bitPowers", power_index)
        packed_value = M("floor", O("operator_divide", packed_value, power))
        return O("operator_subtract", O("operator_mod", packed_value, 8), 3)

    def weight_scale(name, row_index):
        info = tensor_info[name]
        if "scale" in info:
            return info["scale"]
        if isinstance(row_index, int):
            return I("weightScales", info["scale_offset"] + row_index + 1)
        return I("weightScales", O("operator_add", info["scale_offset"] + 1, row_index))

    def tensor_scale(name, flat_index):
        info = tensor_info[name]
        row = M("floor", O("operator_divide", flat_index, info.get("row_width", 1)))
        return weight_scale(name, row)

    def command(chain, opcode, fields=None, parent=None):
        return code.command(chain, opcode, fields=fields, parent=parent)

    def set_value(chain, variable_name, value):
        return code.set_var(chain, variable_name, value)

    def change(chain, variable_name, amount=1):
        block = command(chain, "data_changevariableby", fields={
            "VARIABLE": [variable_name, builder.variable_ids[variable_name]],
        })
        code.connect(block, "VALUE", amount)
        return block

    def if_block(chain, condition, build_body, opcode="control_if"):
        return code.if_block(chain, condition, build_body, opcode)

    def repeat(chain, times, build_body):
        return code.repeat(chain, times, build_body)

    def update_progress_bar(chain):
        set_value(chain, "progressBar", code.text("\u2591" * 20))
        set_value(chain, "progressCell", 1)
        filled_cells = M("floor", O("operator_divide", V("generationProgress"), 5))

        def append_cell(cell_chain, _):
            condition = code.unary(
                "operator_not", O("operator_gt", V("progressCell"), filled_cells),
            )

            def append_filled(filled_chain, _):
                set_value(filled_chain, "progressBar", O(
                    "operator_join", V("progressBar"), code.text("\u2588"),
                ))

            branch = if_block(cell_chain, condition, append_filled, opcode="control_if_else")
            empty_chain = []
            set_value(empty_chain, "progressBar", O(
                "operator_join", V("progressBar"), code.text("\u2591"),
            ))
            builder.blocks[branch]["inputs"]["SUBSTACK2"] = [2, empty_chain[0]]
            builder.blocks[empty_chain[0]]["parent"] = branch

        repeat(chain, 20, append_cell)

    def replace(chain, list_name, index, value):
        return code.replace_item(chain, list_name, index, value)

    def append_token(chain, parent=None, substack_name="SUBSTACK"):
        condition = O("operator_gt", code.string_length("currentWord"), 0)

        def finish_word(word_chain, owner):
            set_value(word_chain, "promptToken", O(
                "operator_subtract", code.item_number("vocab", V("currentWord")), 1,
            ))

            def use_unknown(unknown_chain, _):
                set_value(unknown_chain, "promptToken", 1)

            if_block(
                word_chain,
                O("operator_lt", V("promptToken"), 1),
                use_unknown,
            )
            code.add_item(word_chain, "tokens", V("promptToken"))
            set_value(word_chain, "tokenA", V("tokenB"))
            set_value(word_chain, "tokenB", V("promptToken"))

            def trim_context(trim_chain, _):
                code.delete_item(trim_chain, "tokens", 1)

            if_block(
                word_chain,
                O("operator_gt", L("tokens"), CONTEXT_LENGTH - GENERATION_TOKENS),
                trim_context,
            )

        if parent is None:
            if_block(chain, condition, finish_word)
            return chain

        branch = []
        if_block(branch, condition, finish_word, opcode="control_if")
        builder.blocks[parent]["inputs"][substack_name] = [2, branch[0]]
        builder.blocks[branch[0]]["parent"] = parent
        return branch

    start = builder.add_block("event_whenflagclicked", top_level=True, x=30, y=30)
    chain = [start]
    ask = command(chain, "sensing_askandwait")
    builder.blocks[ask]["inputs"]["QUESTION"] = [1, [10, "Type a lowercase story prompt:"]]
    answer = builder.add_block("sensing_answer", ask)
    prompt_set = command(chain, "data_setvariableto", fields={
        "VARIABLE": ["promptText", builder.variable_ids["promptText"]],
    })
    builder.blocks[prompt_set]["inputs"]["VALUE"] = [3, answer, [10, ""]]
    builder.blocks[answer]["parent"] = prompt_set
    set_value(chain, "storyText", V("promptText"))
    for name, value in (("tokenA", 0), ("tokenB", 1), ("scanIndex", 1)):
        set_value(chain, name, value)
    set_value(chain, "currentWord", code.text(""))

    def scan_prompt(scan_chain, owner):
        letter_set = command(scan_chain, "data_setvariableto", fields={
            "VARIABLE": ["currentChar", builder.variable_ids["currentChar"]],
        })
        code.connect(letter_set, "VALUE", ("letter", V("scanIndex"), V("promptText")))

        def split_word(space_chain, _):
            append_token(space_chain)
            set_value(space_chain, "currentWord", code.text(""))

        def add_character(char_chain, _):
            set_value(char_chain, "currentWord", O(
                "operator_join", V("currentWord"), V("currentChar"),
            ))

        if_block(
            scan_chain,
            O("operator_equals", V("currentChar"), code.text(" ")),
            split_word,
            opcode="control_if_else",
        )
        if_else = scan_chain[-1]
        else_chain = []
        add_character(else_chain, None)
        builder.blocks[if_else]["inputs"]["SUBSTACK"] = [2, builder.blocks[if_else]["inputs"]["SUBSTACK"][1]]
        builder.blocks[if_else]["inputs"]["SUBSTACK2"] = [2, else_chain[0]]
        builder.blocks[else_chain[0]]["parent"] = if_else
        change(scan_chain, "scanIndex")

    repeat(chain, code.string_length("promptText"), scan_prompt)
    append_token(chain)

    def supply_unknown_seed(seed_chain, _):
        code.add_item(seed_chain, "tokens", 1)
        set_value(seed_chain, "tokenB", 1)

    if_block(chain, O("operator_equals", L("tokens"), 0), supply_unknown_seed)

    def tensor_value(name, index):
        return O(
            "operator_multiply",
            read_weight(name, index),
            tensor_scale(name, index),
        )

    def linear(chain, weight_name, bias_name, input_list, output_list, input_width, output_width, row_count):
        weight = tensor_info[weight_name]
        bias = tensor_info[bias_name] if bias_name else None
        set_value(chain, "row", 0)

        def each_row(row_chain, _):
            row_in_matrix = O("operator_mod", V("row"), output_width)
            row_scale = weight_scale(weight_name, row_in_matrix)
            if bias:
                set_value(row_chain, "acc", O(
                    "operator_multiply", read_weight(bias_name, row_in_matrix), row_scale,
                ))
            else:
                set_value(row_chain, "acc", 0)
            set_value(row_chain, "col", 0)

            def each_column(column_chain, _):
                weight_index = O("operator_add", O(
                    "operator_multiply", row_in_matrix, input_width,
                ), V("col"))
                position = M("floor", O("operator_divide", V("row"), output_width))
                activation_index = O("operator_add", O("operator_multiply", position, input_width), O("operator_add", V("col"), 1))
                set_value(column_chain, "weightValue", read_weight(weight_name, weight_index))
                set_value(column_chain, "inputValue", I(input_list, activation_index))
                change(column_chain, "acc", O("operator_multiply", V("weightValue"), V("inputValue")))
                change(column_chain, "col")

            repeat(row_chain, input_width, each_column)
            output_index = O("operator_add", V("row"), 1)
            replace(row_chain, output_list, output_index, O("operator_multiply", V("acc"), row_scale))
            change(row_chain, "row")

        repeat(chain, row_count, each_row)

    def layer_norm(chain, input_list, output_list, tensor_prefix, position_count=None):
        gamma = tensor_info[tensor_prefix + ".weight"]
        beta = tensor_info[tensor_prefix + ".bias"]
        set_value(chain, "pos", 0)
        if position_count is None:
            position_count = L("tokens")

        def each_position(position_chain, _):
            set_value(position_chain, "mean", 0)
            set_value(position_chain, "col", 0)

            def sum_values(sum_chain, _):
                index = O("operator_add", O("operator_multiply", V("pos"), EMBED_DIM), O("operator_add", V("col"), 1))
                change(sum_chain, "mean", I(input_list, index))
                change(sum_chain, "col")

            repeat(position_chain, EMBED_DIM, sum_values)
            set_value(position_chain, "mean", O("operator_divide", V("mean"), EMBED_DIM))
            set_value(position_chain, "variance", 0)
            set_value(position_chain, "col", 0)

            def sum_variance(variance_chain, _):
                index = O("operator_add", O("operator_multiply", V("pos"), EMBED_DIM), O("operator_add", V("col"), 1))
                difference = O("operator_subtract", I(input_list, index), V("mean"))
                change(variance_chain, "variance", O("operator_multiply", difference, difference))
                change(variance_chain, "col")

            repeat(position_chain, EMBED_DIM, sum_variance)
            variance = O("operator_divide", V("variance"), EMBED_DIM)
            set_value(position_chain, "stddev", M("sqrt", O("operator_add", variance, 0.00001)))
            set_value(position_chain, "col", 0)

            def normalize(normalize_chain, _):
                index = O("operator_add", O("operator_multiply", V("pos"), EMBED_DIM), O("operator_add", V("col"), 1))
                centered = O("operator_subtract", I(input_list, index), V("mean"))
                normalized = O("operator_divide", centered, V("stddev"))
                gamma_value = O("operator_multiply", read_weight(tensor_prefix + ".weight", V("col")), weight_scale(tensor_prefix + ".weight", V("col")))
                beta_value = O("operator_multiply", read_weight(tensor_prefix + ".bias", V("col")), weight_scale(tensor_prefix + ".bias", V("col")))
                result = O("operator_add", O("operator_multiply", normalized, gamma_value), beta_value)
                replace(normalize_chain, output_list, index, result)
                change(normalize_chain, "col")

            repeat(position_chain, EMBED_DIM, normalize)
            change(position_chain, "pos")

        repeat(chain, position_count, each_position)

    def residual(chain, position_count=None):
        set_value(chain, "row", 0)
        if position_count is None:
            position_count = O("operator_multiply", L("tokens"), EMBED_DIM)

        def each_value(value_chain, _):
            index = O("operator_add", V("row"), 1)
            replace(value_chain, "x", index, O("operator_add", I("x", index), I("norm", index)))
            change(value_chain, "row")

        repeat(chain, position_count, each_value)

    def gelu_hidden(chain, position_count=None):
        set_value(chain, "row", 0)
        if position_count is None:
            position_count = O("operator_multiply", L("tokens"), FFN_DIM)

        def activate(activation_chain, _):
            index = O("operator_add", V("row"), 1)
            set_value(activation_chain, "tempValue", I("hidden", index))
            square = O("operator_multiply", V("tempValue"), V("tempValue"))
            cube = O("operator_multiply", square, V("tempValue"))
            inner = O("operator_multiply", O(
                "operator_add", V("tempValue"), O("operator_multiply", cube, 0.044715),
            ), math.sqrt(2 / math.pi))
            exponential = M("e ^", O("operator_multiply", inner, 2))
            tanh = O("operator_divide", O("operator_subtract", exponential, 1), O("operator_add", exponential, 1))
            result = O("operator_multiply", O("operator_multiply", 0.5, V("tempValue")), O("operator_add", 1, tanh))
            replace(activation_chain, "hidden", index, result)
            change(activation_chain, "row")

        repeat(chain, position_count, activate)

    def initialize_embeddings(chain):
        embedding = tensor_info["token_embedding.weight"]
        positions = tensor_info["position_embedding.weight"]
        set_value(chain, "pos", 0)

        def each_position(position_chain, _):
            set_value(position_chain, "tokenValue", I("tokens", O("operator_add", V("pos"), 1)))
            set_value(position_chain, "col", 0)

            def each_column(column_chain, _):
                token_index = O("operator_add", O("operator_multiply", V("tokenValue"), EMBED_DIM), O("operator_add", V("col"), 1))
                pos_index = O("operator_add", O("operator_multiply", V("pos"), EMBED_DIM), O("operator_add", V("col"), 1))
                token_index = O(
                    "operator_add", O("operator_multiply", V("tokenValue"), EMBED_DIM), V("col"),
                )
                position_index = O(
                    "operator_add", O("operator_multiply", V("pos"), EMBED_DIM), V("col"),
                )
                token_value = O("operator_multiply", read_weight("token_embedding.weight", token_index), tensor_scale("token_embedding.weight", token_index))
                position_value = O("operator_multiply", read_weight("position_embedding.weight", position_index), tensor_scale("position_embedding.weight", position_index))
                replace(column_chain, "x", pos_index, O("operator_add", token_value, position_value))
                change(column_chain, "col")

            repeat(position_chain, EMBED_DIM, each_column)
            change(position_chain, "pos")

        repeat(chain, L("tokens"), each_position)

    def attention(chain, prefix):
        set_value(chain, "query", 0)

        def each_query(query_chain, _):
            set_value(query_chain, "head", 0)

            def each_head(head_chain, _):
                set_value(head_chain, "key", 0)

                def compute_score(score_chain, _):
                    set_value(score_chain, "acc", 0)
                    set_value(score_chain, "feature", 0)

                    def score_feature(feature_chain, _):
                        channel = O("operator_add", O("operator_multiply", V("head"), EMBED_DIM // NUM_HEADS), V("feature"))
                        q_index = O("operator_add", O("operator_multiply", V("query"), 3 * EMBED_DIM), O("operator_add", channel, 1))
                        k_index = O("operator_add", O("operator_multiply", V("key"), 3 * EMBED_DIM), O("operator_add", EMBED_DIM + 1, channel))
                        change(feature_chain, "acc", O("operator_multiply", I("qkv", q_index), I("qkv", k_index)))
                        change(feature_chain, "feature")

                    repeat(score_chain, EMBED_DIM // NUM_HEADS, score_feature)
                    score_index = O("operator_add", V("key"), 1)
                    score = O("operator_divide", V("acc"), math.sqrt(EMBED_DIM // NUM_HEADS))
                    replace(score_chain, "scores", score_index, score)
                    change(score_chain, "key")

                repeat(head_chain, O("operator_add", V("query"), 1), compute_score)
                set_value(head_chain, "maxScore", -1_000_000_000)
                set_value(head_chain, "key", 0)

                def find_max(max_chain, _):
                    score = I("scores", O("operator_add", V("key"), 1))

                    def update_max(update_chain, _):
                        set_value(update_chain, "maxScore", score)

                    if_block(max_chain, O("operator_gt", score, V("maxScore")), update_max)
                    change(max_chain, "key")

                repeat(head_chain, O("operator_add", V("query"), 1), find_max)
                set_value(head_chain, "sumExp", 0)
                set_value(head_chain, "key", 0)

                def exponentiate(exp_chain, _):
                    score_index = O("operator_add", V("key"), 1)
                    exponential = M("e ^", O("operator_subtract", I("scores", score_index), V("maxScore")))
                    replace(exp_chain, "scores", score_index, exponential)
                    change(exp_chain, "sumExp", exponential)
                    change(exp_chain, "key")

                repeat(head_chain, O("operator_add", V("query"), 1), exponentiate)
                set_value(head_chain, "key", 0)

                def normalize_score(norm_chain, _):
                    index = O("operator_add", V("key"), 1)
                    replace(norm_chain, "scores", index, O("operator_divide", I("scores", index), V("sumExp")))
                    change(norm_chain, "key")

                repeat(head_chain, O("operator_add", V("query"), 1), normalize_score)
                set_value(head_chain, "feature", 0)

                def each_output_feature(output_chain, _):
                    channel = O("operator_add", O("operator_multiply", V("head"), EMBED_DIM // NUM_HEADS), V("feature"))
                    set_value(output_chain, "acc", 0)
                    set_value(output_chain, "key", 0)

                    def weight_value(value_chain, _):
                        score_index = O("operator_add", V("key"), 1)
                        value_index = O("operator_add", O("operator_multiply", V("key"), 3 * EMBED_DIM), O("operator_add", 2 * EMBED_DIM + 1, channel))
                        change(value_chain, "acc", O("operator_multiply", I("scores", score_index), I("qkv", value_index)))
                        change(value_chain, "key")

                    repeat(output_chain, O("operator_add", V("query"), 1), weight_value)
                    output_index = O("operator_add", O("operator_multiply", V("query"), EMBED_DIM), O("operator_add", channel, 1))
                    replace(output_chain, "attention", output_index, V("acc"))
                    change(output_chain, "feature")

                repeat(head_chain, EMBED_DIM // NUM_HEADS, each_output_feature)
                change(head_chain, "head")

            repeat(query_chain, NUM_HEADS, each_head)
            change(query_chain, "query")

        repeat(chain, L("tokens"), each_query)

    def cached_attention(chain, layer):
        cache_base = layer * CONTEXT_LENGTH * EMBED_DIM
        set_value(chain, "head", 0)

        def each_head(head_chain, _):
            set_value(head_chain, "key", 0)

            def calculate_score(score_chain, _):
                set_value(score_chain, "acc", 0)
                set_value(score_chain, "feature", 0)

                def score_feature(feature_chain, _):
                    channel = O("operator_add", O("operator_multiply", V("head"), EMBED_DIM // NUM_HEADS), V("feature"))
                    query_value = I("qkv", O("operator_add", channel, 1))
                    key_index = O(
                        "operator_add", cache_base + 1,
                        O("operator_add", O("operator_multiply", V("key"), EMBED_DIM), channel),
                    )
                    change(feature_chain, "acc", O("operator_multiply", query_value, I("keyCache", key_index)))
                    change(feature_chain, "feature")

                repeat(score_chain, EMBED_DIM // NUM_HEADS, score_feature)
                score_index = O("operator_add", V("key"), 1)
                scaled_score = O("operator_divide", V("acc"), math.sqrt(EMBED_DIM // NUM_HEADS))
                replace(score_chain, "scores", score_index, scaled_score)
                change(score_chain, "key")

            repeat(head_chain, O("operator_add", V("cacheLen"), 1), calculate_score)
            set_value(head_chain, "maxScore", -1_000_000_000)
            set_value(head_chain, "key", 0)

            def find_max(max_chain, _):
                score = I("scores", O("operator_add", V("key"), 1))

                def update_max(update_chain, _):
                    set_value(update_chain, "maxScore", score)

                if_block(max_chain, O("operator_gt", score, V("maxScore")), update_max)
                change(max_chain, "key")

            repeat(head_chain, O("operator_add", V("cacheLen"), 1), find_max)
            set_value(head_chain, "sumExp", 0)
            set_value(head_chain, "key", 0)

            def exponentiate(exp_chain, _):
                index = O("operator_add", V("key"), 1)
                exponential = M("e ^", O("operator_subtract", I("scores", index), V("maxScore")))
                replace(exp_chain, "scores", index, exponential)
                change(exp_chain, "sumExp", exponential)
                change(exp_chain, "key")

            repeat(head_chain, O("operator_add", V("cacheLen"), 1), exponentiate)
            set_value(head_chain, "feature", 0)

            def calculate_output(output_chain, _):
                channel = O("operator_add", O("operator_multiply", V("head"), EMBED_DIM // NUM_HEADS), V("feature"))
                set_value(output_chain, "acc", 0)
                set_value(output_chain, "key", 0)

                def sum_values(value_chain, _):
                    score_index = O("operator_add", V("key"), 1)
                    value_index = O(
                        "operator_add", cache_base + 1,
                        O("operator_add", O("operator_multiply", V("key"), EMBED_DIM), channel),
                    )
                    change(value_chain, "acc", O("operator_multiply", I("scores", score_index), I("valueCache", value_index)))
                    change(value_chain, "key")

                repeat(output_chain, O("operator_add", V("cacheLen"), 1), sum_values)
                replace(output_chain, "attention", O("operator_add", channel, 1), V("acc"))
                change(output_chain, "feature")

            repeat(head_chain, EMBED_DIM // NUM_HEADS, calculate_output)
            change(head_chain, "head")

        repeat(chain, NUM_HEADS, each_head)

    def forward_token(chain, phase, step_variable):
        embedding = tensor_info["token_embedding.weight"]
        positions = tensor_info["position_embedding.weight"]
        set_value(chain, "col", 0)

        def add_embedding(embedding_chain, _):
            token_index = O("operator_add", O("operator_multiply", V("tokenValue"), EMBED_DIM), V("col"))
            position_index = O("operator_add", O("operator_multiply", V("cacheLen"), EMBED_DIM), V("col"))
            token_value = O("operator_multiply", read_weight("token_embedding.weight", token_index), tensor_scale("token_embedding.weight", token_index))
            position_value = O("operator_multiply", read_weight("position_embedding.weight", position_index), tensor_scale("position_embedding.weight", position_index))
            replace(embedding_chain, "x", O("operator_add", V("col"), 1), O("operator_add", token_value, position_value))
            change(embedding_chain, "col")

        repeat(chain, EMBED_DIM, add_embedding)

        for layer in range(NUM_LAYERS):
            set_value(chain, "progressText", O(
                "operator_join",
                O("operator_join", code.text(f"{phase} "), V(step_variable)),
                code.text(f", layer {layer + 1} of {NUM_LAYERS}"),
            ))
            if phase == "Generation token":
                completed = O(
                    "operator_add",
                    O("operator_subtract", V("generationStep"), 1),
                    (layer + 1) / NUM_LAYERS,
                )
                set_value(chain, "generationProgress", O(
                    "operator_multiply",
                    O("operator_divide", completed, GENERATION_TOKENS),
                    100,
                ))
                update_progress_bar(chain)
            prefix = f"blocks.{layer}."
            layer_norm(chain, "x", "norm", prefix + "ln1", 1)
            linear(
                chain, prefix + "attention.qkv.weight", prefix + "attention.qkv.bias",
                "norm", "qkv", EMBED_DIM, 3 * EMBED_DIM, 3 * EMBED_DIM,
            )

            key_cache_base = layer * CONTEXT_LENGTH * EMBED_DIM
            set_value(chain, "col", 0)

            def cache_key_value(cache_chain, _):
                position_offset = O("operator_multiply", V("cacheLen"), EMBED_DIM)
                cache_index = O("operator_add", key_cache_base + 1, O("operator_add", position_offset, V("col")))
                key_value = I("qkv", O("operator_add", EMBED_DIM + 1, V("col")))
                value_value = I("qkv", O("operator_add", 2 * EMBED_DIM + 1, V("col")))
                replace(cache_chain, "keyCache", cache_index, key_value)
                replace(cache_chain, "valueCache", O("operator_add", key_cache_base + 1, O("operator_add", position_offset, V("col"))), value_value)
                change(cache_chain, "col")

            repeat(chain, EMBED_DIM, cache_key_value)
            cached_attention(chain, layer)
            linear(
                chain, prefix + "attention.proj.weight", prefix + "attention.proj.bias",
                "attention", "norm", EMBED_DIM, EMBED_DIM, EMBED_DIM,
            )
            residual(chain, EMBED_DIM)
            layer_norm(chain, "x", "norm", prefix + "ln2", 1)
            linear(
                chain, prefix + "ffn.net.0.weight", prefix + "ffn.net.0.bias",
                "norm", "hidden", EMBED_DIM, FFN_DIM, FFN_DIM,
            )
            gelu_hidden(chain, FFN_DIM)
            linear(
                chain, prefix + "ffn.net.2.weight", prefix + "ffn.net.2.bias",
                "hidden", "norm", FFN_DIM, EMBED_DIM, EMBED_DIM,
            )
            residual(chain, EMBED_DIM)

        layer_norm(chain, "x", "norm", "ln_final", 1)
        linear(
            chain, "token_embedding.weight", None,
            "norm", "logits", EMBED_DIM, OUTPUT_VOCAB_SIZE, OUTPUT_VOCAB_SIZE,
        )

    def generate_token(generation_chain, _):
        change(generation_chain, "generationStep")
        set_value(generation_chain, "maxScore", -1_000_000_000)
        set_value(generation_chain, "candidate", 2)

        def find_max(candidate_chain, _):
            candidate = V("candidate")
            score = I("logits", O("operator_add", candidate, 1))

            def update_max(update_chain, _):
                set_value(update_chain, "maxScore", score)

            if_block(candidate_chain, O("operator_gt", score, V("maxScore")), update_max)
            change(candidate_chain, "candidate")

        repeat(generation_chain, OUTPUT_VOCAB_SIZE - 2, find_max)
        set_value(generation_chain, "sumExp", 0)
        set_value(generation_chain, "candidate", 2)

        def calculate_probability(candidate_chain, _):
            candidate = V("candidate")
            index = O("operator_add", candidate, 1)
            score = I("logits", index)
            probability = M(
                "e ^", O("operator_divide", O("operator_subtract", score, V("maxScore")), TEMPERATURE),
            )
            code.replace_item(candidate_chain, "probabilities", index, probability)
            change(candidate_chain, "sumExp", probability)
            change(candidate_chain, "candidate")

        repeat(generation_chain, OUTPUT_VOCAB_SIZE - 2, calculate_probability)
        set_value(generation_chain, "sampleThreshold", O("operator_random", 0, V("sumExp")))
        set_value(generation_chain, "cumulativeProbability", 0)
        set_value(generation_chain, "selected", 0)
        set_value(generation_chain, "bestToken", 2)
        set_value(generation_chain, "candidate", 2)

        def sample_candidate(candidate_chain, _):
            def accumulate(eligible_chain, _):
                index = O("operator_add", V("candidate"), 1)
                change(eligible_chain, "cumulativeProbability", I("probabilities", index))
                crossed_threshold = code.unary(
                    "operator_not",
                    O("operator_lt", V("cumulativeProbability"), V("sampleThreshold")),
                )

                def choose(choose_chain, _):
                    set_value(choose_chain, "bestToken", V("candidate"))
                    set_value(choose_chain, "selected", 1)

                if_block(eligible_chain, crossed_threshold, choose)

            if_block(candidate_chain, O("operator_equals", V("selected"), 0), accumulate)
            change(candidate_chain, "candidate")

        repeat(generation_chain, OUTPUT_VOCAB_SIZE - 2, sample_candidate)
        story_word = I("vocab", O("operator_add", V("bestToken"), 1))
        set_value(generation_chain, "storyText", O(
            "operator_join",
            O("operator_join", V("storyText"), code.text(" ")),
            story_word,
        ))
        set_value(generation_chain, "tokenA", V("tokenB"))
        set_value(generation_chain, "tokenB", V("bestToken"))
        set_value(generation_chain, "tokenValue", V("bestToken"))
        forward_token(generation_chain, "Generation token", "generationStep")
        change(generation_chain, "cacheLen")
        set_value(generation_chain, "progressText", O(
            "operator_join",
            O("operator_join", code.text("Generated "), V("generationStep")),
            code.text(f" of {GENERATION_TOKENS} tokens"),
        ))

    set_value(chain, "generationProgress", 0)
    set_value(chain, "progressBar", code.text("\u2591" * 20))
    set_value(chain, "cacheLen", 0)
    set_value(chain, "prefillIndex", 0)

    def prefill_token(prefill_chain, _):
        set_value(prefill_chain, "tokenValue", I("tokens", O("operator_add", V("prefillIndex"), 1)))
        forward_token(prefill_chain, "Prefill token", "prefillIndex")
        change(prefill_chain, "cacheLen")
        change(prefill_chain, "prefillIndex")

    set_value(chain, "progressText", code.text("Prefilling prompt"))
    repeat(chain, L("tokens"), prefill_token)
    set_value(chain, "progressText", code.text("Prompt cached"))
    set_value(chain, "generationStep", 0)
    set_value(chain, "progressText", code.text("Prompt parsed"))
    repeat(chain, GENERATION_TOKENS, generate_token)
    say = command(chain, "looks_say")
    code.connect(say, "MESSAGE", V("storyText"))
    set_value(chain, "progressText", code.text("Story complete"))

    stage_svg = b'<svg xmlns="http://www.w3.org/2000/svg" width="480" height="360"><rect width="480" height="360" fill="#fff"/><rect x="60" y="30" width="360" height="14" rx="3" fill="#d9e2ec"/></svg>'
    sprite_svg = b'<svg xmlns="http://www.w3.org/2000/svg" width="1" height="1"><rect width="1" height="1" fill="none"/></svg>'
    progress_svg = b'<svg xmlns="http://www.w3.org/2000/svg" width="360" height="14"><rect width="360" height="14" rx="3" fill="#168a5b"/></svg>'
    stage = {
        "isStage": True, "name": "Stage", "variables": {
            builder.variable_ids["generationProgress"]: ["generationProgress", 0],
        }, "lists": {}, "broadcasts": {},
        "blocks": {}, "comments": {}, "currentCostume": 0,
        "costumes": [costume(stage_svg, "backdrop1", 240, 180)], "sounds": [], "volume": 100,
        "layerOrder": 0, "tempo": 60, "videoTransparency": 50, "videoState": "on",
        "textToSpeechLanguage": None,
    }
    bar_event = stable_id("progress-bar:flag")
    bar_goto = stable_id("progress-bar:goto")
    bar_size = stable_id("progress-bar:initial-size")
    bar_show = stable_id("progress-bar:show")
    bar_forever = stable_id("progress-bar:forever")
    bar_update = stable_id("progress-bar:update-size")
    bar_wait = stable_id("progress-bar:wait")
    bar_reporter = stable_id("progress-bar:read-progress")
    progress_blocks = {
        bar_event: {
            "opcode": "event_whenflagclicked", "next": bar_goto, "parent": None,
            "inputs": {}, "fields": {}, "shadow": False, "topLevel": True,
            "x": 40, "y": 40,
        },
        bar_goto: {
            "opcode": "motion_gotoxy", "next": bar_size, "parent": bar_event,
            "inputs": {"X": [1, [4, "-180"]], "Y": [1, [4, "143"]]},
            "fields": {}, "shadow": False, "topLevel": False,
        },
        bar_size: {
            "opcode": "looks_setsizeto", "next": bar_show, "parent": bar_goto,
            "inputs": {"SIZE": [1, [4, "0"]]},
            "fields": {}, "shadow": False, "topLevel": False,
        },
        bar_show: {
            "opcode": "looks_show", "next": bar_forever, "parent": bar_size,
            "inputs": {}, "fields": {}, "shadow": False, "topLevel": False,
        },
        bar_forever: {
            "opcode": "control_forever", "next": None, "parent": bar_show,
            "inputs": {"SUBSTACK": [2, bar_update]},
            "fields": {}, "shadow": False, "topLevel": False,
        },
        bar_update: {
            "opcode": "looks_setsizeto", "next": bar_wait, "parent": bar_forever,
            "inputs": {"SIZE": [3, bar_reporter, [4, "0"]]},
            "fields": {}, "shadow": False, "topLevel": False,
        },
        bar_reporter: {
            "opcode": "data_variable", "next": None, "parent": bar_update,
            "inputs": {},
            "fields": {"VARIABLE": ["generationProgress", builder.variable_ids["generationProgress"]]},
            "shadow": False, "topLevel": False,
        },
        bar_wait: {
            "opcode": "control_wait", "next": None, "parent": bar_update,
            "inputs": {"DURATION": [1, [4, "0.1"]]},
            "fields": {}, "shadow": False, "topLevel": False,
        },
    }
    progress_sprite = {
        "isStage": False, "name": "Generation Progress", "variables": {}, "lists": {},
        "broadcasts": {}, "blocks": progress_blocks, "comments": {}, "currentCostume": 0,
        "costumes": [costume(progress_svg, "progress", 0, 7)], "sounds": [], "volume": 100,
        "layerOrder": 2, "visible": True, "x": -180, "y": 143, "size": 0,
        "direction": 90, "draggable": False, "rotationStyle": "all around",
    }
    list_values = {
        "weights": weights,
        "weightScales": weight_scales,
        "bitPowers": [1, 8, 64, 512, 4096, 32768, 262144, 2097152],
        "vocab": vocab,
        "tokens": [],
        "x": [0] * EMBED_DIM,
        "norm": [0] * max(FFN_DIM, EMBED_DIM),
        "qkv": [0] * (3 * EMBED_DIM),
        "attention": [0] * EMBED_DIM,
        "hidden": [0] * FFN_DIM,
        "logits": [0] * OUTPUT_VOCAB_SIZE,
        "probabilities": [0] * OUTPUT_VOCAB_SIZE,
        "scores": [0] * CONTEXT_LENGTH,
        "keyCache": [0] * (NUM_LAYERS * CONTEXT_LENGTH * EMBED_DIM),
        "valueCache": [0] * (NUM_LAYERS * CONTEXT_LENGTH * EMBED_DIM),
    }
    sprite = {
        "isStage": False, "name": "TinyGPT Scratch", "variables": {
            builder.variable_ids[name]: [name, "" if name in ScratchCode.STRING_VARIABLES else 0]
            for name in builder.variable_ids if name != "generationProgress"
        }, "lists": {
            builder.list_ids[name]: [name, values] for name, values in list_values.items()
        }, "broadcasts": {}, "blocks": builder.blocks, "comments": {}, "currentCostume": 0,
        "costumes": [costume(sprite_svg, "blank", 0.5, 0.5)], "sounds": [], "volume": 100,
        "layerOrder": 1, "visible": True, "x": 0, "y": 0, "size": 100,
        "direction": 90, "draggable": False, "rotationStyle": "all around",
    }
    project = {
        "targets": [stage, sprite, progress_sprite], "monitors": [{
            "id": builder.variable_ids["storyText"],
            "mode": "large",
            "opcode": "data_variable",
            "params": {"VARIABLE": builder.variable_ids["storyText"]},
            "spriteName": "TinyGPT Scratch",
            "value": "",
            "width": 480,
            "height": 120,
            "x": 5,
            "y": 5,
            "visible": True,
            "sliderMin": 0,
            "sliderMax": 100,
            "isDiscrete": True,
        }, {
            "id": builder.variable_ids["progressText"],
            "mode": "large",
            "opcode": "data_variable",
            "params": {"VARIABLE": builder.variable_ids["progressText"]},
            "spriteName": "TinyGPT Scratch",
            "value": "",
            "width": 300,
            "height": 40,
            "x": 5,
            "y": 130,
            "visible": True,
            "sliderMin": 0,
            "sliderMax": 100,
            "isDiscrete": True,
        }, {
            "id": builder.variable_ids["generationProgress"],
            "mode": "default",
            "opcode": "data_variable",
            "params": {"VARIABLE": builder.variable_ids["generationProgress"]},
            "spriteName": "Stage",
            "value": 0,
            "width": 120,
            "height": 30,
            "x": 370,
            "y": 20,
            "visible": True,
            "sliderMin": 0,
            "sliderMax": 100,
            "isDiscrete": False,
        }, {
            "id": builder.variable_ids["progressBar"],
            "mode": "large",
            "opcode": "data_variable",
            "params": {"VARIABLE": builder.variable_ids["progressBar"]},
            "spriteName": "TinyGPT Scratch",
            "value": "░" * 20,
            "width": 300,
            "height": 35,
            "x": 5,
            "y": 170,
            "visible": True,
            "sliderMin": 0,
            "sliderMax": 100,
            "isDiscrete": True,
        }], "extensions": [],
        "meta": {"semver": "3.0.0", "vm": "0.2.0", "agent": "native-tinygpt-5m"},
    }
    return project, stage_svg, sprite_svg, progress_svg, sum(tensor_info[name]["size"] for name in tensor_info)


def make_turbowarp_project(torch, tensors, vocab):
    import base64

    variable_ids = {
        name: stable_id("fast-var:" + name)
        for name in ("storyText", "progressText", "progressBar", "generationProgress")
    }
    tensor_info = {}
    encoded_chunks = []
    pending = b""
    offset = 0
    for name, tensor in tensors.items():
        flat = tensor.detach().cpu().float().reshape(-1)
        maximum = float(flat.abs().max().item()) if flat.numel() else 0.0
        scale = maximum / 127 if maximum else 1.0
        quantized = (flat / scale).round().clamp(-127, 127).to(dtype=torch.int8).contiguous()
        tensor_info[name] = {"offset": offset, "scale": scale, "size": quantized.numel()}
        offset += quantized.numel()
        for start in range(0, quantized.numel(), 65_535):
            raw = bytes(value & 255 for value in quantized[start:start + 65_535].tolist())
            chunk = pending + raw
            complete_size = len(chunk) - len(chunk) % 3
            if complete_size:
                encoded_chunks.append(base64.b64encode(chunk[:complete_size]).decode("ascii"))
            pending = chunk[complete_size:]
    if pending:
        encoded_chunks.append(base64.b64encode(pending).decode("ascii"))

    model_data = {
        "dimensions": {
            "embedding": EMBED_DIM,
            "heads": NUM_HEADS,
            "headWidth": EMBED_DIM // NUM_HEADS,
            "ffn": FFN_DIM,
            "layers": NUM_LAYERS,
            "context": CONTEXT_LENGTH,
            "defaultTokens": GENERATION_TOKENS,
            "temperature": TEMPERATURE,
        },
        "vocab": vocab,
        "tensors": tensor_info,
        "weights": "".join(encoded_chunks),
        "variables": variable_ids,
    }
    template = Path(__file__).with_name("turbowarp_extension.js").read_text(encoding="utf-8")
    extension_source = template.replace(
        "__MODEL_DATA__",
        json.dumps(model_data, separators=(",", ":"), ensure_ascii=False),
    )

    def bid(name):
        return stable_id("fast-block:" + name)

    sprite_svg = b'<svg xmlns="http://www.w3.org/2000/svg" width="1" height="1"><rect width="1" height="1" fill="none"/></svg>'
    flag, ask, say, generate, answer = (bid(name) for name in ("flag", "ask", "say", "generate", "answer"))
    model_sprite = {
        "isStage": False,
        "name": "TinyGPT Scratch",
        "variables": {
            variable_ids["storyText"]: ["storyText", ""],
            variable_ids["progressText"]: ["progressText", "Ready"],
            variable_ids["progressBar"]: ["progressBar", "░" * 20],
        },
        "lists": {},
        "broadcasts": {},
        "blocks": {
            flag: {
                "opcode": "event_whenflagclicked", "next": ask, "parent": None,
                "inputs": {}, "fields": {}, "shadow": False, "topLevel": True,
                "x": 35, "y": 35,
            },
            ask: {
                "opcode": "sensing_askandwait", "next": say, "parent": flag,
                "inputs": {"QUESTION": [1, [10, "Type a story prompt:"]]},
                "fields": {}, "shadow": False, "topLevel": False,
            },
            say: {
                "opcode": "looks_say", "next": None, "parent": ask,
                "inputs": {"MESSAGE": [3, generate, [10, ""]]},
                "fields": {}, "shadow": False, "topLevel": False,
            },
            generate: {
                "opcode": "tinygpt_generate", "next": None, "parent": say,
                "inputs": {
                    "PROMPT": [3, answer, [10, ""]],
                    "MAX_TOKENS": [1, [4, str(GENERATION_TOKENS)]],
                    "TEMPERATURE": [1, [4, str(TEMPERATURE)]],
                },
                "fields": {}, "shadow": False, "topLevel": False,
            },
            answer: {
                "opcode": "sensing_answer", "next": None, "parent": generate,
                "inputs": {}, "fields": {}, "shadow": True, "topLevel": False,
            },
        },
        "comments": {}, "currentCostume": 0,
        "costumes": [costume(sprite_svg, "blank", 0.5, 0.5)], "sounds": [], "volume": 100, "layerOrder": 1,
        "visible": True, "x": 0, "y": 0, "size": 100,
        "direction": 90, "draggable": False, "rotationStyle": "all around",
    }

    track_svg = b'<svg xmlns="http://www.w3.org/2000/svg" width="480" height="360"><rect width="480" height="360" fill="#fff"/><rect x="60" y="30" width="360" height="14" rx="3" fill="#d9e2ec"/></svg>'
    progress_svg = b'<svg xmlns="http://www.w3.org/2000/svg" width="360" height="14"><rect width="360" height="14" rx="3" fill="#168a5b"/></svg>'
    progress_flag, progress_goto, progress_size, progress_show, progress_forever, progress_update, progress_reporter, progress_wait = (
        bid(name) for name in ("bar-flag", "bar-goto", "bar-size", "bar-show", "bar-forever", "bar-update", "bar-value", "bar-wait")
    )
    progress_sprite = {
        "isStage": False, "name": "Generation Progress", "variables": {}, "lists": {},
        "broadcasts": {},
        "blocks": {
            progress_flag: {
                "opcode": "event_whenflagclicked", "next": progress_goto, "parent": None,
                "inputs": {}, "fields": {}, "shadow": False, "topLevel": True, "x": 30, "y": 30,
            },
            progress_goto: {
                "opcode": "motion_gotoxy", "next": progress_size, "parent": progress_flag,
                "inputs": {"X": [1, [4, "-180"]], "Y": [1, [4, "143"]]},
                "fields": {}, "shadow": False, "topLevel": False,
            },
            progress_size: {
                "opcode": "looks_setsizeto", "next": progress_show, "parent": progress_goto,
                "inputs": {"SIZE": [1, [4, "0"]]}, "fields": {}, "shadow": False, "topLevel": False,
            },
            progress_show: {
                "opcode": "looks_show", "next": progress_forever, "parent": progress_size,
                "inputs": {}, "fields": {}, "shadow": False, "topLevel": False,
            },
            progress_forever: {
                "opcode": "control_forever", "next": None, "parent": progress_show,
                "inputs": {"SUBSTACK": [2, progress_update]}, "fields": {}, "shadow": False, "topLevel": False,
            },
            progress_update: {
                "opcode": "looks_setsizeto", "next": progress_wait, "parent": progress_forever,
                "inputs": {"SIZE": [3, progress_reporter, [4, "0"]]},
                "fields": {}, "shadow": False, "topLevel": False,
            },
            progress_reporter: {
                "opcode": "data_variable", "next": None, "parent": progress_update,
                "inputs": {}, "fields": {"VARIABLE": ["generationProgress", variable_ids["generationProgress"]]},
                "shadow": False, "topLevel": False,
            },
            progress_wait: {
                "opcode": "control_wait", "next": None, "parent": progress_update,
                "inputs": {"DURATION": [1, [4, "0.1"]]}, "fields": {}, "shadow": False, "topLevel": False,
            },
        },
        "comments": {}, "currentCostume": 0,
        "costumes": [costume(progress_svg, "progress", 0, 7)],
        "sounds": [], "volume": 100, "layerOrder": 2,
        "visible": True, "x": -180, "y": 143, "size": 0,
        "direction": 90, "draggable": False, "rotationStyle": "all around",
    }
    stage = {
        "isStage": True, "name": "Stage",
        "variables": {variable_ids["generationProgress"]: ["generationProgress", 0]},
        "lists": {}, "broadcasts": {}, "blocks": {}, "comments": {}, "currentCostume": 0,
        "costumes": [costume(track_svg, "backdrop1", 240, 180)],
        "sounds": [], "volume": 100, "layerOrder": 0, "tempo": 60,
        "videoTransparency": 50, "videoState": "on", "textToSpeechLanguage": None,
    }
    monitors = []
    for name, mode, x, y, width, height, value in (
        ("storyText", "large", 5, 5, 480, 120, ""),
        ("progressText", "large", 5, 130, 300, 40, "Ready"),
        ("generationProgress", "default", 370, 20, 120, 30, 0),
        ("progressBar", "large", 5, 170, 300, 35, "░" * 20),
    ):
        variable_id = variable_ids[name]
        monitors.append({
            "id": variable_id, "mode": mode, "opcode": "data_variable",
            "params": {"VARIABLE": variable_id},
            "spriteName": "Stage" if name == "generationProgress" else "TinyGPT Scratch",
            "value": value, "width": width, "height": height, "x": x, "y": y,
            "visible": True, "sliderMin": 0, "sliderMax": 100,
            "isDiscrete": name != "generationProgress",
        })
    project = {
        "targets": [stage, model_sprite, progress_sprite],
        "monitors": monitors,
        "extensions": ["tinygpt"],
        "meta": {"semver": "3.0.0", "vm": "0.2.0", "agent": "turbowarp-tinygpt-fast"},
    }
    return project, track_svg, progress_svg, sprite_svg, extension_source, offset


def prune_vocab(vocab, size):
    ordered = [None] * len(vocab)
    for word, token_id in vocab.items():
        ordered[token_id] = word
    return ordered[:size]


def export(args):
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("PyTorch is required to read the .pt checkpoint. Install it in a Python environment supported by PyTorch.") from error

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    state = checkpoint.get("model", checkpoint)
    source_vocab = json.loads(Path(args.vocab).read_text(encoding="utf-8"))
    vocab = prune_vocab(source_vocab, VOCAB_SIZE)
    if len(vocab) != VOCAB_SIZE or any(word is None for word in vocab):
        raise ValueError(f"The source vocabulary must contain at least {VOCAB_SIZE} contiguous token IDs")
    tensors = build_state(torch, state)
    output_path = Path(args.output or ("tinygpt_scratch_compressed.sb3" if args.mode == "native" else "tinygpt_turbowarp.sb3"))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    compression_level = 9 if args.mode == "native" else 6
    if args.mode == "native":
        project, stage_svg, sprite_svg, progress_svg, parameter_count = make_full_native_project(tensors, vocab)
        extension_path = None
    else:
        project, stage_svg, progress_svg, sprite_svg, extension_source, parameter_count = make_turbowarp_project(torch, tensors, vocab)
        extension_path = Path(args.extension_output)
        extension_path.parent.mkdir(parents=True, exist_ok=True)
        extension_path.write_text(extension_source, encoding="utf-8")
    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED, compresslevel=compression_level) as archive:
        archive.writestr("project.json", json.dumps(project, separators=(",", ":"), ensure_ascii=False, allow_nan=False))
        archive.writestr(costume(stage_svg, "backdrop1", 240, 180)["md5ext"], stage_svg)
        archive.writestr(costume(sprite_svg, "blank", 0.5, 0.5)["md5ext"], sprite_svg)
        archive.writestr(costume(progress_svg, "progress", 0, 7)["md5ext"], progress_svg)
    project_json_size = len(json.dumps(project, separators=(",", ":"), ensure_ascii=False).encode("utf-8"))
    print(f"{'Native Scratch' if args.mode == 'native' else 'TurboWarp'} project: {output_path}")
    if extension_path:
        print(f"Custom extension: {extension_path}")
    print(f"Model: {parameter_count:,} unique parameters; {NUM_LAYERS} layers; full vocabulary ({len(vocab)} tokens)")
    print(f"project.json: {project_json_size:,} bytes; generated tokens: {GENERATION_TOKENS}")


def parse_args():
    parser = argparse.ArgumentParser(description="Export TinyGPT as native Scratch blocks or a fast TurboWarp project.")
    parser.add_argument("checkpoint", nargs="?", default="tinygpt.pt")
    parser.add_argument("--vocab", default="vocab.json")
    parser.add_argument("--mode", choices=("native", "turbowarp"), default="native")
    parser.add_argument("--output", default=None)
    parser.add_argument("--extension-output", default="tinygpt_turbowarp_extension.js")
    return parser.parse_args()


if __name__ == "__main__":
    try:
        export(parse_args())
    except (OSError, RuntimeError, ValueError, KeyError) as error:
        raise SystemExit(f"Export failed: {error}") from error