import struct

import tiktoken

import torch
from transformers import GPT2LMHeadModel

class GPT2:

    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints

        # create a from-scratch initialized minGPT model
        #config = GPTConfig(**config_args)
        #model = GPT(config)
        #sd = model.state_dict()
        #sd_keys = sd.keys()
        #sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        sd = {}
        
        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        #assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            #print(f"{k}: {sd_hf[k].shape}")
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                #assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    #sd[k].copy_(sd_hf[k].t())
                    sd[k] = (sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                #assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    #sd[k].copy_(sd_hf[k])
                    sd[k] = (sd_hf[k])
        
        return config_args, sd

def write_string(name, file):
    # Encode the string to bytes
    len_name = struct.pack('i', len(name))
    file.write(len_name)

    bin_name = name.encode('utf-8')
    file.write(bin_name)

def write_shape(shape, file):
    len_shape = struct.pack('i', len(shape))
    file.write(len_shape)
    
    for _ in shape:
        data = struct.pack('i', _)
        file.write(data)
    
def write_fp32(tensor, file):
    t = tensor.detach().cpu().to(torch.float32)
    b = t.numpy().tobytes()
    file.write(b)

def write_bf16(tensor, file):
    t = tensor.detach().cpu().to(torch.bfloat16)
    # numpy doesn't have bf16 datatype so we have to trick it
    t = t.view(torch.int16) # trick: reinterpret as int16
    b = t.numpy().tobytes()
    file.write(b)

def write_weights(model_weights, file_weights, dtype="fp32"):

    f = open(f"{file_weights}.{dtype}.v1.bin", "wb")
    write_string(f"dtype:{dtype}", f)
    
    for k,v in model_weights.items():

        write_string(k, f)
        write_shape(v.shape, f)

        if dtype=='fp32':
            write_fp32(v, f)
        elif dtype=='bf16':
            write_bf16(v, f)
            
    f.close()

def write_header(name, dims, f):
    print(f"writing {name} with {dims}")
    write_string(name, f)
    write_shape(dims, f)
    
def write_weights_gpt2(model_weights, file_weights, V, L, T, C, dtype="fp32"):

    f = open(f"{file_weights}.{dtype}.v2.bin", "wb")
    write_string(f"dtype:{dtype}", f)

    write_fun = write_fp32 #if dtype == "fp32" else write_bf16

    write_header("token-embedding", [V, C], f)
    write_fun(model_weights["transformer.wte.weight"], f) # (V, C)

    write_header("pos-embedding", [T, C], f)
    write_fun(model_weights["transformer.wpe.weight"], f) # (T, C)

    write_header("ln_1.weight", [L,C], f)
    for i in range(L): # (L, C)
        write_fun(model_weights[f"transformer.h.{i}.ln_1.weight"], f)

    write_header("ln_1.bias", [L,C], f)
    for i in range(L): # (L, C)
        write_fun(model_weights[f"transformer.h.{i}.ln_1.bias"], f)

    write_header("attn.qkv.weight", [L,3*C,C], f)
    for i in range(L): # (L, 3C, C)
        write_fun(model_weights[f"transformer.h.{i}.attn.c_attn.weight"], f)

    write_header("attn.qkv.bias", [L,3*C], f)
    for i in range(L): # (L, 3C)
        write_fun(model_weights[f"transformer.h.{i}.attn.c_attn.bias"], f)

    write_header("attn.proj.weight", [L,C,C], f)
    for i in range(L): # (L, C, C)
        write_fun(model_weights[f"transformer.h.{i}.attn.c_proj.weight"], f)

    write_header("attn.proj.bias", [L,C], f)
    for i in range(L): # (L, C)
        write_fun(model_weights[f"transformer.h.{i}.attn.c_proj.bias"], f)

    write_header("ln_2.weight", [L,C], f)
    for i in range(L): # (L, C)
        write_fun(model_weights[f"transformer.h.{i}.ln_2.weight"], f)

    write_header("ln_2.bias", [L,C], f)
    for i in range(L): # (L, C)
        write_fun(model_weights[f"transformer.h.{i}.ln_2.bias"], f)

    write_header("mlp.fc.weight", [L,4*C,C], f)
    for i in range(L): # (L, 4C, C)
        write_fun(model_weights[f"transformer.h.{i}.mlp.c_fc.weight"], f)

    write_header("mlp.fc.bias", [L,4*C], f)
    for i in range(L): # (L, 4C)
        write_fun(model_weights[f"transformer.h.{i}.mlp.c_fc.bias"], f)

    write_header("mlp.proj.weight", [L,C,4*C], f)
    for i in range(L): # (L, C, 4C)
        write_fun(model_weights[f"transformer.h.{i}.mlp.c_proj.weight"], f)

    write_header("mlp.proj.bias", [L,C], f)
    for i in range(L): # (L, C)
        write_fun(model_weights[f"transformer.h.{i}.mlp.c_proj.bias"], f)
    
    write_header("ln_f.weight", [C], f)
    write_fun(model_weights["transformer.ln_f.weight"], f) # (C, )
    
    write_header("ln_f.bias", [C], f)
    write_fun(model_weights["transformer.ln_f.bias"], f) # (C, )

    # write end-of-file ...
    eof = struct.pack('i', -1)
    f.write(eof)
    
    f.close()

def write_tokenizer_gpt2():
    
    # init (and write) the tokenizer
    enc = tiktoken.get_encoding("gpt2")
    #encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    #decode = lambda l: enc.decode(l)

    filename = "gpt2_tokenizer.bin"
        
    n = enc.max_token_value + 1

    header = torch.zeros(256, dtype=torch.int32)
    header[0] = 20240328 # magic
    header[1] = 2 # tokenizer version = 2 (1 -> 2: includes EOT token)
    header[2] = n # number of tokens
    header[3] = enc.eot_token # EOT token

    with open(filename, "wb") as file:
        #file.write(header.numpy().tobytes())
        for i in range(n):
            print(i, "\t", enc.decode_bytes([i]))
            
            if i>10:
                break
            
            """
            b = enc.decode_bytes([i])
            length = len(b)
            assert length < 256, f"Token length exceeds 255: {length}"
            file.write(struct.pack("<B", length))  # Write the length as a 1-byte unsigned integer
            file.write(b)  # Write the actual bytes
            """
            
    print(f"wrote {filename}")        

def test_tokenizer_gpt2():
    
    # init (and write) the tokenizer
    enc = tiktoken.get_encoding("gpt2")

    tokens = enc.encode("The old wise man was impressed with the sea [12-34].")
    for t in tokens:
        print(t, "\t", enc.decode_bytes([t]))
    
if __name__ == "__main__":
    
    model_name = "gpt2"
    
    config_args, model_weights = GPT2.from_pretrained(model_name)
    
    for k,v in model_weights.items():
        if len(v.shape)==2:
            print(k, ": ", v.shape, " => ", v[0,1])
        else:
            print(k, ": ", v.shape)
    
    V = config_args["vocab_size"]
    L = config_args["n_layer"]
    T = config_args["block_size"]
    C = config_args["n_embd"]

    #write_tokenizer_gpt2()
    
    #write_weights(model_weights, model_name, dtype="fp32")
    write_weights_gpt2(model_weights, model_name, V, L, T, C, dtype="fp32")

    test_tokenizer_gpt2()
    
    #
    #write_weights(model_weights, model_name, dtype="bf16")
