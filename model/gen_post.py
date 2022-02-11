from utils import generate

def gen(gen_num=100, top=10, device=None, path='../saved_model'):
    contents = []
    for _ in range(gen_num):
        contents.append(generate(device=device, path=path)['generated_text'])
    contents = sorted(contents, key=len)[: top]

def post(text):
    pass

def gen_post(gen_num=100, top=10, device=None, path='../saved_model'):
    contents = gen(gen_num, top, device, path)
    for line in contents:
        post(line)



