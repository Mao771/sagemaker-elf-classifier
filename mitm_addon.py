from mitmproxy import http
import requests
from io import BytesIO
from elftools.elf.elffile import ELFFile
from capstone import *
import re
import json


def parse_elf(elf_bytes):
    elf = ELFFile(BytesIO(elf_bytes))
    code = elf.get_section_by_name('.rodata')
    ops = code.data()
    addr = code['sh_addr']
    md = Cs(CS_ARCH_X86, CS_MODE_64)
    # original_commands = [f"{i.mnemonic} {i.op_str}" for i in md.disasm(ops, addr)]
    processed_commands = [re.sub("0x[0-9a-zA-Z]{4,8}", '', f"{i.mnemonic} {i.op_str}") for i
                          in md.disasm(ops, addr)]
    processed_commands_replaced = [
        command.replace("add dword ptr [rax], eax add al, byte ptr [rax]", "") for command in processed_commands
    ]

    return "".join(processed_commands_replaced)


def responseheaders(flow: http.HTTPFlow):

    flow.response.stream = False


def disasm_raw_content(raw_content, url: str, flow_name: str):
    try:
        print("Parsing elf")
        disasm_file = parse_elf(raw_content)
    except Exception as e:
        print(e)
        print("Exception parsing elf")
        return "Undefined"

    sagemaker_response = requests.post("http://127.0.0.1:8080/invocations", json={"files": [[disasm_file]]})
    return json.loads(sagemaker_response.text)['prediction'][0]


def response(flow: http.HTTPFlow):
    print("Intercepting!!!!")

    if '/invocations' in flow.request.pretty_url:
        print("Its API request")
        return

    if flow.response:
        print("Headers")
        print(flow.request.headers)
        result = disasm_raw_content(flow.response.raw_content, flow.request.pretty_url, 'response')
        print()
        flow.response.headers["virus"] = result
