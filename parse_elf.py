from io import BytesIO
from elftools.elf.elffile import ELFFile
from capstone import *
import re
import os
import shutil


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


for file in os.listdir("/bin"):
    try:
        if os.path.getsize(f"/bin/{file}") / 1024 / 1024 < 10:
            with open(f"/bin/{file}", "rb") as f:
                elf = parse_elf(f.read())
            shutil.copyfile(f"/bin/{file}", f"elfs-from-bin-safe/{file}")
    except Exception as e:
        print(e)
