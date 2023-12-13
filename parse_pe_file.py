import os

from capstone import *
import pefile
import re
import random
import pandas as pd


def parse_pe(file_name):
    pe = pefile.PE(file_name)

    entry_point = pe.OPTIONAL_HEADER.AddressOfEntryPoint
    data = pe.get_memory_mapped_image()[entry_point:]

    cs = Cs(CS_ARCH_X86, CS_MODE_32)

    return [re.sub("0x[0-9a-zA-Z]{4,8}", '', f"{i.mnemonic} {i.op_str}") for i in cs.disasm(data, 0x1000)]


if __name__ == '__main__':
    # create csv with parsed folder
    result = []
    for file in os.listdir('exes'):
        virus_type = random.choice(['Trojan:Win32', 'Backdoor:Win32', 'PUA:Win32'])
        try:
            parsed_pe = parse_pe(f'exes/{file}')
        except Exception as e:
            print(str(e))
            continue
        result.append([parsed_pe, virus_type])

    for file in os.listdir('dlls-safe')[:300]:
        try:
            parsed_pe = parse_pe(f'dlls-safe/{file}')
        except Exception as e:
            print(str(e))
            continue
        result.append([parsed_pe, 'not a virus'])

    for file in os.listdir('exes-safe'):
        try:
            parsed_pe = parse_pe(f'exes-safe/{file}')
        except Exception as e:
            print("EXE FAILED TO PARSE", str(e))
            continue
        result.append([parsed_pe, 'not a virus'])

    result_df = pd.DataFrame(result)
    result_df.to_csv('pe_files_random.csv')
