import requests
from time import sleep, time

num_request = 1000
true_undetected = 0
total_time = 0
total_size_bytes = 0
false_alarm = 0
num_success_requests = 0

for i in range(1, num_request + 1):
    start_time = time()
    elf_response = requests.get("https://dejy6ac3hi.execute-api.eu-west-1.amazonaws.com/dev/elf", proxies={
        'http': '127.0.0.1:8002',
        'https': '127.0.0.1:8002'
    }, verify=False)
    request_time = time() - start_time
    try:
        virus = elf_response.headers["Virus"]
        num_success_requests += 1
        total_time += request_time
    except KeyError:
        num_request += 1
        continue
    if virus == "undetected":
        total_size_bytes += len(elf_response.content)
        true_undetected += 1
    elif virus.lower() != "undefined":
        print("False alarm:", virus.lower())
        false_alarm += 1

    print(f"-----request{i}-----")
    print("Average size:", total_size_bytes / num_success_requests)
    print("True undetected:", true_undetected)
    print("Average time:", total_time / num_success_requests)
    print("False alarms:", false_alarm)
    sleep(1)


print("Average size:", total_size_bytes / num_success_requests)
print("True undetected:", true_undetected)
print("Average time:", total_time / num_success_requests)

with open("elf_undetected_statistics.txt", "w+") as stats:
    stats.write(f"True undetected: {true_undetected}\n"
                f"Average time: {total_time / num_success_requests}\n"
                f"Average size: {total_size_bytes / num_success_requests}\n"
                f"False alarms: {false_alarm}")
