import subprocess

code = "python print.py"

# exec(open('/home/younghun/IoT/old_code/print.py').read())

file_addr = "rladu/rl"
result = subprocess.run(['python', '/home/younghun/IoT/old_code/print.py', '--volume_path', file_addr], check = True, capture_output=True)
output = result.stdout.decode('utf-8')
exit_status = result.returncode

print("Check: " , output)
print(exit_status)